"""The Python code in README.md and the docs must still match the code.

Fast and offline (part of CI). Every ```python block in README.md and docs/src/content/docs/**/*.md(x) is
  * compiled,
  * checked for its epftoolbox2 imports, which are executed (a renamed or removed name fails), and
  * checked call by call: the arguments of every epftoolbox2 class, function or method the block calls are bound
    against the real signature, so a renamed, removed or missing parameter fails. Calls are resolved from the
    block's imports, and a call on a variable is resolved when the variable was assigned from an epftoolbox2 class
    or from a method whose return annotation names one (`ModelPipeline(...).run(...)` is an EvaluationReport).
    Blocks of one page run in order and share their imports and variables, as the page reads.
The class stubs in reference/api.md are not code to run: they are checked against the real signatures instead
(every real parameter documented, in the real order, with the real literal defaults).

A block that is deliberately not runnable Python opts out with a meta string on its fence, which Starlight ignores
when it renders the block:

    ```python nocheck
    ...
    ```

Use it sparingly and prefer fixing the block. MAX_OPT_OUTS caps how many there are.

Live (also marked `integration` and `examples`, so CI skips them): the quick start of the README and of the docs
are executed block by block against the real APIs, see the end of this file.
"""

import ast
import datetime as dt
import functools
import importlib
import inspect
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from tests.live import (
    DATA_END,
    DATA_START,
    HORIZON,
    PRICE_COLUMNS,
    RUN_DAYS,
    TEST_END,
    TEST_START,
    assert_data_frame,
    assert_excel,
    assert_forecasts,
    assert_information_set_checked,
    assert_metrics,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "src" / "content" / "docs"
PAGES = [ROOT / "README.md", *sorted(DOCS.rglob("*.md")), *sorted(DOCS.rglob("*.mdx"))]

OPT_OUT = "nocheck"
MAX_OPT_OUTS = 0
PUBLIC_PACKAGES = (
    "epftoolbox2",
    "epftoolbox2.pipelines",
    "epftoolbox2.data.sources",
    "epftoolbox2.data.transformers",
    "epftoolbox2.data.validators",
    "epftoolbox2.models",
    "epftoolbox2.scalers",
    "epftoolbox2.evaluators",
    "epftoolbox2.exporters",
    "epftoolbox2.results",
    "epftoolbox2.information_set",
    "epftoolbox2.presets",
)

FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$")


@dataclass(frozen=True)
class Block:
    path: Path
    line: int  # of the first line of code
    code: str
    meta: tuple  # the words after the language on the opening fence

    @property
    def where(self) -> str:
        return f"{self.path.relative_to(ROOT).as_posix()}:{self.line}"

    @property
    def opted_out(self) -> bool:
        return OPT_OUT in self.meta


def extract_blocks(path: Path) -> list:
    """The python code blocks of a Markdown or MDX file, de-indented when a component indents the fence."""
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        opening = FENCE.match(lines[i])
        if not opening:
            i += 1
            continue
        indent, fence = opening.group("indent"), opening.group("fence")
        words = opening.group("info").split()
        body = []
        i += 1
        while i < len(lines):
            closing = FENCE.match(lines[i])
            if (
                closing
                and closing.group("fence")[0] == fence[0]
                and len(closing.group("fence")) >= len(fence)
                and not closing.group("info").strip()
            ):
                break
            body.append(lines[i])
            i += 1
        i += 1
        if words and words[0].lower() in ("python", "py"):
            code = "\n".join(line[len(indent) :] if line.startswith(indent) else line.lstrip() for line in body)
            blocks.append(Block(path, i - len(body), code, tuple(words[1:])))
    return blocks


@functools.cache
def public_names() -> dict:
    """Every public class and function of the documented packages, by name."""
    found = {}
    for package in PUBLIC_PACKAGES:
        module = importlib.import_module(package)
        for name in getattr(module, "__all__", None) or dir(module):
            obj = getattr(module, name, None)
            if (inspect.isclass(obj) or inspect.isfunction(obj)) and getattr(obj, "__module__", "").startswith(
                "epftoolbox2"
            ):
                found.setdefault(name, obj)
    return found


# ---------------------------------------------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------------------------------------------


def class_signature(cls: type) -> inspect.Signature:
    """The signature of `cls(...)`, with the keyword arguments that `**kwargs` forwards to the parents added.

    `BaseModel` subclasses take `name`, `scaler` and so on through `**kwargs`, so `inspect.signature` alone would
    call every one of those a mistake. The forwarded parameters are keyword-only and optional here.
    """
    own = inspect.signature(cls)
    params = list(own.parameters.values())
    if not any(p.kind is p.VAR_KEYWORD for p in params):
        return own
    names = {p.name for p in params}
    provider = next((k for k in cls.__mro__ if "__init__" in k.__dict__), object)
    still_open = True
    extra = []
    for klass in cls.__mro__[cls.__mro__.index(provider) + 1 :]:
        init = klass.__dict__.get("__init__")
        if klass is object or init is None:
            continue
        parent_open = False
        for p in list(inspect.signature(init).parameters.values())[1:]:
            if p.kind is p.VAR_KEYWORD:
                parent_open = True
            elif p.kind is not p.VAR_POSITIONAL and p.name not in names:
                names.add(p.name)
                extra.append(p.replace(kind=p.KEYWORD_ONLY, default=None if p.default is p.empty else p.default))
        if not parent_open:
            still_open = False
            break
    kept = [p for p in params if p.kind is not p.VAR_KEYWORD]
    tail = [p for p in params if p.kind is p.VAR_KEYWORD] if still_open else []
    return own.replace(parameters=sorted(kept + extra + tail, key=lambda p: p.kind))


def callable_signature(obj, owner=None):
    """The signature of a class, function or method as it is called, or None when it cannot be inspected.

    `owner` is the class the callable was looked up on: an instance method found there still has `self`.
    """
    try:
        if inspect.isclass(obj):
            return class_signature(obj)
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    if owner is not None and inspect.isfunction(obj):
        static = inspect.getattr_static(owner, obj.__name__, None)
        if not isinstance(static, (staticmethod, classmethod)):
            sig = sig.replace(parameters=list(sig.parameters.values())[1:])
    return sig


def bind_problem(sig: inspect.Signature, call: ast.Call):
    """Why `call` does not fit `sig`, or None. Arguments are bound by position and name, never by value."""
    params = sig.parameters
    accepted = [p.name for p in params.values() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    keywords = [k.arg for k in call.keywords if k.arg is not None]
    if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
        unknown = [name for name in keywords if name not in params or params[name].kind is params[name].POSITIONAL_ONLY]
        if unknown:
            return f"unexpected keyword argument(s) {unknown} (it accepts {accepted})"
    expanded = any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords)
    elided = any(isinstance(a, ast.Constant) and a.value is Ellipsis for a in call.args)
    args = [] if any(isinstance(a, ast.Starred) for a in call.args) else [None] * len(call.args)
    try:
        if expanded or elided:
            sig.bind_partial(*args, **dict.fromkeys(keywords))
        else:
            sig.bind(*args, **dict.fromkeys(keywords))
    except TypeError as exc:
        return f"{exc} (it accepts {accepted})"
    return None


# ---------------------------------------------------------------------------------------------------------------
# The checker: one per page, fed its blocks in order
# ---------------------------------------------------------------------------------------------------------------

MISSING = object()


def dotted(node):
    """['a', 'b', 'c'] for the expression a.b.c, or None when it is anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id, *reversed(parts)]
    return None


def is_epf(obj) -> bool:
    return getattr(obj, "__module__", "").startswith("epftoolbox2") or (
        inspect.ismodule(obj) and obj.__name__.startswith("epftoolbox2")
    )


class PageChecker:
    def __init__(self):
        self.namespace = {}  # what the page imported from epftoolbox2, by the name it is bound to
        # variable -> the epftoolbox2 class it holds an instance of. Pages show `report.summary()` without the run that
        # made the report, so `report` is an EvaluationReport unless the page assigns it another epftoolbox2 object.
        self.defaults = {"report": public_names()["EvaluationReport"]}
        self.types = dict(self.defaults)
        self.local = set()  # names the page defines itself, which shadow the public ones
        self.checked_calls = 0
        self.problems = []

    # -- imports ------------------------------------------------------------------------------------------------

    def run_imports(self, tree: ast.AST, where: str) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mentions = (node.module or "").split(".")[0] == "epftoolbox2" and node.level == 0
            elif isinstance(node, ast.Import):
                mentions = any(a.name.split(".")[0] == "epftoolbox2" for a in node.names)
            else:
                continue
            if not mentions:
                continue
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), where, "exec"), self.namespace)
            except Exception as exc:  # ImportError, AttributeError, or anything the import itself raises
                self.problems.append(f"line {node.lineno}: `{ast.unparse(node)}` fails: {type(exc).__name__}: {exc}")

    # -- types --------------------------------------------------------------------------------------------------

    def named(self, name: str):
        """What a bare name refers to: an epftoolbox2 import of the page, else the public class or function of that
        name (a snippet may leave out its import), else None."""
        if name in self.types:
            return None
        if name in self.namespace:
            return self.namespace[name]
        return None if name in self.local else public_names().get(name)

    def resolve(self, func: ast.expr):
        """(object, owner class or None) for the callee of a call, MISSING when it names something that is not
        there, or None when it cannot be resolved (a variable of unknown type, a call on a result, ...)."""
        chain = dotted(func)
        if chain and self.named(chain[0]) is not None:
            obj, owner = self.named(chain[0]), None
            for attr in chain[1:]:
                owner = obj if inspect.isclass(obj) else None
                obj = getattr(obj, attr, MISSING)
                if obj is MISSING:
                    return MISSING, owner
            return obj, owner
        if isinstance(func, ast.Attribute):
            cls = self.infer(func.value)
            if cls is not None:
                method = getattr(cls, func.attr, MISSING)
                if method is MISSING and not self.sets_attribute(cls, func.attr):
                    return MISSING, cls
                return (method, cls) if method is not MISSING else None
        return None

    @staticmethod
    def sets_attribute(cls: type, name: str) -> bool:
        """Whether an instance gets `name` as an attribute in __init__ (so `obj.name(...)` may be valid)."""
        try:
            return f"self.{name}" in inspect.getsource(cls.__init__)
        except (OSError, TypeError):
            return True

    def infer(self, expr: ast.expr):
        """The epftoolbox2 class `expr` is an instance of, or None when that is not known."""
        if isinstance(expr, ast.Name):
            return self.types.get(expr.id)
        if isinstance(expr, ast.Call):
            target = self.resolve(expr.func)
            if target is None or target[0] is MISSING:
                return None
            obj, owner = target
            if inspect.isclass(obj):
                return obj if is_epf(obj) else None
            return self.returned_class(obj, owner)
        return None

    @staticmethod
    def returned_class(func, owner):
        if not callable(func) or not is_epf(func):
            return None
        try:
            annotation = inspect.signature(func).return_annotation
        except (TypeError, ValueError):
            return None
        if isinstance(annotation, str):
            name = annotation.strip("'\"")
            module = sys.modules.get(getattr(func, "__module__", ""), None)
            annotation = getattr(module, name, None) or (owner if owner and owner.__name__ == name else None)
            annotation = annotation or public_names().get(name)
        return annotation if inspect.isclass(annotation) and is_epf(annotation) else None

    # -- calls --------------------------------------------------------------------------------------------------

    def check_block(self, tree: ast.Module, where: str) -> None:
        self.local |= {
            n.name for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        } | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        self.run_imports(tree, where)
        self.visit_statements(tree.body)

    def visit_statements(self, statements) -> None:
        for statement in statements:
            self.visit(statement)

    def visit(self, node: ast.AST) -> None:
        """Check the calls under `node` in source order and track the variables assigned from epftoolbox2 objects."""
        if isinstance(node, ast.Call):
            for child in ast.iter_child_nodes(node):
                self.visit(child)
            self.check_call(node)
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            self.visit(node.value)
            cls = self.infer(node.value)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    if cls is not None:
                        self.types[target.id] = cls
                    elif target.id in self.defaults:
                        self.types[target.id] = self.defaults[target.id]
                    else:
                        self.types.pop(target.id, None)
            return
        for child in ast.iter_child_nodes(node):
            self.visit(child)

    def check_call(self, call: ast.Call) -> None:
        target = self.resolve(call.func)
        if target is None:
            return
        obj, owner = target
        name = ast.unparse(call.func)
        if obj is MISSING:
            subject = owner.__name__ if owner is not None else name.rsplit(".", 1)[0]
            self.problems.append(f"line {call.lineno}: `{name}` does not exist ({subject} has no such name)")
            return
        if not is_epf(obj) or not callable(obj):
            return
        sig = callable_signature(obj, owner)
        if sig is None:
            return
        self.checked_calls += 1
        problem = bind_problem(sig, call)
        if problem:
            self.problems.append(f"line {call.lineno}: `{name}(...)`: {problem}")


def is_stub_class(node: ast.AST) -> bool:
    """A class whose methods are all `...`: a signature listing, as in reference/api.md."""
    if not isinstance(node, ast.ClassDef) or node.name not in public_names():
        return False
    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
    return bool(methods) and all(
        len(m.body) == 1 and isinstance(m.body[0], ast.Expr) and getattr(m.body[0].value, "value", 0) is Ellipsis
        for m in methods
    )


def check_stub_class(node: ast.ClassDef) -> list:
    """A class stub lists the real parameters of each method it shows, in the real order, with the real defaults."""
    real = public_names().get(node.name)
    if real is None or not inspect.isclass(real):
        return [f"line {node.lineno}: class {node.name} is not a public epftoolbox2 class"]
    problems = []
    for method in (n for n in node.body if isinstance(n, ast.FunctionDef)):
        if method.name == "__init__":
            sig = class_signature(real)
        else:
            attr = getattr(real, method.name, None)
            if attr is None:
                problems.append(f"line {method.lineno}: {node.name} has no method {method.name}")
                continue
            sig = callable_signature(attr, real)
            if sig is None:
                continue
        problems.extend(compare_stub_method(f"{node.name}.{method.name}", method, sig))
    return problems


def compare_stub_method(label: str, method: ast.FunctionDef, sig: inspect.Signature) -> list:
    problems = []
    real = sig.parameters
    order = [name for name, p in real.items() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    args = method.args
    documented = [a for a in args.posonlyargs + args.args + args.kwonlyargs if a.arg not in ("self", "cls")]
    defaults = {a.arg: d for a, d in zip(reversed(args.posonlyargs + args.args), reversed(args.defaults))}
    defaults.update({a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None})
    has_var_keyword = any(p.kind is p.VAR_KEYWORD for p in real.values())
    has_var_positional = any(p.kind is p.VAR_POSITIONAL for p in real.values())
    stub_has_var = args.kwarg is not None or args.vararg is not None

    for name in order:
        if name not in {a.arg for a in documented} and not stub_has_var:
            problems.append(f"line {method.lineno}: {label} does not document the parameter `{name}`")
    last = -1
    for arg in documented:
        if arg.arg not in real:
            if not has_var_keyword and not has_var_positional:
                problems.append(f"line {arg.lineno}: {label}({arg.arg}) is not a parameter (it has {order})")
            continue
        position = order.index(arg.arg)
        if position < last and real[arg.arg].kind is not inspect.Parameter.KEYWORD_ONLY:
            problems.append(f"line {arg.lineno}: {label}({arg.arg}) is out of order (the real order is {order})")
        last = max(last, position)
        real_default = real[arg.arg].default
        required = real_default is inspect.Parameter.empty
        if arg.arg not in defaults:
            if not required:
                problems.append(
                    f"line {arg.lineno}: {label}({arg.arg}) is documented as required, its default is {real_default!r}"
                )
            continue
        try:
            documented_default = ast.literal_eval(defaults[arg.arg])
        except ValueError:
            continue  # not a literal (a name or a call), so there is nothing to compare
        if required:
            problems.append(f"line {arg.lineno}: {label}({arg.arg}) has a default, the real parameter is required")
        elif as_list(real_default) != as_list(documented_default):
            problems.append(
                f"line {arg.lineno}: {label}({arg.arg}) defaults to {documented_default!r}, "
                f"the real default is {real_default!r}"
            )
    return problems


def as_list(value):
    """Lists and tuples compare equal, as a docs stub writes a tuple default as a list."""
    return list(value) if isinstance(value, (list, tuple)) else value


@dataclass
class BlockResult:
    block: Block
    problems: list
    checked_calls: int


def check_source(checker: PageChecker, code: str, where: str) -> list:
    """Problems of one block's code, fed to the page's checker so that later blocks see its imports and variables."""
    before = len(checker.problems)
    try:
        tree = ast.parse(code, filename=where)
        compile(tree, where, "exec")
    except SyntaxError as exc:
        checker.problems.append(f"line {exc.lineno}: does not compile: {exc.msg}")
    else:
        stubs = [n for n in tree.body if is_stub_class(n)]
        if stubs:
            for stub in stubs:
                checker.problems.extend(check_stub_class(stub))
        else:
            checker.check_block(tree, where)
    return checker.problems[before:]


@functools.cache
def analyse(path: Path) -> tuple:
    """Check every block of a page in order. Returns a BlockResult for each block that is not opted out."""
    checker = PageChecker()
    results = []
    for block in extract_blocks(path):
        if not block.opted_out:
            calls_before = checker.checked_calls
            problems = check_source(checker, block.code, block.where)
            results.append(BlockResult(block, problems, checker.checked_calls - calls_before))
    return tuple(results)


ALL_BLOCKS = [block for page in PAGES for block in extract_blocks(page)]
CHECKED = [result for page in PAGES for result in analyse(page)]


def test_the_extractor_finds_the_blocks():
    """A broken extractor would pass every check below on nothing."""
    assert ROOT / "README.md" in PAGES
    assert len(PAGES) > 40
    assert len(ALL_BLOCKS) > 150
    assert {b.path.name for b in ALL_BLOCKS} >= {"README.md", "quickstart.mdx", "api.md"}
    assert sum(r.checked_calls for r in CHECKED) > 500, "calls are no longer resolved against signatures"


def test_opt_outs_are_rare():
    opted_out = [b.where for b in ALL_BLOCKS if b.opted_out]
    assert len(opted_out) <= MAX_OPT_OUTS, (
        f"{len(opted_out)} blocks opt out with `{OPT_OUT}` (at most {MAX_OPT_OUTS}): {opted_out}. "
        "Fix the block instead, or raise MAX_OPT_OUTS in review."
    )


@pytest.mark.parametrize("result", CHECKED, ids=[r.block.where for r in CHECKED])
def test_block_matches_the_code(result):
    assert not result.problems, f"{result.block.where}\n" + "\n".join(result.problems)


# ---------------------------------------------------------------------------------------------------------------
# The checker itself: what it must flag and what it must let through
# ---------------------------------------------------------------------------------------------------------------

FLAGGED = [
    pytest.param(
        "from epftoolbox2.models import OLSModel\nOLSModel(predictors=['a'], trainning_window=3)",
        "trainning_window",
        id="renamed constructor keyword",
    ),
    pytest.param(
        "from epftoolbox2.models import LassoCVModel\nLassoCVModel(predictors=[], nam='x')",
        "nam",
        id="keyword that a parent class would take",
    ),
    pytest.param("from epftoolbox2.models import OlsModel", "OlsModel", id="renamed class"),
    pytest.param("from epftoolbox2.moddels import OLSModel", "moddels", id="renamed module"),
    pytest.param("import epftoolbox2\nepftoolbox2.verify(x=1)", "unexpected keyword", id="function keyword"),
    pytest.param(
        "from epftoolbox2.pipelines import DataPipeline\nDataPipeline().run(start='a', finish='b')",
        "finish",
        id="method of a fresh instance",
    ),
    pytest.param(
        "from epftoolbox2.pipelines import ModelPipeline\nModelPipeline().add_model(m).run(df, test_begin='x')",
        "test_begin",
        id="method after a builder chain",
    ),
    pytest.param(
        "from epftoolbox2.pipelines import ModelPipeline\nreport = ModelPipeline().run(df)\nreport.by_hours()",
        "by_hours",
        id="method of a result",
    ),
    pytest.param(
        "from epftoolbox2.pipelines import Workflow\nreport = Workflow.load('x').run(now=True)",
        "now",
        id="classmethod result",
    ),
    pytest.param(
        "from epftoolbox2.pipelines import DataPipeline\nDataPipeline().run('a', 'b', True, 1)",
        "too many positional",
        id="too many positional arguments",
    ),
    pytest.param(
        "from epftoolbox2.pipelines import DataPipeline\nDataPipeline().run(start='a')",
        "end",
        id="missing required argument",
    ),
    pytest.param(
        "from epftoolbox2.pipelines import DataPipeline\nDataPipeline().runn('a', 'b')", "runn", id="renamed method"
    ),
    pytest.param("OLSModel(predictors=[], bogus=1)", "bogus", id="snippet without any import"),
    pytest.param("EntsoeSource(country='PL')", "country", id="snippet that leaves out its import"),
    pytest.param("x = (", "compile", id="syntax error"),
]

LET_THROUGH = [
    pytest.param(
        "from epftoolbox2.models import LassoCVModel\nLassoCVModel(predictors=[], name='x', scaler='asinh', cv=3)",
        id="keywords forwarded to parent classes",
    ),
    pytest.param("from epftoolbox2.pipelines import Workflow\nWorkflow.load('x').run()", id="classmethod chain"),
    pytest.param("from epftoolbox2.data.sources import EntsoeSource\nEntsoeSource(...)", id="elided arguments"),
    pytest.param(
        "from epftoolbox2.pipelines import DataPipeline\nDataPipeline(timezone='UTC').add_source(s).run('a', 'b', True)",
        id="positional arguments",
    ),
    pytest.param("from epftoolbox2.pipelines import DataPipeline\nargs = {}\nDataPipeline(**args)", id="expansion"),
    pytest.param(
        "from epftoolbox2.pipelines import ModelPipeline\nreport = ModelPipeline().run(df)\nreport.summary()",
        id="method of a result",
    ),
    pytest.param(
        "class LassoCVModel:\n    def __init__(self, bogus):\n        self.bogus = bogus\nLassoCVModel(bogus=1)",
        id="the page's own class",
    ),
    pytest.param("import pandas as pd\npd.DataFrame(bogus=1)", id="not epftoolbox2"),
]


@pytest.mark.parametrize("code, expected", FLAGGED)
def test_the_checker_flags_a_wrong_block(code, expected):
    problems = check_source(PageChecker(), code, "snippet")
    assert any(expected in problem for problem in problems), problems


@pytest.mark.parametrize("code", LET_THROUGH)
def test_the_checker_accepts_a_right_block(code):
    assert check_source(PageChecker(), code, "snippet") == []


def test_blocks_of_a_page_share_their_imports_and_variables():
    checker = PageChecker()
    assert (
        check_source(checker, "from epftoolbox2.pipelines import ModelPipeline\npipeline = ModelPipeline()", "a") == []
    )
    assert check_source(checker, "report = pipeline.run(df, test_begin='x')", "b")
    assert check_source(checker, "report.by_hours()", "c")


def test_the_quick_start_blocks_are_found():
    """The live tests below step through these blocks one by one."""
    assert len(blocks_between(ROOT / "README.md", "## Quick Start", "## Examples")) == 8
    assert len(extract_blocks(DOCS / "getting-started" / "quickstart.mdx")) == 4


def test_a_live_edit_that_no_longer_matches_the_docs_fails_before_anything_runs():
    block = Block(ROOT / "README.md", 1, "x = 1", ())
    with pytest.raises(AssertionError, match="no longer contain"):
        PageRunner([block], [("y = 2", "y = 3")])


def test_the_runner_applies_its_edits_and_shares_one_namespace():
    first, second = Block(ROOT / "README.md", 1, "x = 1", ()), Block(ROOT / "README.md", 5, "y = x + 1", ())
    runner = PageRunner([first, second], [("x = 1", "x = 41")])
    runner.step()
    assert not runner.done
    runner.step()
    assert runner.done and runner.namespace["y"] == 42


def test_a_fence_meta_string_opts_a_block_out(tmp_path):
    page = tmp_path / "page.md"
    page.write_text("```python nocheck\nnot python (\n```\n\n```python\nx = 1\n```\n\n```bash\nls\n```\n")
    blocks = extract_blocks(page)
    assert [(b.opted_out, b.code) for b in blocks] == [(True, "not python ("), (False, "x = 1")]


def test_a_block_in_an_indented_component_is_de_indented(tmp_path):
    page = tmp_path / "page.mdx"
    page.write_text("<Steps>\n\n1. Step\n\n   ```python\n   if True:\n       x = 1\n   ```\n\n</Steps>\n")
    (block,) = extract_blocks(page)
    assert block.code == "if True:\n    x = 1"
    compile(block.code, "x", "exec")


# ---------------------------------------------------------------------------------------------------------------
# Live: the quick starts run for real
# ---------------------------------------------------------------------------------------------------------------


def blocks_between(path: Path, first_heading: str, last_heading: str) -> list:
    """The python blocks of a page between two of its headings, the second heading excluded."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines, 1) if line.startswith(first_heading))
    end = next(i for i, line in enumerate(lines, 1) if i > start and line.startswith(last_heading))
    return [b for b in extract_blocks(path) if start < b.line < end]


class PageRunner:
    """Executes the blocks of a page one by one in a shared namespace, as a reader running the page top to bottom.

    `edits` are exact (old, new) replacements that make the page runnable here: the placeholder API key becomes the
    real one, read from the environment so it never appears in the code, and the date ranges shrink to a few weeks.
    Every edit must match somewhere in the blocks, so a docs change that makes one stale fails the test before
    anything is downloaded.
    """

    def __init__(self, blocks: list, edits: list):
        self.blocks = blocks
        self.edits = edits
        self.namespace = {"os": os, "__name__": "docs_quick_start"}
        self.position = 0
        stale = [old for old, _ in edits if not any(old in b.code for b in blocks)]
        assert not stale, f"the docs no longer contain {stale}: update the edits that shorten them for this test"

    def step(self) -> Block:
        block = self.blocks[self.position]
        self.position += 1
        code = block.code
        for old, new in self.edits:
            code = code.replace(old, new)
        exec(compile(code, block.where, "exec"), self.namespace)
        return block

    @property
    def done(self) -> bool:
        return self.position == len(self.blocks)


README_EDITS = [
    ('api_key="YOUR_KEY"', 'api_key=os.environ["ENTSOE_API_KEY"]'),
    ('pd.Timestamp("2024-06-01", tz="UTC")', 'pd.Timestamp("2024-01-08", tz="UTC")'),
    ('pipeline.run("2024-01-01", "2024-06-08", cache=True)', f'pipeline.run("{DATA_START}", "{DATA_END}", cache=True)'),
    ('test_start="2024-04-01", test_end="2024-06-01"', f'test_start="{TEST_START}", test_end="{TEST_END}"'),
]
QUICKSTART_EDITS = [
    ('"your-api-key-here"', 'os.environ["ENTSOE_API_KEY"]'),
    ('start="2023-01-01", end="2024-04-01"', f'start="{DATA_START}", end="{DATA_END}"'),
    ('test_end="2024-03-01"', f'test_end="{TEST_END}"'),
]
README_COLUMNS = [
    "price",
    "load_actual",
    "load_actual_d-1",
    "is_monday_d+1",
    "is_monday_d+7",
    "is_holiday_d+1",
    "daylight_hours_d+1",
    "temperature_2m_d+1",
]


@pytest.mark.integration
@pytest.mark.examples
def test_readme_quick_start_runs(live_dir, information_set_checks):
    runner = PageRunner(blocks_between(ROOT / "README.md", "## Quick Start", "## Examples"), README_EDITS)
    namespace = runner.namespace
    first, last = dt.date.fromisoformat(DATA_START), dt.date.fromisoformat(DATA_END)
    planned = RUN_DAYS * 24 * 7  # the README leaves horizon at its default of 7

    runner.step()  # standalone source: UTC index, the half-open range
    df = namespace["df"]
    assert str(df.index.tz) == "UTC" and {"load_actual", "price"} <= set(df.columns)
    assert df.index[0] == pd.Timestamp("2024-01-01", tz="UTC") and df.index[-1] < pd.Timestamp("2024-01-08", tz="UTC")

    runner.step()  # standalone transformers: local time, one row per hour
    df = namespace["df"]
    assert str(df.index.tz) == "Europe/Warsaw"
    assert df.index.to_series().diff().dropna().eq(pd.Timedelta("1h")).all()

    runner.step()  # standalone validator
    assert list(namespace["result"].stats["column"]) == ["load_actual", "price"]  # one row per column

    runner.step()  # data pipeline
    assert_data_frame(namespace["df"], README_COLUMNS, "Europe/Warsaw", first, last)

    runner.step()  # standalone model: every forecast it plans is made
    ref = namespace["ref"]
    assert ref.count == planned and ref.produced == planned

    runner.step()  # standalone evaluator: the value in the README comment
    assert namespace["mae"] == pytest.approx(1.75)

    runner.step()  # standalone exporter
    assert_excel(live_dir / "results.xlsx", ["OLS"], planned)

    runner.step()  # model pipeline with an issue time
    report = namespace["report"]
    models = ["OLS", "Lasso"]
    assert_forecasts(report, models, planned)
    assert_metrics(report, models, planned, columns=("MAE",))
    assert_information_set_checked(information_set_checks, models)
    assert_excel(live_dir / "results.xlsx", models, planned)
    assert runner.done


@pytest.mark.integration
@pytest.mark.examples
def test_docs_quick_start_runs(live_dir, information_set_checks):
    page = DOCS / "getting-started" / "quickstart.mdx"
    runner = PageRunner(extract_blocks(page), QUICKSTART_EDITS)
    namespace = runner.namespace
    models = ["OLS", "LassoCV"]
    planned = RUN_DAYS * 24 * HORIZON

    runner.step()  # set up the environment
    assert os.environ["ENTSOE_API_KEY"]

    runner.step()  # download data
    columns = [c for c in PRICE_COLUMNS if c != "load_forecast_d+1"]
    assert_data_frame(
        namespace["df"], columns, "Europe/Warsaw", dt.date.fromisoformat(DATA_START), dt.date.fromisoformat(DATA_END)
    )

    runner.step()  # train the models
    report = namespace["report"]
    assert_forecasts(report, models, planned)
    assert_metrics(report, models, planned, columns=("MAE",))
    assert_information_set_checked(information_set_checks, models)
    assert_excel(live_dir / "results.xlsx", models, planned)
    assert sorted(p.name for p in (live_dir / "results").glob("*.jsonl")) == ["lassocv.jsonl", "ols.jsonl"]

    runner.step()  # analyse the results: the three calls the page shows
    assert runner.done
    assert len(report.by_horizon()) == len(models) * HORIZON
    assert len(report.by_hour()) == len(models) * 24
