#!/usr/bin/env python3
"""Report drift between epftoolbox2's code and its documentation surfaces.

Checks
  registry  every public component is registered for YAML loading and offered in the TUI,
            and TUI parameter schemas match the constructors
  coverage  every component has a docs page and a README mention
  params    every constructor parameter appears in the component's docs page
  index     llms.txt (the entry point for agents) has an H1 and a summary, links every docs page under
            docs/src/content/docs, and every repo-relative link in it points to an existing file
  imports   `from epftoolbox2... import X` statements in code blocks and examples resolve
  calls     keyword arguments in documented `Component(...)` calls are accepted by the constructor
  report    `report.<method>(` calls in code blocks exist on EvaluationReport

Usage
  uv run python .claude/skills/docs-sync/scripts/check_docs.py              # human-readable report
  uv run python .claude/skills/docs-sync/scripts/check_docs.py --inventory  # JSON: params, defaults, docs pages
  uv run python .claude/skills/docs-sync/scripts/check_docs.py --json       # findings as JSON

Exit code 1 when any error is found; warnings don't fail.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

DOCS_DIR = ROOT / "docs" / "src" / "content" / "docs"
LLMS = ROOT / "llms.txt"
README = ROOT / "README.md"
EXAMPLES = ROOT / "examples"

# (category, package, base class name, docs sub-directory, registry module, TUI list name)
CATEGORIES = [
    ("sources", "epftoolbox2.data.sources", "DataSource", "data-sources", "data_pipeline", "_SOURCES"),
    ("transformers", "epftoolbox2.data.transformers", "Transformer", "transformers", "data_pipeline", "_TRANSFORMERS"),
    ("validators", "epftoolbox2.data.validators", "Validator", "validators", "data_pipeline", "_VALIDATORS"),
    ("models", "epftoolbox2.models", "BaseModel", "models", "model_pipeline", "_MODELS"),
    ("evaluators", "epftoolbox2.evaluators", "Evaluator", "evaluators", "model_pipeline", "_EVALUATORS"),
    ("exporters", "epftoolbox2.exporters", "Exporter", "exporters", "model_pipeline", "_EXPORTERS"),
]
SHORT_NAME_SUFFIXES = ("Source", "Transformer", "Validator", "Model", "Evaluator", "Exporter")


@dataclass
class Finding:
    check: str
    severity: str  # "error" | "warning"
    subject: str
    message: str
    location: str = ""


@dataclass
class Component:
    category: str
    name: str
    module: str
    params: dict  # name -> repr(default) or None when required
    docs_page: str | None = None


# --------------------------------------------------------------------------- helpers


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def constructor_params(cls: type) -> dict:
    """Constructor parameters, following **kwargs up the MRO (e.g. LassoCVModel → BaseModel)."""
    params: dict = {}
    for klass in cls.__mro__:
        if klass is object:
            break
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        passes_kwargs = False
        for p in list(inspect.signature(init).parameters.values())[1:]:
            if p.kind is p.VAR_KEYWORD:
                passes_kwargs = True
            elif p.kind is not p.VAR_POSITIONAL and p.name not in params:
                params[p.name] = None if p.default is p.empty else repr(p.default)
        if not passes_kwargs:
            break
    return params


def normalize(value: str) -> str:
    """Comparable form of a default: first token, unquoted, lower-case; '' and None are equivalent."""
    value = value.strip().strip("`").strip()
    value = re.split(r"\s|=", value, maxsplit=1)[0] if value else value
    value = value.strip("'\"").lower().rstrip(".")
    return "none" if value == "" else value


def code_blocks(path: Path) -> list[tuple[int, str]]:
    """(first_line_number, code) for python code blocks; a whole .py file is one block."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return [(1, text)]
    blocks, current, start, lang = [], None, 0, ""
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            if current is None:
                lang, current, start = stripped[3:].strip().lower(), [], i + 1
            else:
                # A fence meta string `nocheck` (```python nocheck) marks a block that is deliberately not runnable
                # Python. tests/test_docs_code.py skips it too.
                if lang.startswith(("python", "py")) and "nocheck" not in lang.split()[1:]:
                    blocks.append((start, "\n".join(current)))
                current = None
            continue
        if current is not None:
            current.append(line)
    return blocks


IMPORT_LINE = re.compile(r"^\s*from\s+(epftoolbox2[\w.]*)\s+import\s+(.+)$")


def epf_imports(code: str) -> list[tuple[int, str, str]]:
    """(line_offset, module, name) for every `from epftoolbox2... import name` in a block."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    found = []
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("epftoolbox2"):
                found += [(node.lineno - 1, node.module, a.name) for a in node.names]
        return found
    # Snippet isn't valid Python on its own: fall back to line matching (handles `import (a, b)` too).
    lines = code.splitlines()
    for i, line in enumerate(lines):
        m = IMPORT_LINE.match(line)
        if not m:
            continue
        names = m.group(2)
        j = i
        while "(" in names and ")" not in names and j + 1 < len(lines):
            j += 1
            names += " " + lines[j]
        names = re.sub(r"\bas\s+\w+", "", names.split("#")[0])  # drop `as alias`
        for name in re.split(r"[,\s()]+", names):
            if name and name.isidentifier():
                found.append((i, m.group(1), name))
    return found


def doc_files() -> list[Path]:
    files = [README] + sorted(DOCS_DIR.rglob("*.md")) + sorted(DOCS_DIR.rglob("*.mdx"))
    return files + sorted(EXAMPLES.glob("*.py"))


# --------------------------------------------------------------------------- discovery


def discover() -> list[Component]:
    components = []
    for category, package, base_name, docs_dir, _, _ in CATEGORIES:
        module = importlib.import_module(package)
        base = getattr(module, base_name)
        for name in getattr(module, "__all__", []):
            obj = getattr(module, name)
            if not inspect.isclass(obj) or obj is base or not issubclass(obj, base) or inspect.isabstract(obj):
                continue
            page = None
            best = 0
            for path in sorted((DOCS_DIR / docs_dir).glob("*.md*")):
                hits = len(re.findall(rf"\b{re.escape(name)}\b", path.read_text(encoding="utf-8")))
                if hits > best:
                    page, best = rel(path), hits
            components.append(
                Component(
                    category=category,
                    name=name,
                    module=obj.__module__,
                    params=constructor_params(obj),
                    docs_page=page,
                )
            )
    return components


# --------------------------------------------------------------------------- checks


def check_registry(components: list[Component]) -> list[Finding]:
    findings = []
    registries = {
        "data_pipeline": importlib.import_module("epftoolbox2.pipelines.data_pipeline").COMPONENT_REGISTRY,
        "model_pipeline": importlib.import_module("epftoolbox2.pipelines.model_pipeline").COMPONENT_REGISTRY,
    }
    tui = importlib.import_module("epftoolbox2.tui")
    schemas = getattr(tui, "PARAM_SCHEMAS", {})
    by_category = {c[0]: c for c in CATEGORIES}
    for comp in components:
        _, _, _, _, registry_module, tui_list = by_category[comp.category]
        registry = registries[registry_module].get(comp.category, {})
        where = f"epftoolbox2/pipelines/{registry_module}.py"
        if comp.name not in registry:
            findings.append(
                Finding("registry", "error", comp.name, "not in COMPONENT_REGISTRY — can't be loaded from YAML", where)
            )
        elif registry[comp.name] != comp.module:
            findings.append(
                Finding(
                    "registry",
                    "error",
                    comp.name,
                    f"registry points to {registry[comp.name]}, class lives in {comp.module}",
                    where,
                )
            )
        if comp.name not in getattr(tui, tui_list, []):
            findings.append(
                Finding("registry", "warning", comp.name, f"not offered in the TUI ({tui_list})", "epftoolbox2/tui.py")
            )
        schema = schemas.get(comp.name)
        if schema is None:
            if comp.name in getattr(tui, tui_list, []) and comp.params:
                findings.append(
                    Finding("registry", "warning", comp.name, "no PARAM_SCHEMAS entry in the TUI", "epftoolbox2/tui.py")
                )
            continue
        schema_names = {f["name"] for f in schema}
        for extra in sorted(schema_names - comp.params.keys()):
            findings.append(
                Finding(
                    "registry",
                    "error",
                    comp.name,
                    f"TUI asks for '{extra}', which the constructor doesn't accept",
                    "epftoolbox2/tui.py",
                )
            )
        for missing in sorted(comp.params.keys() - schema_names):
            findings.append(
                Finding(
                    "registry", "warning", comp.name, f"TUI doesn't ask for parameter '{missing}'", "epftoolbox2/tui.py"
                )
            )
        for field in schema:
            code_default = comp.params.get(field["name"])
            if field["name"] == "name" or code_default is None or field.get("default") is None:
                continue  # the TUI proposes display names ("OLS") on purpose
            if normalize(str(field["default"])) != normalize(code_default):
                findings.append(
                    Finding(
                        "registry",
                        "warning",
                        comp.name,
                        f"TUI default for '{field['name']}' is {field['default']!r}, constructor default is {code_default}",
                        "epftoolbox2/tui.py",
                    )
                )
    return findings


def check_coverage_and_params(components: list[Component]) -> list[Finding]:
    """Every component has a README mention and a docs page that names all of its constructor parameters."""
    findings = []
    readme = README.read_text(encoding="utf-8").lower()
    for comp in components:
        short = comp.name
        for suffix in SHORT_NAME_SUFFIXES:
            if short.endswith(suffix) and short != suffix:
                short = short[: -len(suffix)]
                break
        if comp.name.lower() not in readme and short.lower() not in readme:
            findings.append(
                Finding(
                    "coverage", "warning", comp.name, f"not mentioned in README (looked for '{short}')", "README.md"
                )
            )

        if comp.docs_page is None:
            findings.append(
                Finding(
                    "coverage",
                    "error",
                    comp.name,
                    "no docs page mentions this class",
                    f"docs/src/content/docs/{dict((c[0], c[3]) for c in CATEGORIES)[comp.category]}/",
                )
            )
        else:
            page_text = (ROOT / comp.docs_page).read_text(encoding="utf-8")
            for param in comp.params:
                if not re.search(rf"\b{re.escape(param)}\b", page_text):
                    findings.append(
                        Finding("params", "error", comp.name, f"parameter '{param}' not documented", comp.docs_page)
                    )
    return findings


MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
INLINE_CODE = re.compile(r"`[^`]*`")


def markdown_links(text: str) -> list[tuple[int, str]]:
    """(line_number, target) for every `[text](target)` link outside code fences and inline code."""
    links = []
    in_code = False
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        links += [(i, m.group(2)) for m in MARKDOWN_LINK.finditer(INLINE_CODE.sub("", line))]
    return links


def check_llms_index() -> list[Finding]:
    """llms.txt is an index of the docs: an H1 and a summary, a link to every docs page, no dead links."""
    findings = []
    text = LLMS.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or not re.match(r"^# \S", lines[0]):
        findings.append(
            Finding("index", "error", "llms.txt", "the first line must be an H1 with the project name", "llms.txt:1")
        )
    if not any(line.startswith("> ") for line in lines):
        findings.append(Finding("index", "error", "llms.txt", "no `>` summary after the H1", "llms.txt"))

    linked: set[Path] = set()
    for line_no, target in markdown_links(text):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
            continue  # external URL: not checkable offline
        path = (ROOT / target.split("#", 1)[0]).resolve()
        if not path.exists():
            findings.append(
                Finding("index", "error", target, "link points to a file that does not exist", f"llms.txt:{line_no}")
            )
        linked.add(path)

    pages = sorted(DOCS_DIR.rglob("*.md")) + sorted(DOCS_DIR.rglob("*.mdx"))
    for page in pages:
        if page.resolve() not in linked:
            findings.append(Finding("index", "error", rel(page), "docs page is not linked from llms.txt", "llms.txt"))
    return findings


def check_imports() -> list[Finding]:
    findings = []
    for path in doc_files():
        for start, code in code_blocks(path):
            for offset, module_name, name in epf_imports(code):
                location = f"{rel(path)}:{start + offset}"
                try:
                    module = importlib.import_module(module_name)
                except ImportError as exc:
                    findings.append(
                        Finding("imports", "error", module_name, f"module doesn't import ({exc})", location)
                    )
                    continue
                if name == "*" or hasattr(module, name):
                    continue
                try:
                    importlib.import_module(f"{module_name}.{name}")
                except ImportError:
                    findings.append(
                        Finding("imports", "error", f"{module_name}.{name}", "name doesn't exist", location)
                    )
    return findings


def check_calls(components: list[Component]) -> list[Finding]:
    """Documented `Component(kw=...)` calls must only use keywords the constructor accepts."""
    params = {c.name: c.params for c in components}
    findings = []
    for path in doc_files():
        for start, code in code_blocks(path):
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue  # fragments like `...(` can't be checked reliably
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
                if name not in params:
                    continue
                for kw in node.keywords:
                    if kw.arg is not None and kw.arg not in params[name]:
                        findings.append(
                            Finding(
                                "calls",
                                "error",
                                name,
                                f"called with '{kw.arg}=', constructor accepts {sorted(params[name])}",
                                f"{rel(path)}:{start + node.lineno - 1}",
                            )
                        )
    return findings


def check_report_methods() -> list[Finding]:
    from epftoolbox2.results.report import EvaluationReport

    findings = []
    for path in doc_files():
        for start, code in code_blocks(path):
            for offset, line in enumerate(code.splitlines()):
                for method in re.findall(r"\breport\.(\w+)\(", line):
                    if not hasattr(EvaluationReport, method):
                        findings.append(
                            Finding(
                                "report",
                                "error",
                                f"report.{method}",
                                "EvaluationReport has no such method",
                                f"{rel(path)}:{start + offset}",
                            )
                        )
    return findings


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="print findings as JSON")
    parser.add_argument(
        "--inventory", action="store_true", help="print components with params, defaults and docs pages as JSON"
    )
    args = parser.parse_args()

    components = discover()
    if args.inventory:
        print(json.dumps([asdict(c) for c in components], indent=2))
        return 0

    findings = (
        check_registry(components)
        + check_coverage_and_params(components)
        + check_llms_index()
        + check_imports()
        + check_calls(components)
        + check_report_methods()
    )
    errors = [f for f in findings if f.severity == "error"]

    if args.json:
        print(json.dumps([asdict(f) for f in findings], indent=2))
        return 1 if errors else 0

    print(f"Checked {len(components)} components against README.md, llms.txt, docs/ and examples/.")
    if not findings:
        print("No drift found.")
        return 0
    for check in ("registry", "coverage", "params", "index", "imports", "calls", "report"):
        group = [f for f in findings if f.check == check]
        if not group:
            continue
        print(f"\n## {check}")
        for f in sorted(group, key=lambda f: (f.severity != "error", f.subject)):
            print(f"- [{f.severity}] {f.subject}: {f.message}" + (f"  ({f.location})" if f.location else ""))
    print(f"\n{len(errors)} error(s), {len(findings) - len(errors)} warning(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
