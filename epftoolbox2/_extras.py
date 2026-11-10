"""Optional dependency groups ("extras"): which ones are installed, and the error raised when a missing one is needed.

The core never imports an extra at module level. Code that needs one imports it when first used and raises
`MissingExtraError` naming the extra to install. `has_extra` only looks for the modules (`importlib.util.find_spec`), so asking
doesn't import them.
"""

import importlib.util
from typing import Dict, List, Optional, Tuple

# Each extra in pyproject.toml's [project.optional-dependencies] that code asks for, with the modules it installs.
EXTRAS: Dict[str, Tuple[str, ...]] = {
    "fast": ("numba",),
    "torch": ("torch",),
    "lightgbm": ("lightgbm",),
    "xgboost": ("xgboost",),
    "catboost": ("catboost",),
    "tune": ("optuna",),
    "t0": ("t0", "torch", "safetensors"),
    "chronos": ("chronos", "torch", "transformers"),
    # 6j's foundation families. `timestral` isn't on PyPI: its extra installs the package's PyPI dependencies, and the package
    # comes from its pinned commit (`TIMESTRAL_COMMIT`, `install_line`).
    "patchtst-fm": ("tsfm_public", "torch", "transformers"),
    "timesfm": ("timesfm3", "torch", "safetensors"),
    "timestral": ("timestral", "torch", "safetensors"),
    # The in-context families (step 6e), each with torch, which its package imports.
    "tabpfn": ("tabpfn", "torch"),
    "tabicl": ("tabicl", "torch"),
}

# The commit of github.com/mistralai/timestral that TimestralModel runs on (the research's, 2026-10-15, its first): its card
# pins it, and the pre-flight refuses another installed commit unless the model says otherwise (`runtime_commit=`, `weights=`).
TIMESTRAL_COMMIT = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
TIMESTRAL_SOURCE = f"timestral @ git+https://github.com/mistralai/timestral@{TIMESTRAL_COMMIT}"
# Packages an extra needs that pip can't get from the index: `install_line` adds them after `epftoolbox2[extra]`.
EXTRA_SOURCES: Dict[str, Tuple[str, ...]] = {
    "timestral": (TIMESTRAL_SOURCE,),
}


def install_line(extra: str) -> str:
    """The command that installs `extra` (one name, or several comma-separated): `pip install 'epftoolbox2[extra]'`, followed
    by the packages `EXTRA_SOURCES` lists for it (the timestral package at its commit)."""
    names = [name.strip() for name in extra.split(",") if name.strip()]
    sources = [source for name in names for source in EXTRA_SOURCES.get(name, ())]
    return f"pip install 'epftoolbox2[{','.join(names)}]'" + "".join(f" '{source}'" for source in sources)


class MissingExtraError(ImportError):
    """`what` needs an optional dependency group that is not installed; the message says how to install it.

    `extra` may name several, comma-separated (`"t0,tabpfn"`, which pip reads as both), and `message` replaces the one-line
    message: the pre-flight of a run (`models._cards.preflight`) names every problem of every model in one error."""

    def __init__(self, what: str, extra: str, message: Optional[str] = None):
        self.what = what
        self.extra = extra
        self.message = message
        super().__init__(message if message is not None else f"{what} needs: {install_line(extra)}")

    def __reduce__(self):
        return type(self), (self.what, self.extra, self.message)


def has_extra(name: str) -> bool:
    """True when every module of the extra `name` can be found. An unknown name raises ValueError."""
    if name not in EXTRAS:
        raise ValueError(f"Unknown extra {name!r}. Known extras: {sorted(EXTRAS)}")
    return all(importlib.util.find_spec(module) is not None for module in EXTRAS[name])


def extras_rows() -> List[Tuple[str, bool, str]]:
    """(extra, installed, versions) for every extra of `EXTRAS`, for `epftoolbox2.verify()`: the version of the distribution
    that installs each of its modules (`numba 0.65.0`), "-" for a module not found. Nothing is imported."""
    import importlib.metadata

    owners = importlib.metadata.packages_distributions()
    rows = []
    for name, modules in EXTRAS.items():
        shown = []
        for module in modules:
            top = module.split(".")[0]
            version = None
            if importlib.util.find_spec(top) is not None:
                for distribution in owners.get(top, [top]):
                    try:
                        version = importlib.metadata.version(distribution)
                        break
                    except importlib.metadata.PackageNotFoundError:
                        continue
                version = version or "found"
            shown.append(f"{module} {version or '-'}")
        rows.append((name, has_extra(name), ", ".join(shown)))
    return rows
