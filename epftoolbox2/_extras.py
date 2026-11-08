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
}


class MissingExtraError(ImportError):
    """`what` needs an optional dependency group that is not installed; the message says how to install it.

    `extra` may name several, comma-separated (`"t0,tabpfn"`, which pip reads as both), and `message` replaces the one-line
    message: the pre-flight of a run (`models._cards.preflight`) names every problem of every model in one error."""

    def __init__(self, what: str, extra: str, message: Optional[str] = None):
        self.what = what
        self.extra = extra
        self.message = message
        super().__init__(message if message is not None else f"{what} needs: pip install 'epftoolbox2[{extra}]'")

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
