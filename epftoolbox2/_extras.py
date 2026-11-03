"""Optional dependency groups ("extras"): which ones are installed, and the error raised when a missing one is needed.

The core never imports an extra at module level. Code that needs one imports it when first used and raises
`MissingExtraError` naming the extra to install. `has_extra` only looks for the modules (`importlib.util.find_spec`), so asking
doesn't import them.
"""

import importlib.util
from typing import Dict, Tuple

# Each extra in pyproject.toml's [project.optional-dependencies] that code asks for, with the modules it installs.
EXTRAS: Dict[str, Tuple[str, ...]] = {
    "fast": ("numba",),
}


class MissingExtraError(ImportError):
    """`what` needs an optional dependency group that is not installed; the message says how to install it."""

    def __init__(self, what: str, extra: str):
        self.what = what
        self.extra = extra
        super().__init__(f"{what} needs: pip install 'epftoolbox2[{extra}]'")

    def __reduce__(self):
        return type(self), (self.what, self.extra)


def has_extra(name: str) -> bool:
    """True when every module of the extra `name` can be found. An unknown name raises ValueError."""
    if name not in EXTRAS:
        raise ValueError(f"Unknown extra {name!r}. Known extras: {sorted(EXTRAS)}")
    return all(importlib.util.find_spec(module) is not None for module in EXTRAS[name])
