from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple, Union

import pandas as pd

_INFO_ITEMS = 5  # list/dict items shown per info entry in str()
_INFO_WIDTH = 200  # characters per info entry in str()


@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    validator: str = ""  # class name of the validator that produced the result

    def __bool__(self) -> bool:
        return self.is_valid

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        parts = [f"{self.validator or 'ValidationResult'}: {status}"]
        if self.errors:
            parts.append(f"  Errors: {len(self.errors)}")
            for e in self.errors:
                parts.append(f"    - {e}")
        if self.warnings:
            parts.append(f"  Warnings: {len(self.warnings)}")
            for w in self.warnings:
                parts.append(f"    - {w}")
        if self.info:
            parts.append("  Info:")
            for key, value in self.info.items():
                parts.append(f"    {key}: {_format_value(value)}")
        if not self.stats.empty:
            parts.append(f"  Stats: {len(self.stats)} rows x {self.stats.shape[1]} columns (see .stats)")
        return "\n".join(parts)

    def to_frame(self) -> pd.DataFrame:
        """One row per error and warning; see `results_to_frame`."""
        return results_to_frame([self])


def results_to_frame(
    results: Union[Mapping, Iterable[Union[ValidationResult, Tuple[str, ValidationResult]]]],
) -> pd.DataFrame:
    """Table of validation messages with columns validator, valid, level, message.

    `results` is a list of results, a list of (name, result) pairs, or a dict {name: result}. Each error and warning
    becomes a row (errors first); a result without messages becomes one row with level "ok".

    Example:
        >>> checks = [CoverageValidator(), DSTStructureValidator()]
        >>> print(results_to_frame([v.validate(df) for v in checks]).to_string())
    """
    items = results.items() if isinstance(results, Mapping) else results
    rows = []
    for item in items:
        name, result = item if isinstance(item, tuple) else (item.validator, item)
        name = name or result.validator
        messages = [("error", m) for m in result.errors] + [("warning", m) for m in result.warnings]
        for level, message in messages or [("ok", "")]:
            rows.append({"validator": name, "valid": result.is_valid, "level": level, "message": message})
    return pd.DataFrame(rows, columns=["validator", "valid", "level", "message"])


def _format_value(value: Any) -> str:
    if isinstance(value, (pd.DataFrame, pd.Series)):
        text = f"<{type(value).__name__} with {len(value)} rows>"
    elif isinstance(value, Mapping):
        items = [f"{k!r}: {_format_scalar(v)}" for k, v in list(value.items())[:_INFO_ITEMS]]
        more = f", ... ({len(value) - _INFO_ITEMS} more)" if len(value) > _INFO_ITEMS else ""
        text = "{" + ", ".join(items) + more + "}"
    elif isinstance(value, (list, tuple)):
        items = [_format_scalar(v) for v in list(value)[:_INFO_ITEMS]]
        more = f", ... ({len(value) - _INFO_ITEMS} more)" if len(value) > _INFO_ITEMS else ""
        text = "[" + ", ".join(items) + more + "]"
    else:
        text = _format_scalar(value)
    return text if len(text) <= _INFO_WIDTH else text[: _INFO_WIDTH - 3] + "..."


def _format_scalar(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{k!r}: {_format_scalar(v)}" for k, v in value.items()) + "}"
    if isinstance(value, float):
        return f"{value:g}"
    return repr(value)
