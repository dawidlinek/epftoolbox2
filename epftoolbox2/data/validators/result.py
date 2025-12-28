from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd


@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    stats: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __bool__(self) -> bool:
        return self.is_valid

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        parts = [f"ValidationResult: {status}"]
        if self.errors:
            parts.append(f"  Errors: {len(self.errors)}")
            for e in self.errors:
                parts.append(f"    - {e}")
        if self.warnings:
            parts.append(f"  Warnings: {len(self.warnings)}")
            for w in self.warnings:
                parts.append(f"    - {w}")
        return "\n".join(parts)
