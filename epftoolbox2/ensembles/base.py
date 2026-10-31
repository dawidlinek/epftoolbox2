"""What an ensemble is: a rule over the stored forecasts of named members.

An ensemble fits nothing and reads no data. `ModelPipeline` computes it from its members' records (`ensembles._compute`) and
hands it plain arrays on identical forecast keys, so a class here only says what to choose and how to combine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Choice:
    """What an ensemble chose on one validation period.

    Attributes:
        chosen: The members kept, in member order.
        mae: MAE of the combination of `chosen` on the keys it was scored on.
        member_mae: Each member's own MAE on those keys.
        k_mae: `top="validation"` alone: the MAE of the combination of the best k members, for each k.
    """

    chosen: List[str]
    mae: float
    member_mae: Dict[str, float]
    k_mae: Optional[Dict[int, float]] = None


class Ensemble(ABC):
    """A rule over the forecasts of named models or other ensembles, given to `ModelPipeline.add_ensemble`.

    Every constructor parameter is a public attribute of the same name, which is what a YAML file holds. An ensemble keeps
    nothing of what it chose: that goes to the results and to `report.choices()`, so one object can run twice.

    Args:
        members: Names of models or ensembles of the pipeline, resolved when it runs.
        name: Names the ensemble in every table and its results file; a model can't have the same name.
    """

    def __init__(self, members: Sequence[str], name: str):
        self.members = members if isinstance(members, str) else list(members)
        self.name = name

    def _check(self) -> None:
        """`ValueError` for parameters that can't work, whichever way they got here."""
        if isinstance(self.members, str) or not all(isinstance(member, str) for member in self.members):
            raise ValueError(f"members must be a list of model names, got {self.members!r}")
        if not self.members:
            raise ValueError("An ensemble needs at least one member")
        if len(set(self.members)) != len(self.members):
            raise ValueError(f"A member is listed twice, which would weigh it double: {self.members}")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"An ensemble needs a name, got {self.name!r}")

    @property
    def needs_validation(self) -> bool:
        """Whether it chooses on a validation period. If so, its forecasts exist in the test window alone."""
        return False

    @abstractmethod
    def choose(self, predictions: np.ndarray, actual: np.ndarray) -> Choice:
        """Choose from `predictions`, one row per common validation key and one column per member, all finite."""

    @abstractmethod
    def combine(self, predictions: np.ndarray) -> np.ndarray:
        """One forecast per row of `predictions`, whose columns are the chosen members in member order, all finite."""
