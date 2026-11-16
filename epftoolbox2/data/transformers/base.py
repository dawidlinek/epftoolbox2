from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class Transformer(ABC):
    """Abstract base class for all data transformers.

    Transformers take a DataFrame and return a transformed DataFrame.
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def lookback_days(self) -> Optional[int]:
        """How many local days before a row its output reads, at most (0: the row's own day only), or None when that can't be
        told. `ModelPipeline.forecast()` asks a data pipeline for that much more history before the oldest row a fit reads, so a
        lag or a rolling window is as the backtest's frame has it; a transformer answering None makes it warn that the range
        may be short (pass `span=`). The built-in transformers answer; a transformer of your own should override this."""
        return None
