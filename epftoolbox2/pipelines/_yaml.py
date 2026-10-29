"""YAML conversion and writing shared by the pipelines' ``save()`` methods."""

from pathlib import Path
from typing import Any

import numpy as np
import yaml


class NoAliasSafeDumper(yaml.SafeDumper):
    """A ``SafeDumper`` that writes an object used twice (a predictor list shared by two models) in full each time,
    instead of as an anchor and alias (``&id001`` / ``*id001``). The aliases are valid YAML, but they make a saved
    configuration hard to read and to edit, since changing one model's list would silently change the other's."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def dump_yaml(data: Any) -> str:
    """``data`` as block-style YAML text with the keys in their given order and no anchors or aliases.

    Returns the text instead of writing to a stream, so a value YAML cannot represent raises before the caller
    opens (and truncates) the file it was going to replace.
    """
    return yaml.dump(data, Dumper=NoAliasSafeDumper, default_flow_style=False, sort_keys=False)


def to_yaml_value(value: Any, where: str) -> Any:
    """Plain YAML-safe value (numpy scalars and arrays, tuples, ranges and paths converted); raises otherwise.

    ``where`` names the parameter in the ``TypeError`` raised for a value YAML cannot represent, so a parameter is
    never silently dropped or rewritten as a Python object that ``yaml.safe_load`` cannot read back.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, np.generic):
        return to_yaml_value(value.item(), where)
    if isinstance(value, np.ndarray):
        return to_yaml_value(value.tolist(), where)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, range)):
        return [to_yaml_value(v, where) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, (str, int)) for k in value):
            raise TypeError(f"Cannot save {where}: dictionary keys must be strings or integers")
        return {k: to_yaml_value(v, f"{where}.{k}") for k, v in value.items()}
    kind = "a callable" if callable(value) else f"a {type(value).__name__}"
    raise TypeError(
        f"Cannot save {where}: it is {kind}, which YAML cannot represent "
        "(use str, int, float, bool, None, list or dict)"
    )
