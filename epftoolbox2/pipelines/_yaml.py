"""YAML conversion and writing shared by the pipelines' ``save()`` methods."""

import importlib
import inspect
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


def class_path(cls: type, where: str) -> str:
    """``package.module:Class``, the import path a class is saved by (``TorchModel(module=...)``), if importing it gives the
    class back; a ``TypeError`` naming ``where`` otherwise: a class defined in the script being run (``__main__``) or inside a
    function, which a later load can't import, or one whose import path gives another object."""
    module, qualname = getattr(cls, "__module__", None), getattr(cls, "__qualname__", None)
    if not module or not qualname:
        reason = "it has no import path"
    elif "<locals>" in qualname:
        reason = "it is defined inside a function"
    elif module == "__main__":
        reason = "it is defined in the script being run (__main__), which a later load can't import"
    else:
        try:
            found = _import(module, qualname)
            reason = None if found is cls else f"importing {module}:{qualname} gives another object"
        except (ImportError, AttributeError) as exc:
            reason = f"{module}:{qualname} can't be imported ({exc})"
    if reason:
        raise TypeError(
            f"Cannot save {where}: a class is saved by its import path, and {reason}. Define it at the top level of a module "
            "that can be imported wherever the file is loaded."
        )
    return f"{module}:{qualname}"


def load_class(path: str, where: str) -> type:
    """The class a saved file names as ``package.module:Class`` (``class_path``). A module that can't be imported raises its
    ``ImportError`` (``MissingExtraError`` is one), a missing name or a value that isn't a class ``ValueError``/``TypeError``
    naming ``where``."""
    module, _, qualname = str(path).partition(":")
    if not module or not qualname:
        raise ValueError(f"{where}: a class is given as 'package.module:Class', got {path!r}")
    try:
        found = _import(module, qualname)
    except AttributeError as exc:
        raise ValueError(f"{where}: {path!r} names nothing that {module} defines ({exc})") from None
    if not inspect.isclass(found):
        raise TypeError(f"{where}: {path!r} is a {type(found).__name__}, not a class")
    return found


def _import(module: str, qualname: str) -> Any:
    found: Any = importlib.import_module(module)
    for part in qualname.split("."):
        found = getattr(found, part)
    return found


def to_yaml_value(value: Any, where: str) -> Any:
    """Plain YAML-safe value (numpy scalars and arrays, tuples, ranges and paths converted, a class written by its import path,
    ``class_path``); raises otherwise.

    ``where`` names the parameter in the ``TypeError`` raised for a value YAML cannot represent, so a parameter is
    never silently dropped or rewritten as a Python object that ``yaml.safe_load`` cannot read back.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if inspect.isclass(value):
        return class_path(value, where)
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
