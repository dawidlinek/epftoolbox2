import copy
from typing import Any, Dict, Tuple, Type, Union

from .base import BaseScaler
from .standard import StandardScaler
from .vst import AsinhScaler, NPITScaler

SCALERS: Dict[str, Type[BaseScaler]] = {
    "standard": StandardScaler,
    "asinh": AsinhScaler,
    "npit": NPITScaler,
}

ScalerSpec = Union[str, Dict[str, Any], BaseScaler]


def _split_spec(spec: ScalerSpec) -> Tuple[str, Dict[str, Any]]:
    if isinstance(spec, BaseScaler):
        name = type(spec).name
        if SCALERS.get(name) is not type(spec):
            raise ValueError(f"{type(spec).__name__} is not a registered scaler. Available: {sorted(SCALERS)}")
        return name, copy.deepcopy(spec.get_params())
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict):
        params = spec.get("params") or {}
        if "name" not in spec or set(spec) - {"name", "params"} or not isinstance(params, dict):
            raise ValueError(f"A scaler dict must look like {{'name': 'asinh', 'params': {{...}}}}, got {spec!r}")
        return spec["name"], copy.deepcopy(params)
    raise TypeError(
        f"scaler must be a name {sorted(SCALERS)}, a {{'name': ..., 'params': ...}} dict or a scaler instance, "
        f"got {spec!r}"
    )


def make_scaler(spec: ScalerSpec = "standard") -> BaseScaler:
    """Build a fresh scaler from a name, a ``{"name": ..., "params": {...}}`` dict or a scaler instance."""
    name, params = _split_spec(spec)
    if name not in SCALERS:
        raise ValueError(f"Unknown scaler {name!r}. Available: {sorted(SCALERS)}")
    try:
        return SCALERS[name](**params)
    except TypeError as e:
        raise ValueError(f"Invalid parameters for scaler {name!r}: {e}") from e


def scaler_spec(spec: ScalerSpec = "standard") -> Union[str, Dict[str, Any]]:
    """Validate ``spec`` and return its YAML-safe form: the name, or ``{"name": ..., "params": {...}}``."""
    name, params = _split_spec(spec)
    scaler = make_scaler({"name": name, "params": params})
    if not params:
        return name
    return {"name": name, "params": scaler.get_params()}
