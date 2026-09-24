"""Resolves a component's market time zone against `DataPipeline.timezone`.

`DataPipeline.timezone` is meant to be the single place the market zone is written. Components whose own zone
decides a local calendar day or an issue time (`CalendarSource.timezone`, `OpenMeteoSource.timezone`,
`OpenMeteoSingleRunsSource.timezone`, `OpenMeteoWeightedSource.source_params["timezone"]`,
`TimezoneTransformer.target_tz`) may leave it `None` and inherit `DataPipeline.timezone` instead.

`DataPipeline` resolves this once, before running (`DataPipeline._bind_component_timezones`), and stores the
result on each component's private `_pipeline_timezone` attribute — never the public one, so
`DataPipeline.save()` keeps writing the value the user actually gave (often `None`) and a YAML round trip
re-resolves the same way as long as it is reloaded with the same `DataPipeline.timezone`. A component used
without a `DataPipeline` never has `_pipeline_timezone` bound, so it falls back to its own default (if any) or
raises through `require`.
"""

from typing import Optional

PIPELINE_TIMEZONE_ATTR = "_pipeline_timezone"
PIPELINE_OWNER_ATTR = "_pipeline_timezone_owner"


def resolve(own: Optional[str], component: object) -> Optional[str]:
    """`own` if set, else the zone a `DataPipeline` bound onto `component` (may still be `None`)."""
    return own if own is not None else getattr(component, PIPELINE_TIMEZONE_ATTR, None)


def require(zone: Optional[str], where: str) -> str:
    """`zone`, or a clear error naming `where` a zone can be set."""
    if zone is None:
        raise ValueError(
            f"{where} has no time zone: pass it explicitly here, or set it once with DataPipeline(timezone=...) "
            "and leave this component's own zone unset so it inherits it."
        )
    return zone


def check_match(own: Optional[str], pipeline_timezone: Optional[str], where: str) -> None:
    """Raise if both `own` and `pipeline_timezone` are set and disagree."""
    if own is not None and pipeline_timezone is not None and own != pipeline_timezone:
        raise ValueError(
            f"DataPipeline.timezone={pipeline_timezone!r} does not match {where}={own!r}. DataPipeline.timezone "
            "is meant to be the single place the market zone is set; use the same zone in both, or leave the "
            "component's own zone unset so it inherits DataPipeline.timezone."
        )


def bind(
    component: object, own: Optional[str], pipeline_timezone: Optional[str], where: str, owner: object = None
) -> None:
    """Check `own` against `pipeline_timezone` and store the inherited zone on `component` (see module docstring).

    A component instance belongs to one pipeline's zone: binding it from a different `owner` to a different zone
    raises instead of silently changing the zone the first pipeline (or a later standalone use) relies on.
    """
    check_match(own, pipeline_timezone, where)
    bound = getattr(component, PIPELINE_TIMEZONE_ATTR, None)
    bound_owner = getattr(component, PIPELINE_OWNER_ATTR, None)
    if bound_owner is not None and owner is not None and bound_owner != id(owner) and bound != pipeline_timezone:
        raise ValueError(
            f"{where} is already used by another DataPipeline with timezone={bound!r}, so it can't also inherit "
            f"timezone={pipeline_timezone!r}. Give each pipeline its own component instance."
        )
    setattr(component, PIPELINE_TIMEZONE_ATTR, pipeline_timezone)
    if owner is not None:
        setattr(component, PIPELINE_OWNER_ATTR, id(owner))
