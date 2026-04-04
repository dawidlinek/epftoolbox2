import re
from datetime import date, timedelta
from typing import Any

_DATE_PATTERN = re.compile(r"^(today|now)(?:_d([+-]\d+))?$", re.IGNORECASE)


def resolve_date(s: Any) -> Any:
    """Resolve relative date keywords to ISO date strings.

    Supported patterns:
        'today', 'now'          → today's date
        'today_d+7', 'now_d+7'  → today + 7 days
        'today_d-365'           → today - 365 days
        '2024-01-01'            → passed through unchanged

    The offset N in _d±N accepts any integer, so training windows of any
    length can be expressed (e.g. 'now_d-800' for an 800-day history).
    """
    if not isinstance(s, str):
        return s
    m = _DATE_PATTERN.match(s.strip())
    if not m:
        return s
    delta = int(m.group(2)) if m.group(2) else 0
    return (date.today() + timedelta(days=delta)).isoformat()
