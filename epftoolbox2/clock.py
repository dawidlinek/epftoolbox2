"""The clock every time read and every wait of epftoolbox2 goes through, for tests and simulations.

    from epftoolbox2.clock import FakeClock, use_clock

    clock = FakeClock("2026-10-21 23:30")
    with use_clock(clock):
        ...  # every "now", "today", cache boundary, Retry-After wait and manifest timestamp reads `clock`

A `FakeClock` stands still until it is advanced (`advance`, or `sleep`, which returns at once), so code that waits on it
never waits. Outside a `use_clock` block the clock is `SYSTEM`, the machine's.
"""

from ._clock import SYSTEM, Clock, FakeClock, current_clock, use_clock

__all__ = ["Clock", "SYSTEM", "FakeClock", "use_clock", "current_clock"]
