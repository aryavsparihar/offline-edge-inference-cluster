"""Uplink-window scheduling.

Nodes sit at sites with no reliable connectivity. Inter-node links (and the
internet uplink) are only up during configured windows, e.g. two hours each
morning. Sync is attempted only inside a window. Local reads and writes are
always allowed.

Time is supplied by a :class:`Clock`, so the same schedule can run in real
time on a Pi or compressed (``time_scale``) for demos and tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


class Clock(Protocol):
    """Source of (possibly simulated) seconds."""

    def now(self) -> float:
        """Current time in seconds."""
        ...


class SystemClock:
    """Real wall-clock time."""

    def now(self) -> float:
        """Seconds since the Unix epoch."""
        return time.time()


class ScaledClock:
    """Wall clock running ``scale`` times faster than real time from ``origin``.

    With ``scale=1440`` one simulated day passes per real minute.
    """

    def __init__(self, scale: float = 1.0, origin: float | None = None) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = scale
        self._real0 = time.monotonic()
        self._sim0 = time.time() if origin is None else origin

    def now(self) -> float:
        """Simulated seconds."""
        return self._sim0 + (time.monotonic() - self._real0) * self.scale


class ManualClock:
    """Clock advanced explicitly (used by the in-process simulator and tests)."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def now(self) -> float:
        """Current simulated time."""
        return self.t

    def advance(self, seconds: float) -> None:
        """Move time forward."""
        self.t += seconds


def parse_duration(raw: str | int | float) -> float:
    """Parse ``"90"``, ``"90s"``, ``"15m"``, ``"2h"`` or ``"1d"`` into seconds."""
    if isinstance(raw, int | float):
        return float(raw)
    raw = raw.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if raw and raw[-1] in units:
        return float(raw[:-1]) * units[raw[-1]]
    return float(raw)


def parse_offset(raw: str | int | float) -> float:
    """Parse ``"HH:MM"`` / ``"HH:MM:SS"`` (or seconds) into seconds past the period start."""
    if isinstance(raw, str) and ":" in raw:
        parts = [int(p) for p in raw.split(":")]
        while len(parts) < 3:
            parts.append(0)
        h, m, s = parts
        return float(h * 3600 + m * 60 + s)
    return parse_duration(raw)


@dataclass(frozen=True)
class UplinkWindow:
    """A window opening ``start_s`` seconds into each period, lasting ``duration_s``."""

    start_s: float
    duration_s: float


class UplinkSchedule:
    """Repeating set of uplink windows.

    Args:
        windows: Windows within one period. They may wrap past the period end.
        period_s: Length of the repeating period (86400 for daily).
        utc_offset_s: Shift applied to clock time before computing the phase,
            so ``"06:00"`` means 06:00 local time (IST is ``19800``).
        always_open: Ignore windows (useful for a LAN-connected lab cluster).
    """

    def __init__(
        self,
        windows: list[UplinkWindow],
        period_s: float = 86400.0,
        utc_offset_s: float = 0.0,
        always_open: bool = False,
    ) -> None:
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        for w in windows:
            if w.duration_s <= 0 or w.duration_s > period_s:
                raise ValueError(f"invalid window duration {w.duration_s}")
        self.windows = sorted(windows, key=lambda w: w.start_s % period_s)
        self.period_s = period_s
        self.utc_offset_s = utc_offset_s
        self.always_open = always_open

    def _phase(self, t: float) -> float:
        return (t + self.utc_offset_s) % self.period_s

    def is_open(self, t: float) -> bool:
        """Whether a window is open at time ``t``."""
        return self.always_open or self.time_left(t) > 0

    def time_left(self, t: float) -> float:
        """Seconds until the current window closes (0 if closed)."""
        if self.always_open:
            return float("inf")
        phase = self._phase(t)
        best = 0.0
        for w in self.windows:
            start = w.start_s % self.period_s
            # Check this period's and the previous period's occurrence (wrap-around).
            for s in (start, start - self.period_s):
                if s <= phase < s + w.duration_s:
                    best = max(best, s + w.duration_s - phase)
        return best

    def time_until_open(self, t: float) -> float:
        """Seconds until the next window opens (0 if open now)."""
        if self.is_open(t):
            return 0.0
        if not self.windows:
            return float("inf")
        phase = self._phase(t)
        return min(((w.start_s % self.period_s) - phase) % self.period_s for w in self.windows)
