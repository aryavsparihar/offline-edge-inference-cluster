"""Hybrid logical clocks (Kulkarni et al., 2014).

A Raspberry Pi has no real-time clock. Without NTP (which needs the uplink)
its wall clock can jump backwards after a reboot (``fake-hwclock`` restores
the last saved time) or drift by minutes between nodes. Last-writer-wins
CRDTs ordered by raw wall time would then silently drop newer writes.

An HLC timestamp is ``(wall_ms, logical, node_id)``:

* ``wall_ms`` tracks the largest physical time seen, locally or from a peer,
  so it never goes backwards even if the local clock does;
* ``logical`` breaks ties between events in the same millisecond and
  preserves causality (a received timestamp is always smaller than any
  timestamp generated afterwards);
* ``node_id`` makes the order total, so concurrent writes on two nodes
  resolve the same way on every replica.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


class ClockDriftError(RuntimeError):
    """Raised when a remote timestamp is further ahead than ``max_drift_ms``."""


@dataclass(frozen=True, order=True)
class Timestamp:
    """A totally ordered HLC timestamp. Field order defines the sort order."""

    wall_ms: int
    logical: int
    node_id: str

    def encode(self) -> str:
        """Encode as a string that sorts lexicographically in timestamp order."""
        return f"{self.wall_ms:015d}:{self.logical:06d}:{self.node_id}"

    @classmethod
    def decode(cls, raw: str) -> Timestamp:
        """Inverse of :meth:`encode`."""
        wall, logical, node_id = raw.split(":", 2)
        return cls(int(wall), int(logical), node_id)

    def to_list(self) -> list[int | str]:
        """JSON-friendly representation."""
        return [self.wall_ms, self.logical, self.node_id]

    @classmethod
    def from_list(cls, raw: list[int | str] | tuple[int | str, ...]) -> Timestamp:
        """Inverse of :meth:`to_list`."""
        wall, logical, node_id = raw
        return cls(int(wall), int(logical), str(node_id))


ZERO = Timestamp(0, 0, "")


def wall_clock_ms() -> int:
    """Current physical time in milliseconds since the Unix epoch."""
    return time.time_ns() // 1_000_000


class HybridLogicalClock:
    """Thread-safe HLC for a single node.

    Args:
        node_id: Unique, stable identifier of this node.
        physical_ms: Source of physical time (injectable for tests/simulation).
        max_drift_ms: If set, :meth:`update` rejects remote timestamps that are
            more than this far ahead of local physical time. ``None`` disables
            the check, which is the sensible default for clusters that may run
            for days without NTP.
        last: Last issued timestamp, e.g. restored from disk after a restart,
            so that timestamps stay monotonic across reboots.
    """

    def __init__(
        self,
        node_id: str,
        physical_ms: Callable[[], int] = wall_clock_ms,
        max_drift_ms: int | None = None,
        last: Timestamp | None = None,
    ) -> None:
        if not node_id or ":" in node_id:
            raise ValueError("node_id must be non-empty and must not contain ':'")
        self.node_id = node_id
        self._physical_ms = physical_ms
        self._max_drift_ms = max_drift_ms
        self._wall = last.wall_ms if last else 0
        self._logical = last.logical if last else 0
        self._lock = threading.Lock()

    @property
    def last(self) -> Timestamp:
        """The most recently issued (or observed) timestamp."""
        with self._lock:
            return Timestamp(self._wall, self._logical, self.node_id)

    def now(self) -> Timestamp:
        """Issue a timestamp for a local event (a write or a message send)."""
        with self._lock:
            pt = self._physical_ms()
            if pt > self._wall:
                self._wall, self._logical = pt, 0
            else:
                self._logical += 1
            return Timestamp(self._wall, self._logical, self.node_id)

    def update(self, remote: Timestamp) -> Timestamp:
        """Merge a timestamp received from a peer and issue a new local one."""
        with self._lock:
            pt = self._physical_ms()
            if self._max_drift_ms is not None and remote.wall_ms - pt > self._max_drift_ms:
                raise ClockDriftError(
                    f"remote clock {remote.wall_ms} is {remote.wall_ms - pt} ms ahead "
                    f"of local {pt} (max {self._max_drift_ms})"
                )
            old_wall = self._wall
            new_wall = max(old_wall, remote.wall_ms, pt)
            if new_wall == old_wall and new_wall == remote.wall_ms:
                self._logical = max(self._logical, remote.logical) + 1
            elif new_wall == old_wall:
                self._logical += 1
            elif new_wall == remote.wall_ms:
                self._logical = remote.logical + 1
            else:
                self._logical = 0
            self._wall = new_wall
            return Timestamp(self._wall, self._logical, self.node_id)
