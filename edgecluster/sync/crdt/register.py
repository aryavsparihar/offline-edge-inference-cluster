"""Last-writer-wins register ordered by hybrid logical clock timestamps."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from edgecluster.sync.hlc import ZERO, Timestamp


def canonical_json(value: Any) -> str:
    """Deterministic JSON encoding used for tie-breaking and hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class LWWRegister:
    """A single value; the write with the greatest timestamp wins.

    HLC timestamps include the node id, so two distinct nodes never produce the
    same timestamp. The canonical JSON of the value is used as a final
    tie-breaker anyway, which keeps ``merge`` commutative even for states that
    could only arise from a misconfigured cluster (two nodes sharing an id).
    """

    value: Any = None
    timestamp: Timestamp = field(default=ZERO)

    def _key(self) -> tuple[Timestamp, str]:
        return (self.timestamp, canonical_json(self.value))

    def assign(self, value: Any, timestamp: Timestamp) -> bool:
        """Write ``value`` at ``timestamp``. Returns True if it became current."""
        candidate = LWWRegister(value, timestamp)
        if candidate._key() > self._key():
            self.value, self.timestamp = value, timestamp
            return True
        return False

    def merge(self, other: LWWRegister) -> LWWRegister:
        """Return whichever state has the larger (timestamp, value) key."""
        winner = self if self._key() >= other._key() else other
        return LWWRegister(winner.value, winner.timestamp)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly state."""
        return {"value": self.value, "ts": self.timestamp.to_list()}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LWWRegister:
        """Inverse of :meth:`to_dict`."""
        return cls(raw.get("value"), Timestamp.from_list(raw["ts"]))
