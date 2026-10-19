"""Map of last-writer-wins registers with tombstoned deletes."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from edgecluster.sync.crdt.register import LWWRegister
from edgecluster.sync.hlc import Timestamp


def entry_value(value: Any) -> dict[str, Any]:
    """Register payload for a live entry."""
    return {"deleted": False, "value": value}


TOMBSTONE: dict[str, Any] = {"deleted": True, "value": None}


@dataclass
class LWWMap:
    """String-keyed map where each key is an independent :class:`LWWRegister`.

    Deletes write a tombstone register rather than dropping the key, so a
    delete propagates and correctly beats older concurrent writes.
    """

    entries: dict[str, LWWRegister] = field(default_factory=dict)

    def set(self, key: str, value: Any, timestamp: Timestamp) -> bool:
        """Write ``value`` under ``key``. Returns True if it became current."""
        return self.entries.setdefault(key, LWWRegister()).assign(entry_value(value), timestamp)

    def delete(self, key: str, timestamp: Timestamp) -> bool:
        """Tombstone ``key``. Returns True if the tombstone became current."""
        return self.entries.setdefault(key, LWWRegister()).assign(dict(TOMBSTONE), timestamp)

    def get(self, key: str, default: Any = None) -> Any:
        """Current value, or ``default`` if missing or deleted."""
        reg = self.entries.get(key)
        if reg is None or reg.value is None or reg.value.get("deleted"):
            return default
        return reg.value["value"]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.get(key, _MISSING) is not _MISSING

    def keys(self) -> Iterator[str]:
        """Live keys."""
        return (k for k in self.entries if k in self)

    def items(self) -> Iterator[tuple[str, Any]]:
        """Live (key, value) pairs."""
        return ((k, self.get(k)) for k in self.keys())

    def merge(self, other: LWWMap) -> LWWMap:
        """Per-key register merge."""
        keys = self.entries.keys() | other.entries.keys()
        merged: dict[str, LWWRegister] = {}
        for k in keys:
            a, b = self.entries.get(k), other.entries.get(k)
            if a is None or b is None:
                src = a or b
                assert src is not None
                merged[k] = LWWRegister(src.value, src.timestamp)
            else:
                merged[k] = a.merge(b)
        return LWWMap(merged)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly state."""
        return {k: reg.to_dict() for k, reg in sorted(self.entries.items())}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LWWMap:
        """Inverse of :meth:`to_dict`."""
        return cls({k: LWWRegister.from_dict(v) for k, v in raw.items()})


_MISSING = object()
