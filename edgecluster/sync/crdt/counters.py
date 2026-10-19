"""State-based counters: grow-only (GCounter) and increment/decrement (PNCounter)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GCounter:
    """Grow-only counter: one non-negative slot per node, merged by max.

    Each node only ever increments its own slot, so the per-slot maximum is
    the latest value that node has published and ``sum`` is the global count.
    """

    counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(v < 0 for v in self.counts.values()):
            raise ValueError("GCounter slots must be non-negative")
        # Zero slots carry no information; dropping them keeps equality canonical.
        self.counts = {k: v for k, v in self.counts.items() if v > 0}

    @property
    def value(self) -> int:
        """Global count (sum over all nodes)."""
        return sum(self.counts.values())

    def increment(self, node_id: str, amount: int = 1) -> None:
        """Add ``amount`` (>= 0) to ``node_id``'s slot."""
        if amount < 0:
            raise ValueError("GCounter can only grow; use PNCounter to decrement")
        if amount:
            self.counts[node_id] = self.counts.get(node_id, 0) + amount

    def merge(self, other: GCounter) -> GCounter:
        """Least upper bound of two states (per-slot max)."""
        keys = self.counts.keys() | other.counts.keys()
        return GCounter({k: max(self.counts.get(k, 0), other.counts.get(k, 0)) for k in keys})

    def to_dict(self) -> dict[str, int]:
        """JSON-friendly state."""
        return dict(sorted(self.counts.items()))

    @classmethod
    def from_dict(cls, raw: dict[str, int]) -> GCounter:
        """Inverse of :meth:`to_dict`."""
        return cls({str(k): int(v) for k, v in raw.items()})


@dataclass
class PNCounter:
    """Counter supporting decrements, built from two GCounters (P - N)."""

    p: GCounter = field(default_factory=GCounter)
    n: GCounter = field(default_factory=GCounter)

    @property
    def value(self) -> int:
        """Increments minus decrements across all nodes."""
        return self.p.value - self.n.value

    def increment(self, node_id: str, amount: int = 1) -> None:
        """Add ``amount``; negative amounts decrement."""
        if amount >= 0:
            self.p.increment(node_id, amount)
        else:
            self.n.increment(node_id, -amount)

    def decrement(self, node_id: str, amount: int = 1) -> None:
        """Subtract ``amount`` (>= 0)."""
        if amount < 0:
            raise ValueError("decrement amount must be non-negative")
        self.n.increment(node_id, amount)

    def merge(self, other: PNCounter) -> PNCounter:
        """Merge both halves independently."""
        return PNCounter(self.p.merge(other.p), self.n.merge(other.n))

    def to_dict(self) -> dict[str, dict[str, int]]:
        """JSON-friendly state."""
        return {"p": self.p.to_dict(), "n": self.n.to_dict()}

    @classmethod
    def from_dict(cls, raw: dict[str, dict[str, int]]) -> PNCounter:
        """Inverse of :meth:`to_dict`."""
        return cls(GCounter.from_dict(raw.get("p", {})), GCounter.from_dict(raw.get("n", {})))
