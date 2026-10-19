"""Observed-remove set with add-wins semantics."""

from __future__ import annotations

from dataclasses import dataclass, field

Tag = tuple[str, int]


@dataclass
class ORSet:
    """Add-wins observed-remove set of strings (tombstone variant).

    Every ``add`` attaches a globally unique tag ``(node_id, n)``. ``remove``
    tombstones only the tags it has *observed*, so an add that is concurrent
    with a remove survives the merge (add wins). State is the pair
    ``(adds, removed)``; merge is set union on both, which is trivially
    commutative, associative and idempotent.
    """

    adds: dict[str, frozenset[Tag]] = field(default_factory=dict)
    removed: frozenset[Tag] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        self.adds = {e: frozenset(t) for e, t in self.adds.items() if t}
        self.removed = frozenset(self.removed)

    def _next_tag(self, node_id: str) -> Tag:
        seen = [n for tags in self.adds.values() for (nid, n) in tags if nid == node_id]
        seen += [n for (nid, n) in self.removed if nid == node_id]
        return (node_id, max(seen, default=0) + 1)

    def add(self, element: str, node_id: str) -> Tag:
        """Add ``element`` with a fresh tag minted by ``node_id``."""
        tag = self._next_tag(node_id)
        self.adds[element] = self.adds.get(element, frozenset()) | {tag}
        return tag

    def remove(self, element: str) -> None:
        """Remove ``element`` as currently observed; a no-op if absent."""
        self.removed = self.removed | self.adds.get(element, frozenset())

    def __contains__(self, element: object) -> bool:
        return isinstance(element, str) and bool(self.adds.get(element, frozenset()) - self.removed)

    @property
    def value(self) -> frozenset[str]:
        """Elements with at least one live (non-tombstoned) tag."""
        return frozenset(e for e, tags in self.adds.items() if tags - self.removed)

    def merge(self, other: ORSet) -> ORSet:
        """Union of add-tags and of tombstones."""
        keys = self.adds.keys() | other.adds.keys()
        adds = {k: self.adds.get(k, frozenset()) | other.adds.get(k, frozenset()) for k in keys}
        return ORSet(adds, self.removed | other.removed)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly state (sorted for determinism)."""
        return {
            "adds": {e: sorted([list(t) for t in tags]) for e, tags in sorted(self.adds.items())},
            "removed": sorted([list(t) for t in self.removed]),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> ORSet:
        """Inverse of :meth:`to_dict`."""
        adds_raw = raw.get("adds", {})
        removed_raw = raw.get("removed", [])
        assert isinstance(adds_raw, dict) and isinstance(removed_raw, list)
        adds = {str(e): frozenset((str(n), int(c)) for n, c in tags) for e, tags in adds_raw.items()}
        return cls(adds, frozenset((str(n), int(c)) for n, c in removed_raw))
