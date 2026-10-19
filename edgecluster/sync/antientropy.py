"""Delta-based push-pull anti-entropy between two replicas.

One exchange between node A (initiator) and peer B does:

1. **Pull**: A sends ``since`` (the highest B-seq it has merged, plus the
   B ``store_id`` that cursor belongs to). B replies with its changes in
   ``(since, upto]``. A merges them and only then advances its cursor, so a
   lost reply just means the same delta is resent next time.
2. **Push**: A sends its own changes after the last seq B acknowledged.
   B merges and acks. If the ack is lost A resends, which is harmless
   because CRDT merges are idempotent.

Both directions are paginated and bounded per exchange, so a short uplink
window makes partial progress instead of failing outright. Each request is
retried a bounded number of times, because on a lossy link a retry is far
cheaper than waiting for the next round. Callers can piggyback extra fields
(e.g. gossip heartbeats) on pull requests and responses through ``hooks``. The transport is
abstract. :mod:`edgecluster.cluster.transport` provides HTTP, and the
simulator provides an in-process transport with injected faults.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from edgecluster.sync.store import CounterSlot, Entry, ReplicatedStore

log = logging.getLogger(__name__)

PULL, PUSH = "pull", "push"


class TransportError(RuntimeError):
    """A request could not be delivered or its response was lost."""


class PeerTransport(Protocol):
    """Request/response channel from this node to a named peer."""

    def request(self, peer_id: str, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send ``payload`` for operation ``op`` (``pull``/``push``/``gossip``) to ``peer_id``."""
        ...


class SyncHooks(Protocol):
    """Piggyback extra fields on pull messages (used for gossip heartbeats)."""

    def outgoing(self) -> dict[str, Any]:
        """Extra fields to attach to a pull request or response."""
        ...

    def incoming(self, payload: dict[str, Any]) -> None:
        """Consume extra fields from a received pull request or response."""
        ...


class SyncService:
    """Server-side handlers for the anti-entropy protocol."""

    def __init__(self, store: ReplicatedStore, hooks: SyncHooks | None = None) -> None:
        self.store = store
        self.hooks = hooks

    def handle_pull(self, req: dict[str, Any]) -> dict[str, Any]:
        """Return a page of changes newer than the requester's cursor."""
        since = int(req.get("since", 0))
        if req.get("store_id") != self.store.store_id:
            since = 0  # requester's cursor refers to another incarnation of this store
        limit = max(1, min(int(req.get("limit", 500)), 5000))
        # With a valid cursor the requester still holds everything it sent us,
        # so don't echo it back. After a reset (since == 0) send everything.
        requester = req.get("from") if since > 0 else None
        batch = self.store.changes_since(since, limit, exclude_src=requester)
        extras: dict[str, Any] = {}
        if self.hooks is not None:
            self.hooks.incoming(req)
            extras = self.hooks.outgoing()
        return {
            **extras,
            "store_id": batch.store_id,
            "entries": [e.to_wire() for e in batch.entries],
            "counters": [c.to_wire() for c in batch.counters],
            "upto": batch.upto,
            "has_more": batch.has_more,
        }

    def handle_push(self, req: dict[str, Any]) -> dict[str, Any]:
        """Merge pushed changes and acknowledge them."""
        entries = [Entry.from_wire(e) for e in req.get("entries", [])]
        counters = [CounterSlot.from_wire(c) for c in req.get("counters", [])]
        applied = self.store.apply(entries, counters, source=str(req.get("from", "?")))
        return {"store_id": self.store.store_id, "applied": applied, "ack": int(req.get("upto", 0))}

    def handle_digest(self) -> dict[str, Any]:
        """State fingerprint used by monitoring and convergence checks."""
        return {
            "node_id": self.store.node_id,
            "store_id": self.store.store_id,
            "digest": self.store.digest(),
            "records": self.store.count(include_deleted=True),
            "max_seq": self.store.max_seq,
        }


@dataclass
class ExchangeResult:
    """Outcome of one push-pull exchange with a peer."""

    peer_id: str
    ok: bool = True
    pulled: int = 0
    pushed: int = 0
    applied_locally: int = 0
    applied_remotely: int = 0
    complete: bool = True
    errors: list[str] = field(default_factory=list)


class AntiEntropy:
    """Client side of the protocol for one replica.

    Args:
        store: The local replica.
        transport: Channel to peers.
        page_size: Max changes per request.
        max_pages: Max pages per direction per exchange (bounds window usage).
        retries: Extra attempts per request after a transport failure.
        hooks: Optional piggyback hooks (see :class:`SyncHooks`).
    """

    def __init__(
        self,
        store: ReplicatedStore,
        transport: PeerTransport,
        page_size: int = 500,
        max_pages: int = 20,
        retries: int = 2,
        hooks: SyncHooks | None = None,
    ) -> None:
        self.store = store
        self.transport = transport
        self.page_size = page_size
        self.max_pages = max_pages
        self.retries = retries
        self.hooks = hooks

    def _call(self, peer_id: str, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        last: TransportError | None = None
        for _ in range(self.retries + 1):
            try:
                return self.transport.request(peer_id, op, payload)
            except TransportError as exc:
                last = exc
        assert last is not None
        raise last

    def pull(self, peer_id: str, result: ExchangeResult) -> None:
        """Pull the peer's delta into the local store."""
        for _ in range(self.max_pages):
            store_id = self.store.known_store_id(peer_id, PULL)
            since = self.store.get_cursor(peer_id, PULL, store_id)
            req: dict[str, Any] = {
                "from": self.store.node_id,
                "since": since,
                "store_id": store_id,
                "limit": self.page_size,
            }
            if self.hooks is not None:
                req.update(self.hooks.outgoing())
            resp = self._call(peer_id, PULL, req)
            if self.hooks is not None:
                self.hooks.incoming(resp)
            entries = [Entry.from_wire(e) for e in resp["entries"]]
            counters = [CounterSlot.from_wire(c) for c in resp["counters"]]
            result.applied_locally += self.store.apply(entries, counters, source=peer_id)
            result.pulled += len(entries) + len(counters)
            self.store.set_cursor(peer_id, PULL, resp["store_id"], int(resp["upto"]))
            pushed_to = self.store.known_store_id(peer_id, PUSH)
            if pushed_to is not None and pushed_to != resp["store_id"]:
                # The peer's database was replaced: everything we pushed is gone.
                self.store.set_cursor(peer_id, PUSH, resp["store_id"], 0)
            if not resp["has_more"]:
                return
        result.complete = False

    def push(self, peer_id: str, result: ExchangeResult) -> None:
        """Push the local delta the peer has not acknowledged yet."""
        for _ in range(self.max_pages):
            peer_store = self.store.known_store_id(peer_id, PUSH)
            since = self.store.get_cursor(peer_id, PUSH, peer_store)
            batch = self.store.changes_since(since, self.page_size, exclude_src=peer_id if since > 0 else None)
            if not batch.entries and not batch.counters and peer_store is not None:
                # Nothing to send in this slice (empty, or all of it came from the peer).
                self.store.set_cursor(peer_id, PUSH, peer_store, batch.upto)
                if batch.has_more:
                    continue
                return
            resp = self._call(
                peer_id,
                PUSH,
                {
                    "from": self.store.node_id,
                    "store_id": self.store.store_id,
                    "entries": [e.to_wire() for e in batch.entries],
                    "counters": [c.to_wire() for c in batch.counters],
                    "upto": batch.upto,
                },
            )
            result.pushed += len(batch.entries) + len(batch.counters)
            result.applied_remotely += int(resp.get("applied", 0))
            # If the peer's store was replaced since our last ack, start over from 0.
            if peer_store is not None and resp["store_id"] != peer_store:
                self.store.set_cursor(peer_id, PUSH, resp["store_id"], 0)
                continue
            self.store.set_cursor(peer_id, PUSH, resp["store_id"], int(resp["ack"]))
            if not batch.has_more:
                return
        result.complete = False

    def exchange(self, peer_id: str) -> ExchangeResult:
        """Run pull then push with ``peer_id``; transport faults are reported, not raised."""
        result = ExchangeResult(peer_id)
        for step in (self.pull, self.push):
            try:
                step(peer_id, result)
            except TransportError as exc:
                result.ok = False
                result.complete = False
                result.errors.append(f"{step.__name__}: {exc}")
                log.debug("sync %s -> %s %s failed: %s", self.store.node_id, peer_id, step.__name__, exc)
        return result
