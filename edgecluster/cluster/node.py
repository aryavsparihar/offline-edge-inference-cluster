"""Glue between the replicated store, gossip and the uplink schedule."""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from typing import Any

from edgecluster.cluster.gossip import Membership
from edgecluster.cluster.scheduler import Clock, UplinkSchedule
from edgecluster.sync.antientropy import AntiEntropy, ExchangeResult, PeerTransport, SyncService
from edgecluster.sync.store import ReplicatedStore


class HeartbeatHooks:
    """Piggybacks the gossip heartbeat map on every pull request/response."""

    def __init__(self, membership: Membership) -> None:
        self.membership = membership

    def outgoing(self) -> dict[str, Any]:
        """Attach our heartbeat map."""
        return {"heartbeats": self.membership.heartbeats()}

    def incoming(self, payload: dict[str, Any]) -> None:
        """Merge a peer's heartbeat map."""
        hb = payload.get("heartbeats")
        if isinstance(hb, dict):
            self.membership.merge({str(k): int(v) for k, v in hb.items()})


@dataclass
class SyncStats:
    """Cumulative counters exported on ``/metrics``."""

    rounds: int = 0
    rounds_skipped_closed: int = 0
    exchanges_ok: int = 0
    exchanges_failed: int = 0
    entries_pulled: int = 0
    entries_pushed: int = 0
    last_round: list[ExchangeResult] = field(default_factory=list)


class SyncNode:
    """A node's replication runtime, shared by the HTTP service and the simulator.

    Args:
        store: Local replica.
        transport: Channel to peers.
        membership: Gossip-based liveness view.
        schedule: Uplink windows. Rounds outside a window are skipped.
        clock: Time source the schedule is evaluated against.
        fanout: Peers contacted per round.
        rng: Randomness for peer selection (seeded in simulation).
        page_size: Changes per sync request.
        max_pages: Pages per direction per exchange.
        retries: Extra attempts per request on transport failure.
    """

    def __init__(
        self,
        store: ReplicatedStore,
        transport: PeerTransport,
        membership: Membership,
        schedule: UplinkSchedule,
        clock: Clock,
        fanout: int = 2,
        rng: random.Random | None = None,
        page_size: int = 500,
        max_pages: int = 20,
        retries: int = 2,
    ) -> None:
        self.store = store
        self.transport = transport
        self.membership = membership
        self.schedule = schedule
        self.clock = clock
        self.fanout = fanout
        self.rng = rng or random.Random()
        hooks = HeartbeatHooks(membership)
        self.service = SyncService(store, hooks)
        self.engine = AntiEntropy(
            store, transport, page_size=page_size, max_pages=max_pages, retries=retries, hooks=hooks
        )
        self.stats = SyncStats()
        self._round_lock = threading.Lock()

    @property
    def node_id(self) -> str:
        """This node's id."""
        return self.store.node_id

    def handle(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Serve an incoming protocol request."""
        if op == "pull":
            return self.service.handle_pull(payload)
        if op == "push":
            return self.service.handle_push(payload)
        if op == "digest":
            return self.service.handle_digest()
        raise ValueError(f"unknown op {op!r}")

    def run_round(self, force: bool = False) -> list[ExchangeResult]:
        """One anti-entropy round (heartbeats piggybacked), if the uplink window is open (or ``force``)."""
        with self._round_lock:
            if not force and not self.schedule.is_open(self.clock.now()):
                self.stats.rounds_skipped_closed += 1
                return []
            self.membership.tick()
            results = []
            for peer in self.membership.choose_peers(self.fanout, self.rng):
                res = self.engine.exchange(peer)
                results.append(res)
                self.stats.entries_pulled += res.pulled
                self.stats.entries_pushed += res.pushed
                if res.ok:
                    self.stats.exchanges_ok += 1
                else:
                    self.stats.exchanges_failed += 1
            self.stats.rounds += 1
            self.stats.last_round = results
            return results
