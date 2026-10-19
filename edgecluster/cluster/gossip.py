"""Heartbeat gossip for failure detection and peer selection.

Each node increments its own heartbeat counter every round and exchanges the
whole heartbeat map with the peers it syncs with. Maps merge by per-node max,
the same join as a G-Counter, so heartbeats spread transitively through the
cluster. A peer whose heartbeat has not increased for ``suspect_after_s`` is
*suspected*. Suspected peers are still probed occasionally, because in this
deployment silence usually means a closed uplink, not a dead node.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class PeerState:
    """What this node knows about one peer."""

    heartbeat: int = 0
    last_change: float = float("-inf")


class Membership:
    """Static membership with gossip-based liveness.

    Args:
        node_id: This node.
        peers: Ids of all other configured nodes (static config).
        now: Clock in seconds (simulated or real).
        suspect_after_s: Time without heartbeat progress before suspecting a peer.
        probe_suspected: Probability of including one suspected peer per round.
    """

    def __init__(
        self,
        node_id: str,
        peers: list[str],
        now: Callable[[], float],
        suspect_after_s: float = 60.0,
        probe_suspected: float = 0.3,
    ) -> None:
        self.node_id = node_id
        self.peers = [p for p in peers if p != node_id]
        self._now = now
        self.suspect_after_s = suspect_after_s
        self.probe_suspected = probe_suspected
        self.state: dict[str, PeerState] = {n: PeerState() for n in [node_id, *self.peers]}

    def tick(self) -> None:
        """Advance our own heartbeat (called once per sync round)."""
        me = self.state[self.node_id]
        me.heartbeat += 1
        me.last_change = self._now()

    def heartbeats(self) -> dict[str, int]:
        """Heartbeat map to send to a peer."""
        return {n: s.heartbeat for n, s in self.state.items() if s.heartbeat > 0}

    def merge(self, remote: dict[str, int]) -> None:
        """Merge a peer's heartbeat map (per-node max)."""
        now = self._now()
        for node, hb in remote.items():
            st = self.state.get(node)
            if st is None or node == self.node_id:
                continue  # ignore unknown nodes: membership is static
            if hb > st.heartbeat:
                st.heartbeat, st.last_change = hb, now

    def is_alive(self, node: str) -> bool:
        """Whether ``node`` showed heartbeat progress recently."""
        st = self.state.get(node)
        return st is not None and self._now() - st.last_change <= self.suspect_after_s

    def alive_peers(self) -> list[str]:
        """Peers currently believed alive."""
        return [p for p in self.peers if self.is_alive(p)]

    def choose_peers(self, fanout: int, rng: random.Random) -> list[str]:
        """Pick up to ``fanout`` sync partners, preferring live peers."""
        alive = self.alive_peers()
        suspected = [p for p in self.peers if p not in alive]
        rng.shuffle(alive)
        rng.shuffle(suspected)
        chosen = alive[:fanout]
        if suspected and (len(chosen) < fanout or rng.random() < self.probe_suspected):
            if len(chosen) >= fanout:
                chosen = chosen[: fanout - 1] if fanout > 1 else []
            chosen += suspected[: fanout - len(chosen)]
        return chosen
