"""HTTP transport for the anti-entropy and gossip protocols."""

from __future__ import annotations

import threading
from typing import Any

import httpx

from edgecluster.sync.antientropy import TransportError

NODE_HEADER = "X-Node-Id"


class LinkControl:
    """Set of peers whose link is administratively down.

    Used to emulate a partition on a real local cluster (``POST /admin/links``):
    outgoing requests to a blocked peer fail, and incoming requests from it are
    rejected, exactly as if the radio link between the two sites were down.
    """

    def __init__(self) -> None:
        self._blocked: set[str] = set()
        self._lock = threading.Lock()

    def set_blocked(self, peers: set[str]) -> None:
        """Replace the blocked set."""
        with self._lock:
            self._blocked = set(peers)

    def blocked(self) -> set[str]:
        """Currently blocked peers."""
        with self._lock:
            return set(self._blocked)

    def is_blocked(self, peer_id: str | None) -> bool:
        """Whether traffic to/from ``peer_id`` is blocked."""
        with self._lock:
            return peer_id is not None and peer_id in self._blocked


class HttpTransport:
    """Sends protocol requests to peers' ``/sync/{op}`` endpoints.

    Args:
        node_id: This node (sent in the ``X-Node-Id`` header).
        peers: Mapping of peer id to base URL.
        timeout_s: Per-request timeout. Keep it short: a slow link inside a
            window should fail fast and be retried in the next round.
        links: Optional link control for emulated partitions.
    """

    def __init__(
        self,
        node_id: str,
        peers: dict[str, str],
        timeout_s: float = 5.0,
        links: LinkControl | None = None,
    ) -> None:
        self.node_id = node_id
        self.peers = {k: v.rstrip("/") for k, v in peers.items()}
        self.links = links or LinkControl()
        self._client = httpx.Client(timeout=timeout_s, headers={NODE_HEADER: node_id})

    def request(self, peer_id: str, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``payload`` to the peer; any failure becomes :class:`TransportError`."""
        if self.links.is_blocked(peer_id):
            raise TransportError(f"link to {peer_id} is down")
        url = self.peers.get(peer_id)
        if url is None:
            raise TransportError(f"unknown peer {peer_id}")
        try:
            resp = self._client.post(f"{url}/sync/{op}", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise TransportError(f"{op} to {peer_id} failed: {exc}") from exc
        if not isinstance(data, dict):
            raise TransportError(f"{op} to {peer_id}: malformed response")
        return data

    def close(self) -> None:
        """Release the connection pool."""
        self._client.close()
