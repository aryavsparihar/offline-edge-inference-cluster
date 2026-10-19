"""Anti-entropy protocol over a direct (optionally lossy) transport."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from edgecluster.sync.antientropy import AntiEntropy, SyncService, TransportError
from edgecluster.sync.store import ReplicatedStore


class DirectTransport:
    """Routes requests to SyncService objects, dropping messages at random."""

    def __init__(self, services: dict[str, SyncService], loss: float = 0.0, seed: int = 0) -> None:
        self.services = services
        self.loss = loss
        self.rng = random.Random(seed)

    def request(self, peer_id: str, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.rng.random() < self.loss:
            raise TransportError("request lost")
        svc = self.services[peer_id]
        payload = json.loads(json.dumps(payload))
        resp = svc.handle_pull(payload) if op == "pull" else svc.handle_push(payload)
        if self.rng.random() < self.loss:
            raise TransportError("response lost")
        return json.loads(json.dumps(resp))


def make(tmp_path: Path, names: list[str]) -> dict[str, ReplicatedStore]:
    return {n: ReplicatedStore(tmp_path / f"{n}.db", n) for n in names}


def test_two_node_push_pull(tmp_path: Path) -> None:
    stores = make(tmp_path, ["a", "b"])
    services = {n: SyncService(s) for n, s in stores.items()}
    for i in range(30):
        stores["a"].put("queries", f"a{i}", {"i": i})
        stores["b"].put("queries", f"b{i}", {"i": i})
    stores["a"].increment("queries:pension", 4)
    ae = AntiEntropy(stores["a"], DirectTransport(services), page_size=7)
    res = ae.exchange("b")
    assert res.ok and res.complete
    assert stores["a"].digest() == stores["b"].digest()
    assert stores["b"].count("queries") == 60
    # a second exchange ships nothing new
    res2 = ae.exchange("b")
    assert res2.pulled == 0 and res2.pushed == 0


def test_lossy_links_eventually_converge(tmp_path: Path) -> None:
    names = ["a", "b", "c", "d"]
    stores = make(tmp_path, names)
    services = {n: SyncService(s) for n, s in stores.items()}
    rng = random.Random(3)
    for n in names:
        for i in range(25):
            stores[n].put("cases", f"case-{rng.randrange(20)}", {"by": n, "i": i})
            stores[n].put("queries", f"{n}-{i}", {"i": i})
    engines = {
        n: AntiEntropy(stores[n], DirectTransport(services, loss=0.4, seed=i), page_size=10, retries=0)
        for i, n in enumerate(names)
    }
    for _ in range(300):
        a, b = rng.sample(names, 2)
        engines[a].exchange(b)
    digests = {stores[n].digest() for n in names}
    assert len(digests) == 1


def test_peer_store_wipe_resets_cursors(tmp_path: Path) -> None:
    stores = make(tmp_path, ["a", "b"])
    services = {n: SyncService(s) for n, s in stores.items()}
    stores["a"].put("cases", "c1", 1)
    ae = AntiEntropy(stores["a"], DirectTransport(services))
    ae.exchange("b")
    assert stores["b"].get("cases", "c1") == 1
    # b's disk is wiped (SD card replaced): new store with a new store_id
    stores["b"].close()
    for suffix in ("", "-wal", "-shm"):
        (tmp_path / f"b.db{suffix}").unlink(missing_ok=True)
    stores["b"] = ReplicatedStore(tmp_path / "b.db", "b")
    services["b"] = SyncService(stores["b"])
    ae.exchange("b")
    assert stores["b"].get("cases", "c1") == 1
