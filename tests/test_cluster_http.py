"""Three real uvicorn nodes syncing over HTTP, with an emulated partition."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn

from edgecluster.config import load_config
from edgecluster.serving.app import create_app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ServerThread:
    def __init__(self, app, port: int) -> None:  # noqa: ANN001
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        while not self.server.started:
            time.sleep(0.02)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture
def cluster(tmp_path: Path) -> Iterator[list[str]]:
    ports = [free_port() for _ in range(3)]
    peers = [{"id": f"n{i}", "url": f"http://127.0.0.1:{p}"} for i, p in enumerate(ports, 1)]
    servers = []
    for i, port in enumerate(ports, 1):
        cfg = load_config(
            None,
            node_id=f"n{i}",
            data_dir=str(tmp_path / f"n{i}"),
            peers=peers,
            admin_enabled=True,
            uplink={"always_open": True},
            sync={"interval": 0.2, "fanout": 2, "timeout": 2},
        )
        s = ServerThread(create_app(cfg), port)
        s.start()
        servers.append(s)
    try:
        yield [p["url"] for p in peers]
    finally:
        for s in servers:
            s.stop()


def digests(c: httpx.Client, urls: list[str]) -> set[str]:
    return {c.get(f"{u}/sync/digest").json()["digest"] for u in urls}


def wait_converged(c: httpx.Client, urls: list[str], timeout: float = 20) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(digests(c, urls)) == 1:
            return
        time.sleep(0.2)
    raise AssertionError("cluster did not converge")


def test_three_nodes_converge_after_partition(cluster: list[str]) -> None:
    c = httpx.Client(timeout=5)
    for i, u in enumerate(cluster, 1):
        for j in range(10):
            c.post(f"{u}/records/queries", json={"key": f"q{i}-{j}", "value": {"i": i, "j": j}})
    wait_converged(c, cluster)
    assert all(c.get(f"{u}/records/queries").json()["total"] == 30 for u in cluster)

    # Partition n3 away from n1/n2 and write conflicting versions of one case on both sides.
    c.post(f"{cluster[0]}/admin/links", json={"blocked": ["n3"]})
    c.post(f"{cluster[1]}/admin/links", json={"blocked": ["n3"]})
    c.post(f"{cluster[2]}/admin/links", json={"blocked": ["n1", "n2"]})
    c.post(f"{cluster[0]}/records/cases", json={"key": "case", "value": {"status": "pending"}})
    time.sleep(0.01)
    c.post(f"{cluster[2]}/records/cases", json={"key": "case", "value": {"status": "approved"}})
    c.post(f"{cluster[2]}/records/queries", json={"key": "only-n3", "value": {}})
    time.sleep(1.0)
    assert len(digests(c, cluster)) == 2  # majority side converged, minority diverged
    assert c.get(f"{cluster[0]}/records/queries/only-n3").status_code == 404

    for u in cluster:
        c.post(f"{u}/admin/links", json={"blocked": []})
    wait_converged(c, cluster)
    values = [c.get(f"{u}/records/cases/case").json()["value"] for u in cluster]
    assert values == [{"status": "approved"}] * 3  # later HLC timestamp wins everywhere
    assert all(c.get(f"{u}/records/queries/only-n3").status_code == 200 for u in cluster)
    alive = c.get(f"{cluster[0]}/cluster").json()["peers"]
    assert alive["n2"]["alive"] and alive["n3"]["alive"]
