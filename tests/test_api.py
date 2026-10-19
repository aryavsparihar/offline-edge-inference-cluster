"""HTTP API of a single node (FastAPI TestClient, sync loop disabled)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edgecluster.config import load_config
from edgecluster.serving.app import create_app, text_fingerprint
from edgecluster.serving.classifier import OnnxClassifier


@pytest.fixture
def client(tmp_path: Path, tiny_model_dir: Path) -> Iterator[TestClient]:
    cfg = load_config(
        None,
        node_id="node1",
        data_dir=str(tmp_path),
        admin_enabled=True,
        uplink={"always_open": True},
        peers=[{"id": "node1", "url": "http://x"}, {"id": "node2", "url": "http://127.0.0.1:9"}],
    )
    clf = OnnxClassifier(tiny_model_dir, "model.onnx", intra_op_threads=1)
    with TestClient(create_app(cfg, classifier=clf, start_sync_loop=False)) as c:
        yield c


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["node_id"] == "node1"
    assert body["classifier_loaded"] is True
    assert body["uplink_open"] is True


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("मेरी पेंशन नहीं आई", "pension"),
        ("راشن کارڈ گم ہو گیا", "ration_card"),
        ("आधार कार्ड में पता बदलना है", "documents_id"),
        ("namaste", "other"),
    ],
)
def test_classify(client: TestClient, text: str, label: str) -> None:
    r = client.post("/classify", json={"text": text, "lang": "hi"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == label
    assert abs(sum(body["scores"].values()) - 1.0) < 1e-4
    assert body["latency_ms"] >= body["inference_ms"] >= 0


def test_classify_logs_pseudonymous_query_and_counter(client: TestClient) -> None:
    text = "मेरी पेंशन नहीं आई"
    qid = client.post("/classify", json={"text": text, "lang": "hi"}).json()["query_id"]
    rec = client.get(f"/records/queries/{qid}").json()["value"]
    assert rec["label"] == "pension"
    assert rec["text_sha256"] == text_fingerprint(text)
    assert "text" not in rec  # raw text is not replicated by default
    assert client.get("/counters").json()["queries:pension"] == 1
    client.post("/classify", json={"text": text, "log": False})
    assert client.get("/counters").json()["queries:pension"] == 1


def test_classify_validation(client: TestClient) -> None:
    assert client.post("/classify", json={"text": ""}).status_code == 422
    assert client.post("/classify", json={}).status_code == 422


def test_classify_without_model_is_503(tmp_path: Path) -> None:
    cfg = load_config(None, node_id="n", data_dir=str(tmp_path))
    with TestClient(create_app(cfg, start_sync_loop=False)) as c:
        assert c.post("/classify", json={"text": "x"}).status_code == 503
        assert c.get("/health").json()["classifier_loaded"] is False


def test_generate_route_absent_when_disabled(client: TestClient) -> None:
    assert client.post("/generate", json={"prompt": "hi"}).status_code == 404


def test_metrics_exposes_latency_histogram(client: TestClient) -> None:
    client.post("/classify", json={"text": "पेंशन"})
    text = client.get("/metrics").text
    assert "edge_classify_latency_seconds_bucket" in text
    assert "edge_classify_latency_seconds_count 1.0" in text
    assert "edge_inference_latency_seconds_count 1.0" in text
    assert "edge_process_rss_bytes" in text
    assert 'edge_store_records{collection="queries"} 1.0' in text


def test_records_crud(client: TestClient) -> None:
    r = client.post("/records/cases", json={"key": "c1", "value": {"status": "open", "scheme": "housing"}})
    assert r.status_code == 200
    assert client.get("/records/cases/c1").json()["value"]["status"] == "open"
    client.post("/records/cases", json={"key": "c1", "value": {"status": "approved"}})
    assert client.get("/records/cases/c1").json()["value"] == {"status": "approved"}
    listing = client.get("/records/cases").json()
    assert listing["total"] == 1
    assert client.delete("/records/cases/c1").json()["deleted"] is True
    assert client.get("/records/cases/c1").status_code == 404
    assert client.post("/records/nope", json={"value": {}}).status_code == 404


def test_sync_endpoints(client: TestClient) -> None:
    client.post("/records/cases", json={"key": "c1", "value": {"v": 1}})
    pull = client.post("/sync/pull", json={"from": "node2", "since": 0, "limit": 10}).json()
    assert [e["k"] for e in pull["entries"]] == ["c1"]
    assert pull["has_more"] is False
    push = client.post(
        "/sync/push",
        json={
            "from": "node2",
            "store_id": "x",
            "entries": [{"c": "cases", "k": "c2", "v": {"v": 2}, "d": False, "ts": [1, 0, "node2"]}],
            "counters": [],
            "upto": 1,
        },
    ).json()
    assert push["applied"] == 1
    assert client.get("/records/cases/c2").json()["value"] == {"v": 2}
    assert client.get("/sync/digest").json()["records"] == 2
    assert client.post("/sync/bogus", json={}).status_code == 404


def test_blocked_link_rejects_peer(client: TestClient) -> None:
    client.post("/admin/links", json={"blocked": ["node2"]})
    r = client.post("/sync/pull", json={"since": 0}, headers={"X-Node-Id": "node2"})
    assert r.status_code == 503
    assert client.get("/cluster").json()["blocked_links"] == ["node2"]


def test_sync_now_with_unreachable_peer_reports_failure(client: TestClient) -> None:
    body = client.post("/admin/sync-now").json()
    assert body["exchanges"] and body["exchanges"][0]["ok"] is False
