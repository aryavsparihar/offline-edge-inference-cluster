"""FastAPI node service: inference, replicated records, and the sync protocol."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import random
import time
import unicodedata
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from edgecluster import __version__
from edgecluster.cluster.gossip import Membership
from edgecluster.cluster.node import SyncNode
from edgecluster.cluster.scheduler import Clock, ScaledClock, SystemClock, parse_duration
from edgecluster.cluster.transport import NODE_HEADER, HttpTransport, LinkControl
from edgecluster.config import NodeConfig
from edgecluster.serving.classifier import OnnxClassifier
from edgecluster.serving.generator import OnnxGenerator
from edgecluster.serving.metrics import NodeMetrics
from edgecluster.sync.store import ReplicatedStore

log = logging.getLogger(__name__)

QUERY_COLLECTION = "queries"
CASE_COLLECTION = "cases"
WRITABLE_COLLECTIONS = {CASE_COLLECTION, QUERY_COLLECTION}


class ClassifyRequest(BaseModel):
    """Body of ``POST /classify``."""

    text: str = Field(min_length=1, max_length=2000)
    lang: str | None = Field(default=None, description="hi | ur | doi (informational)")
    log: bool = Field(default=True, description="record a replicated query-log entry")


class GenerateRequest(BaseModel):
    """Body of ``POST /generate``."""

    prompt: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int | None = Field(default=None, ge=1, le=256)


class RecordWrite(BaseModel):
    """Body of ``POST /records/{collection}``."""

    key: str | None = Field(default=None, max_length=200)
    value: dict[str, Any]


class LinksRequest(BaseModel):
    """Body of ``POST /admin/links``: peers whose link should be down."""

    blocked: list[str] = Field(default_factory=list)


def text_fingerprint(text: str) -> str:
    """SHA-256 of NFC-normalized, whitespace-collapsed text.

    Pseudonymous, not anonymous: short queries can be guessed by brute force,
    so the raw text is only replicated when ``store_query_text`` is enabled.
    """
    norm = " ".join(unicodedata.normalize("NFC", text).split()).casefold()
    return hashlib.sha256(norm.encode()).hexdigest()


class NodeState:
    """Long-lived objects shared by request handlers."""

    def __init__(self, cfg: NodeConfig) -> None:
        self.cfg = cfg
        self.classifier: OnnxClassifier | None = None
        self.generator: OnnxGenerator | None = None
        self.node: SyncNode | None = None
        self.links = LinkControl()
        self.metrics = NodeMetrics(lambda: self.node)
        self.started = time.time()


def build_clock(cfg: NodeConfig) -> Clock:
    """Real clock, or a compressed one for demos."""
    return SystemClock() if cfg.uplink.time_scale == 1.0 else ScaledClock(cfg.uplink.time_scale)


def build_sync_node(cfg: NodeConfig, links: LinkControl, clock: Clock) -> SyncNode:
    """Open the replica and wire the HTTP sync runtime."""
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    drift = parse_duration(cfg.sync.max_drift) if cfg.sync.max_drift is not None else None
    store = ReplicatedStore(cfg.db_path, cfg.node_id, max_drift_ms=int(drift * 1000) if drift else None)
    peers = cfg.peer_urls()
    transport = HttpTransport(cfg.node_id, peers, parse_duration(cfg.sync.timeout), links)
    membership = Membership(cfg.node_id, list(peers), clock.now, suspect_after_s=parse_duration(cfg.sync.suspect_after))
    return SyncNode(
        store,
        transport,
        membership,
        cfg.uplink.schedule(),
        clock,
        fanout=cfg.sync.fanout,
        rng=random.Random(),
        page_size=cfg.sync.page_size,
        max_pages=cfg.sync.max_pages,
    )


async def sync_loop(node: SyncNode, cfg: NodeConfig, stop: asyncio.Event) -> None:
    """Run sync rounds while an uplink window is open; sleep otherwise."""
    scale = cfg.uplink.time_scale
    interval = parse_duration(cfg.sync.interval)
    while not stop.is_set():
        now = node.clock.now()
        if node.schedule.is_open(now):
            try:
                await asyncio.to_thread(node.run_round)
            except Exception:  # keep the loop alive; the next round retries
                log.exception("sync round failed")
            wait = interval / scale
        else:
            wait = min(node.schedule.time_until_open(now) / scale, 60.0)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=max(wait, 0.05))


def create_app(
    cfg: NodeConfig,
    classifier: OnnxClassifier | None = None,
    generator: OnnxGenerator | None = None,
    start_sync_loop: bool = True,
) -> FastAPI:
    """Build the node application.

    ``classifier``/``generator`` may be injected (tests). Otherwise they are
    loaded from the config at startup. ``start_sync_loop=False`` disables the
    background loop; rounds can still be forced via ``/admin/sync-now``.
    """
    state = NodeState(cfg)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state.classifier = classifier
        if state.classifier is None and cfg.classifier.model_dir:
            c = cfg.classifier
            state.classifier = OnnxClassifier(
                c.model_dir,
                c.model_file,
                c.max_seq_len,
                intra_op_threads=c.intra_op_threads,
                inter_op_threads=c.inter_op_threads,
                graph_optimization=c.graph_optimization,
            )
            log.info("loaded classifier %s", state.classifier.model_path)
        state.generator = generator
        if state.generator is None and cfg.generator.enabled and cfg.generator.model_dir:
            g = cfg.generator
            state.generator = OnnxGenerator(g.model_dir, g.model_file, g.intra_op_threads)
            log.info("loaded generator from %s", g.model_dir)
        state.node = build_sync_node(cfg, state.links, build_clock(cfg))
        stop = asyncio.Event()
        task = None
        if cfg.sync.enabled and start_sync_loop:
            task = asyncio.create_task(sync_loop(state.node, cfg, stop))
        try:
            yield
        finally:
            stop.set()
            if task is not None:
                await task
            transport = state.node.transport
            if isinstance(transport, HttpTransport):
                transport.close()
            state.node.store.close()

    app = FastAPI(title="edgecluster node", version=__version__, lifespan=lifespan)
    app.state.node_state = state

    def node() -> SyncNode:
        assert state.node is not None
        return state.node

    # ------------------------------------------------------------------ inference
    @app.get("/health")
    def health() -> dict[str, Any]:
        n = node()
        now = n.clock.now()
        return {
            "status": "ok",
            "node_id": cfg.node_id,
            "version": __version__,
            "uptime_s": round(time.time() - state.started, 1),
            "classifier_loaded": state.classifier is not None,
            "classifier_model": state.classifier.model_name if state.classifier else None,
            "generator_loaded": state.generator is not None,
            "uplink_open": n.schedule.is_open(now),
            "records": n.store.count(),
        }

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(state.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/classify")
    async def classify(req: ClassifyRequest) -> dict[str, Any]:
        clf = state.classifier
        if clf is None:
            state.metrics.requests.labels("classify", "503").inc()
            raise HTTPException(503, "no classifier loaded on this node")
        t0 = time.perf_counter()
        feeds = clf.encode([req.text])
        t1 = time.perf_counter()
        logits = await run_in_threadpool(clf.run, feeds)
        t2 = time.perf_counter()
        pred = clf.decode(logits)[0]
        elapsed = time.perf_counter() - t0
        state.metrics.classify_latency.observe(elapsed)
        state.metrics.inference_latency.observe(t2 - t1)
        state.metrics.requests.labels("classify", "200").inc()
        query_id = None
        if req.log and cfg.classifier.log_queries:
            query_id = uuid.uuid4().hex
            record: dict[str, Any] = {
                "label": pred.label,
                "confidence": round(pred.confidence, 4),
                "lang": req.lang,
                "node": cfg.node_id,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": clf.model_name,
                "text_sha256": text_fingerprint(req.text),
            }
            if cfg.classifier.store_query_text:
                record["text"] = req.text
            n = node()
            await run_in_threadpool(n.store.put, QUERY_COLLECTION, query_id, record)
            await run_in_threadpool(n.store.increment, f"queries:{pred.label}", 1)
        return {
            "label": pred.label,
            "confidence": pred.confidence,
            "scores": pred.scores,
            "latency_ms": round(elapsed * 1000, 2),
            "tokenize_ms": round((t1 - t0) * 1000, 2),
            "inference_ms": round((t2 - t1) * 1000, 2),
            "model": clf.model_name,
            "query_id": query_id,
        }

    if cfg.generator.enabled:

        @app.post("/generate")
        async def generate(req: GenerateRequest) -> dict[str, Any]:
            gen = state.generator
            if gen is None:
                raise HTTPException(503, "generator enabled but not loaded")
            res = await run_in_threadpool(
                gen.generate,
                req.prompt,
                cfg.generator.system_prompt,
                req.max_new_tokens or cfg.generator.max_new_tokens,
            )
            state.metrics.generate_latency.observe(res.total_ms / 1000)
            state.metrics.requests.labels("generate", "200").inc()
            return {
                "text": res.text,
                "new_tokens": res.new_tokens,
                "prompt_tokens": res.prompt_tokens,
                "ttft_ms": round(res.ttft_ms, 1),
                "total_ms": round(res.total_ms, 1),
                "ms_per_token": round(res.ms_per_token, 1),
            }

    # ------------------------------------------------------------------ records
    @app.post("/records/{collection}")
    def put_record(collection: str, body: RecordWrite) -> dict[str, Any]:
        if collection not in WRITABLE_COLLECTIONS:
            raise HTTPException(404, f"unknown collection {collection!r}")
        key = body.key or uuid.uuid4().hex
        entry = node().store.put(collection, key, body.value)
        return {"collection": collection, "key": key, "ts": entry.ts.to_list()}

    @app.get("/records/{collection}")
    def list_records(collection: str, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        limit = max(1, min(limit, 1000))
        items = node().store.list(collection, limit, offset)
        return {
            "collection": collection,
            "total": node().store.count(collection),
            "items": [{"key": k, "value": v} for k, v in items],
        }

    @app.get("/records/{collection}/{key}")
    def get_record(collection: str, key: str) -> dict[str, Any]:
        value = node().store.get(collection, key)
        if value is None:
            raise HTTPException(404, "not found")
        return {"collection": collection, "key": key, "value": value}

    @app.delete("/records/{collection}/{key}")
    def delete_record(collection: str, key: str) -> dict[str, Any]:
        if collection not in WRITABLE_COLLECTIONS:
            raise HTTPException(404, f"unknown collection {collection!r}")
        entry = node().store.delete(collection, key)
        return {"collection": collection, "key": key, "deleted": True, "ts": entry.ts.to_list()}

    @app.get("/counters")
    def counters() -> dict[str, int]:
        store = node().store
        return {name: store.counter(name).value for name in store.counter_names()}

    # ------------------------------------------------------------------ sync protocol
    @app.post("/sync/{op}")
    async def sync_op(op: str, request: Request) -> dict[str, Any]:
        sender = request.headers.get(NODE_HEADER)
        if state.links.is_blocked(sender):
            raise HTTPException(503, "link down")
        if op not in {"pull", "push"}:
            raise HTTPException(404, f"unknown op {op!r}")
        payload = await request.json()
        return await run_in_threadpool(node().handle, op, payload)

    @app.get("/sync/digest")
    def digest() -> dict[str, Any]:
        return node().handle("digest", {})

    @app.get("/cluster")
    def cluster() -> dict[str, Any]:
        n = node()
        now = n.clock.now()
        s = n.stats
        return {
            "node_id": cfg.node_id,
            "uplink_open": n.schedule.is_open(now),
            "window_time_left_s": None if not n.schedule.is_open(now) else n.schedule.time_left(now),
            "next_window_in_s": n.schedule.time_until_open(now),
            "peers": {
                p: {"alive": n.membership.is_alive(p), "heartbeat": n.membership.state[p].heartbeat}
                for p in n.membership.peers
            },
            "blocked_links": sorted(state.links.blocked()),
            "stats": {
                "rounds": s.rounds,
                "exchanges_ok": s.exchanges_ok,
                "exchanges_failed": s.exchanges_failed,
                "entries_pulled": s.entries_pulled,
                "entries_pushed": s.entries_pushed,
            },
        }

    # ------------------------------------------------------------------ admin (demo only)
    if cfg.admin_enabled:

        @app.post("/admin/links")
        def set_links(body: LinksRequest) -> dict[str, Any]:
            state.links.set_blocked(set(body.blocked))
            return {"blocked": sorted(state.links.blocked())}

        @app.post("/admin/sync-now")
        async def sync_now() -> dict[str, Any]:
            results = await asyncio.to_thread(node().run_round, True)
            return {"exchanges": [r.__dict__ for r in results]}

    return app
