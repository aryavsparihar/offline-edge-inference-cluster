"""Prometheus metrics for a node (one registry per app instance)."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import psutil
from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from edgecluster.cluster.node import SyncNode

# Pi 4 latencies span ~50 ms (short int8 input) to several seconds (generation).
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)


class _NodeCollector(Collector):
    """Exports sync statistics and process memory at scrape time."""

    def __init__(self, node: Callable[[], SyncNode | None]) -> None:
        self._node = node
        self._proc = psutil.Process()

    def collect(self) -> Iterator[Metric]:
        rss = GaugeMetricFamily("edge_process_rss_bytes", "Resident set size of this node process")
        rss.add_metric([], float(self._proc.memory_info().rss))
        yield rss
        node = self._node()
        if node is None:
            return
        s = node.stats
        for name, value, doc in (
            ("edge_sync_rounds", s.rounds, "Sync rounds executed inside uplink windows"),
            ("edge_sync_rounds_skipped", s.rounds_skipped_closed, "Rounds skipped (window closed)"),
            ("edge_sync_exchanges_ok", s.exchanges_ok, "Successful peer exchanges"),
            ("edge_sync_exchanges_failed", s.exchanges_failed, "Failed peer exchanges"),
            ("edge_sync_entries_pulled", s.entries_pulled, "Changes pulled from peers"),
            ("edge_sync_entries_pushed", s.entries_pushed, "Changes pushed to peers"),
        ):
            g = GaugeMetricFamily(name, doc)
            g.add_metric([], float(value))
            yield g
        recs = GaugeMetricFamily("edge_store_records", "Records in the local replica", labels=["collection"])
        for coll in ("queries", "cases"):
            recs.add_metric([coll], float(node.store.count(coll)))
        yield recs
        up = GaugeMetricFamily("edge_uplink_open", "1 if an uplink window is open")
        up.add_metric([], float(node.schedule.is_open(node.clock.now())))
        yield up
        alive = GaugeMetricFamily("edge_peers_alive", "Peers with recent heartbeat progress")
        alive.add_metric([], float(len(node.membership.alive_peers())))
        yield alive


class NodeMetrics:
    """Metric handles used by the HTTP layer."""

    def __init__(self, node: Callable[[], SyncNode | None]) -> None:
        self.registry = CollectorRegistry()
        self.classify_latency = Histogram(
            "edge_classify_latency_seconds",
            "End-to-end /classify latency (tokenize + inference)",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.inference_latency = Histogram(
            "edge_inference_latency_seconds",
            "ONNX Runtime session.run latency for classification",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.generate_latency = Histogram(
            "edge_generate_latency_seconds",
            "End-to-end /generate latency",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.requests = Counter(
            "edge_requests", "Requests by endpoint and status", ["endpoint", "status"], registry=self.registry
        )
        self.registry.register(_NodeCollector(node))
