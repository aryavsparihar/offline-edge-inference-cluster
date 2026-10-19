"""Node and cluster configuration (YAML file plus ``EDGE_*`` environment overrides)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from edgecluster.cluster.scheduler import UplinkSchedule, UplinkWindow, parse_duration, parse_offset


class PeerConfig(BaseModel):
    """One statically configured cluster member."""

    id: str
    url: str


class WindowConfig(BaseModel):
    """An uplink window, e.g. ``{start: "06:00", duration: "2h"}``."""

    start: str | float = "06:00"
    duration: str | float = "2h"


class UplinkConfig(BaseModel):
    """When inter-node sync is possible."""

    always_open: bool = False
    period: str | float = "1d"
    utc_offset: str | float = "5.5h"  # IST
    windows: list[WindowConfig] = Field(default_factory=lambda: [WindowConfig()])
    time_scale: float = 1.0  # >1 compresses time for demos (1440 = a day per minute)

    def schedule(self) -> UplinkSchedule:
        """Build the runtime schedule."""
        return UplinkSchedule(
            [UplinkWindow(parse_offset(w.start), parse_duration(w.duration)) for w in self.windows],
            period_s=parse_duration(self.period),
            utc_offset_s=parse_duration(self.utc_offset),
            always_open=self.always_open,
        )


class SyncConfig(BaseModel):
    """Anti-entropy tuning."""

    enabled: bool = True
    interval: str | float = "30s"  # time between rounds inside a window
    fanout: int = 2
    page_size: int = 500
    max_pages: int = 20
    timeout: str | float = "5s"
    suspect_after: str | float = "5m"
    max_drift: str | float | None = None  # reject peers whose HLC is this far ahead


class ClassifierConfig(BaseModel):
    """ONNX classifier served at ``/classify``."""

    model_dir: str | None = None  # contains model.onnx, tokenizer.json, labels.json
    model_file: str = "model.onnx"
    max_seq_len: int = 128
    intra_op_threads: int = 4  # Raspberry Pi 4 has 4 Cortex-A72 cores
    inter_op_threads: int = 1
    graph_optimization: str = "all"  # disable | basic | extended | all
    log_queries: bool = True  # replicate a privacy-preserving query log record
    store_query_text: bool = False  # raw text is NOT replicated unless enabled


class GeneratorConfig(BaseModel):
    """Optional small generative model served at ``/generate``."""

    enabled: bool = False
    model_dir: str | None = None
    model_file: str = "onnx/model_quantized.onnx"
    max_new_tokens: int = 32
    intra_op_threads: int = 4
    system_prompt: str = "You are a helpful assistant for welfare scheme information. Answer briefly."


class NodeConfig(BaseModel):
    """Everything one node needs to run."""

    node_id: str
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: str = "./runs/node-data"
    peers: list[PeerConfig] = Field(default_factory=list)
    uplink: UplinkConfig = Field(default_factory=UplinkConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    admin_enabled: bool = False  # exposes /admin/links for partition emulation

    @field_validator("node_id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        if not v or ":" in v:
            raise ValueError("node_id must be non-empty and contain no ':'")
        return v

    @property
    def db_path(self) -> Path:
        """SQLite file for this node's replica."""
        return Path(self.data_dir) / f"{self.node_id}.db"

    def peer_urls(self) -> dict[str, str]:
        """Other nodes' base URLs keyed by id."""
        return {p.id: p.url for p in self.peers if p.id != self.node_id}


ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "EDGE_NODE_ID": ("node_id",),
    "EDGE_PORT": ("port",),
    "EDGE_DATA_DIR": ("data_dir",),
    "EDGE_MODEL_DIR": ("classifier", "model_dir"),
    "EDGE_MODEL_FILE": ("classifier", "model_file"),
    "EDGE_MAX_SEQ_LEN": ("classifier", "max_seq_len"),
    "EDGE_INTRA_OP_THREADS": ("classifier", "intra_op_threads"),
    "EDGE_GENERATOR_ENABLED": ("generator", "enabled"),
    "EDGE_GENERATOR_DIR": ("generator", "model_dir"),
    "EDGE_ADMIN": ("admin_enabled",),
    "EDGE_UPLINK_ALWAYS_OPEN": ("uplink", "always_open"),
    "EDGE_TIME_SCALE": ("uplink", "time_scale"),
}


def _set_path(cfg: dict[str, Any], path: tuple[str, ...], value: str) -> None:
    node = cfg
    for key in path[:-1]:
        node = node.setdefault(key, {})
    parsed: Any = value
    if value.lower() in {"true", "false"}:
        parsed = value.lower() == "true"
    node[path[-1]] = parsed


def load_config(path: str | Path | None = None, **overrides: Any) -> NodeConfig:
    """Load a node config from YAML, then apply ``EDGE_*`` env vars, then ``overrides``.

    A YAML file may contain a ``cluster`` section shared by all nodes (peers,
    uplink, sync) and per-node keys at top level. This lets docker-compose use
    one file for every container.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    cluster = raw.pop("cluster", {}) or {}
    merged: dict[str, Any] = {**cluster, **raw}
    for env, keypath in ENV_OVERRIDES.items():
        if env in os.environ:
            _set_path(merged, keypath, os.environ[env])
    merged.update(overrides)
    return NodeConfig.model_validate(merged)
