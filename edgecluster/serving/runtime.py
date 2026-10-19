"""ONNX Runtime session construction tuned for small ARM CPUs."""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort

GRAPH_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def make_session(
    model_path: str | Path,
    intra_op_threads: int = 4,
    inter_op_threads: int = 1,
    graph_optimization: str = "all",
    allow_spinning: bool = True,
) -> ort.InferenceSession:
    """Create a CPU inference session.

    Defaults target a Raspberry Pi 4 (4x Cortex-A72, no SMT): one intra-op
    thread per physical core, sequential execution (transformer graphs have
    little inter-op parallelism, so extra pools just contend). Thread spinning
    stays on by default because it lowers latency under continuous load. Turn
    it off to save power on a mostly idle node.
    """
    if graph_optimization not in GRAPH_LEVELS:
        raise ValueError(f"graph_optimization must be one of {sorted(GRAPH_LEVELS)}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_op_threads
    opts.inter_op_num_threads = inter_op_threads
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = GRAPH_LEVELS[graph_optimization]
    opts.add_session_config_entry("session.intra_op.allow_spinning", "1" if allow_spinning else "0")
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
