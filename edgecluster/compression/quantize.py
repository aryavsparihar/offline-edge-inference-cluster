"""INT8 dynamic quantization with ONNX Runtime.

Dynamic quantization stores weights as 8-bit integers and quantizes
activations on the fly, so no calibration data is needed. With the default
op set, ``MatMul`` weights (attention/FFN projections) and ``Gather`` (the
embedding table, which is 69% of XLM-R's parameters) are quantized.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

DEFAULT_OPS = ("MatMul", "Gather")
log = logging.getLogger(__name__)


@contextmanager
def _working_dir(path: str | Path) -> Iterator[None]:
    """Temporarily chdir. ONNX tooling writes ``<uuid>.data`` scratch files into the CWD."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def preprocess_model(src: Path, dst: Path) -> None:
    """ORT's quantization pre-processing (shape inference + optimization).

    Symbolic shape inference sometimes can't resolve every dimension in
    exported graphs (``Incomplete symbolic shape inference``). In that case
    fall back to standard ONNX shape inference, which is sufficient for
    dynamic quantization.
    """
    try:
        quant_pre_process(str(src), str(dst), skip_symbolic_shape=False)
    except Exception as exc:  # ORT raises a bare Exception here
        log.warning("symbolic shape inference failed (%s); using ONNX shape inference", exc)
        quant_pre_process(str(src), str(dst), skip_symbolic_shape=True)


def quantize_int8(
    src: str | Path,
    dst: str | Path,
    per_channel: bool = False,
    reduce_range: bool = False,
    op_types: Sequence[str] = DEFAULT_OPS,
    preprocess: bool = True,
) -> Path:
    """Quantize ``src`` to ``dst``.

    Args:
        per_channel: One scale per output channel instead of per tensor
            (usually more accurate, slightly larger and slower).
        reduce_range: 7-bit weights. Only needed on old x86 CPUs without VNNI.
            Not needed on ARM.
        op_types: Operator types to quantize.
        preprocess: Run ORT's recommended pre-processing (symbolic shape
            inference + graph optimization) first.
    """
    src, dst = Path(src).resolve(), Path(dst).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, _working_dir(tmp):
        model_in = src
        if preprocess:
            model_in = Path(tmp) / "preprocessed.onnx"
            preprocess_model(src, model_in)
        quantize_dynamic(
            str(model_in),
            str(dst),
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            reduce_range=reduce_range,
            op_types_to_quantize=list(op_types),
            extra_options={"MatMulConstBOnly": True},
        )
    return dst
