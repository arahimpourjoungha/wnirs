"""ONNX export and int8 post-training quantisation via ONNX Runtime.

The streaming NumPy kernel (``wnirs.ml.streaming``) is the reference for an
MCU; ONNX is the route to phones, micro-NPUs and anything with a vendor
toolchain that ingests ONNX. Both are checked against the PyTorch model.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .models import WNIRSNet


class _Wrapped(torch.nn.Module):
    """Expose (logits, quality) only; ONNX has no use for the training heads."""

    def __init__(self, model: WNIRSNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        out = self.model(x, mask.bool())
        return out["logits"], out["quality"]


def export_onnx(model: WNIRSNet, n_signals: int, n_samples: int, path: str | Path) -> Path:
    path = Path(path)
    # The wrapper must be in eval mode too: export restores the *wrapper's* mode
    # afterwards, which would otherwise switch the model back to training.
    wrapped = _Wrapped(model).eval()
    x = torch.zeros(1, n_signals, n_samples)
    m = torch.ones(1, n_signals)
    torch.onnx.export(wrapped, (x, m), str(path), input_names=["x", "mask"],
                      output_names=["logits", "quality"], opset_version=17,
                      dynamic_axes={"x": {0: "batch"}, "mask": {0: "batch"},
                                    "logits": {0: "batch"}, "quality": {0: "batch"}},
                      dynamo=False)
    return path


def quantize_onnx(fp32_path: str | Path, calib_x: np.ndarray, calib_mask: np.ndarray,
                  out_path: str | Path) -> Path:
    """Static int8 quantisation (QDQ format), calibrated on real windows."""
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter({"x": calib_x[i:i + 1], "mask": calib_mask[i:i + 1].astype(np.float32)}
                           for i in range(len(calib_x)))

        def get_next(self):
            return next(self.it, None)

    pre = Path(out_path).with_suffix(".pre.onnx")
    quant_pre_process(str(fp32_path), str(pre))
    quantize_static(str(pre), str(out_path), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True)
    pre.unlink(missing_ok=True)
    return Path(out_path)


def onnx_predict(path: str | Path, x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return tuple(sess.run(None, {"x": x.astype(np.float32), "mask": mask.astype(np.float32)}))


def latency_ms(path: str | Path, n_signals: int, n_samples: int, repeats: int = 200) -> float:
    """Median single-window latency on one CPU thread."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    feed = {"x": np.random.randn(1, n_signals, n_samples).astype(np.float32),
            "mask": np.ones((1, n_signals), np.float32)}
    for _ in range(10):
        sess.run(None, feed)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sess.run(None, feed)
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))
