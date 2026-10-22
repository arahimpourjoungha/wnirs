"""What it costs to run the channel-quality model on a device, and what int8 loses.

    python scripts/edge_report.py runs/windows_default.npz \
        --encoder runs/results/pretrained_default/pretrained_s0_e15.pt --out runs/edge

Fine-tunes the quality head from a pretrained encoder on the training
sessions, then on the held-out sessions measures:

  * accuracy (AUC) of the PyTorch model, the streaming float kernel, the
    streaming int8 kernel, and the ONNX fp32 / int8 models;
  * parameters, multiply-accumulates per sample, weight and state memory;
  * latency: one ONNX window, and one streaming step (per-sample cost).

Writes edge_report.json and edge_report.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from wnirs.ml.data import WindowSet, session_split
from wnirs.ml.export import export_onnx, latency_ms, onnx_predict, quantize_onnx
from wnirs.ml.models import WNIRSNet, count_macs, count_params
from wnirs.ml.streaming import StreamingEncoder, calibrate_activation_scales, compare_to_torch
from wnirs.ml.train import (TrainConfig, finetune_quality, inner_val_split, quality_scores,
                            seed_everything)


def auc(scores: np.ndarray, ws: WindowSet) -> float:
    lab = ws.mask & ~np.isnan(ws.quality)
    return float(roc_auc_score(ws.quality[lab] > 0.5, scores[lab]))


def streaming_scores(stream: StreamingEncoder, ws: WindowSet) -> tuple[np.ndarray, float]:
    """Quality logits at the end of each window, fed one sample at a time."""
    out = np.zeros(ws.mask.shape, np.float32)
    t_step = []
    for i in range(len(ws)):
        stream.mask = ws.mask[i]
        stream.reset()
        for t in range(ws.x.shape[2]):
            t0 = time.perf_counter()
            q = stream.step(ws.x[i, :, t])
            t_step.append(time.perf_counter() - t0)
        out[i] = q
    return out, 1e3 * float(np.median(t_step))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("windows", type=Path)
    ap.add_argument("--encoder", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/edge"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--stream-windows", type=int, default=150,
                    help="held-out windows scored by the (slow, pure-Python) streaming kernels")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.backends.nnpack.set_flags(False)

    seed_everything(0)
    ws = WindowSet.load(a.windows)
    tr, te = session_split(ws.meta)
    tr_in, va = inner_val_split(ws.meta, tr)
    n_sig, n_t = ws.x.shape[1:]
    model = WNIRSNet(n_sig)
    model.load_state_dict(torch.load(a.encoder), strict=False)
    finetune_quality(model, ws.subset(tr_in), ws.subset(va), TrainConfig(epochs=a.epochs, patience=5, batch_size=64))
    model.eval()
    test = ws.subset(te)
    rng = np.random.default_rng(0)
    sub = test.subset(np.sort(rng.choice(len(test), min(a.stream_windows, len(test)), replace=False)))

    r: dict = {"n_signals": int(n_sig), "window_samples": int(n_t), "fs_hz": ws.fs,
               "params": count_params(model),
               "macs_per_sample": count_macs(model, n_sig, 1),
               "auc_torch_all_test": auc(quality_scores(model, test), test),
               "auc_torch_stream_subset": auc(quality_scores(model, sub), sub)}

    s32 = StreamingEncoder(model, sub.mask[0])
    r["stream_float_parity"] = compare_to_torch(model, s32, sub.x[0])
    sc, r["stream_float_step_ms"] = streaming_scores(s32, sub)
    r["auc_stream_float"] = auc(sc, sub)
    scales = calibrate_activation_scales(model, ws.x[tr[:256]], ws.mask[tr[:256]])
    s8 = StreamingEncoder(model, sub.mask[0], int8=True, act_scales=scales)
    r["stream_int8_parity"] = compare_to_torch(model, s8, sub.x[0])
    sc8, r["stream_int8_step_ms"] = streaming_scores(s8, sub)
    r["auc_stream_int8"] = auc(sc8, sub)
    r["memory_float32_bytes"] = s32.memory_bytes()
    r["memory_int8_bytes"] = s8.memory_bytes()

    p32 = export_onnx(model, n_sig, n_t, a.out / "quality_fp32.onnx")
    p8 = quantize_onnx(p32, ws.x[tr[:128]], ws.mask[tr[:128]], a.out / "quality_int8.onnx")
    r["auc_onnx_fp32"] = auc(onnx_predict(p32, test.x, test.mask)[1], test)
    r["auc_onnx_int8"] = auc(onnx_predict(p8, test.x, test.mask)[1], test)
    r["onnx_fp32_kb"] = p32.stat().st_size / 1024
    r["onnx_int8_kb"] = p8.stat().st_size / 1024
    r["onnx_fp32_window_ms"] = latency_ms(p32, n_sig, n_t)
    r["onnx_int8_window_ms"] = latency_ms(p8, n_sig, n_t)

    (a.out / "edge_report.json").write_text(json.dumps(r, indent=2))
    m32, m8 = r["memory_float32_bytes"], r["memory_int8_bytes"]
    md = f"""# Edge report: channel-quality model

Held-out sessions = each participant's latest 25 %. Streaming kernels scored on
{len(sub)} held-out windows (pure-Python reference implementation, so per-step
times are an upper bound, not MCU timings).

| | value |
|---|---|
| parameters | {r['params']:,} |
| MACs per input sample (all {n_sig} signals) | {r['macs_per_sample']:,} |
| MACs per second at {ws.fs:g} Hz | {r['macs_per_sample'] * ws.fs:,.0f} |
| weights, float32 / int8 | {m32['weights'] / 1024:.1f} KB / {m8['weights'] / 1024:.1f} KB |
| streaming state, float32 / int8 | {m32['state'] / 1024:.1f} KB / {m8['state'] / 1024:.1f} KB |
| ONNX file, fp32 / int8 | {r['onnx_fp32_kb']:.1f} KB / {r['onnx_int8_kb']:.1f} KB |
| ONNX latency per 30 s window, fp32 / int8 (1 CPU thread) | {r['onnx_fp32_window_ms']:.2f} ms / {r['onnx_int8_window_ms']:.2f} ms |

| model | AUC (QC pass vs fail) |
|---|---|
| PyTorch, all held-out windows | {r['auc_torch_all_test']:.3f} |
| ONNX fp32, all held-out windows | {r['auc_onnx_fp32']:.3f} |
| ONNX int8, all held-out windows | {r['auc_onnx_int8']:.3f} |
| PyTorch, streaming subset | {r['auc_torch_stream_subset']:.3f} |
| streaming float kernel, same subset | {r['auc_stream_float']:.3f} |
| streaming int8 kernel, same subset | {r['auc_stream_int8']:.3f} |

Streaming float vs PyTorch, one window: max |Δ quality logit| = {r['stream_float_parity']['quality_max_abs']:.2e}.
Streaming int8 vs PyTorch: relative embedding error = {r['stream_int8_parity']['embedding_rel']:.3f}.
"""
    (a.out / "edge_report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
