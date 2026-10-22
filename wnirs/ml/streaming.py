"""Sample-by-sample inference with folded, optionally int8, weights -- what runs on the device.

The PyTorch model processes a 30 s window at once. A wearable instead gets one
sample every 200 ms and has kilobytes of RAM, so the deployed form is
different in three ways, each verified here against the PyTorch model:

1. **BatchNorm folded into the convolutions.** At inference BN is an affine
   map per channel, so it becomes part of the preceding conv's weight and bias.
2. **Ring buffers instead of padding.** A causal dilated conv at step t needs
   inputs t, t-r, ..., t-(k-1)r. Each layer keeps only those (k-1)*r+1 past
   values, so state memory is fixed and tiny, independent of recording length.
3. **int8 weights and activations** (optional). Symmetric per-output-channel
   weight scales, per-tensor activation scales calibrated on real windows;
   accumulation in int32 as on Cortex-M / micro-NPU kernels.

``StreamingEncoder.step`` takes one sample of every signal and returns the
per-signal quality logits for the samples seen so far; ``embedding`` gives the
pooled window embedding. With float weights it reproduces the PyTorch model to
floating-point precision; with int8 the deviation is reported by
``compare_to_torch``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .models import CausalConv1d, SepBlock, WNIRSNet


def _fold_bn(w: np.ndarray, b: np.ndarray | None, bn: torch.nn.BatchNorm1d) -> tuple[np.ndarray, np.ndarray]:
    g = bn.weight.detach().numpy() / np.sqrt(bn.running_var.numpy() + bn.eps)
    w = w * g.reshape(-1, *([1] * (w.ndim - 1)))
    b0 = np.zeros(w.shape[0]) if b is None else b
    return w, (b0 - bn.running_mean.numpy()) * g + bn.bias.detach().numpy()


@dataclass
class _Layer:
    dw_w: np.ndarray   # (d, k)   depthwise taps, oldest first
    dw_b: np.ndarray   # (d,)
    pw_w: np.ndarray   # (d, d)
    pw_b: np.ndarray   # (d,)
    k: int
    dilation: int


def _quant_sym(a: np.ndarray, axis: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8: per-channel along ``axis`` (weights) or per-tensor (axis=None)."""
    amax = np.abs(a).max(axis=tuple(i for i in range(a.ndim) if i != axis), keepdims=True) if axis is not None \
        else np.abs(a).max()
    scale = np.where(amax > 0, amax / 127.0, 1.0)
    return np.clip(np.round(a / scale), -127, 127).astype(np.int8), scale


class StreamingEncoder:
    """Stateful one-sample-at-a-time encoder built from a trained ``WNIRSNet``."""

    def __init__(self, model: WNIRSNet, mask: np.ndarray, int8: bool = False,
                 act_scales: list[float] | None = None):
        model = model.eval()
        enc = model.encoder
        self.mask = mask.astype(bool)
        self.n_signals = len(mask)
        self.d = enc.d
        self.inp_w = enc.inp.weight.detach().numpy()[:, 0, 0]          # (d,)
        self.inp_b = enc.inp.bias.detach().numpy()
        self.sig_emb = enc.signal_emb.weight.detach().numpy()[: self.n_signals]   # (S, d)
        self.layers: list[_Layer] = []
        for blk in enc.blocks:
            assert isinstance(blk, SepBlock) and isinstance(blk.dw, CausalConv1d)
            dw_w, dw_b = _fold_bn(blk.dw.weight.detach().numpy()[:, 0, :], None, blk.bn1)
            pw_w, pw_b = _fold_bn(blk.pw.weight.detach().numpy()[:, :, 0], None, blk.bn2)
            self.layers.append(_Layer(dw_w, dw_b, pw_w, pw_b, blk.dw.kernel_size[0], blk.dw.dilation[0]))
        self.q_w = model.quality.weight.detach().numpy()[0]
        self.q_b = float(model.quality.bias.detach().numpy()[0])
        self.score_w = model.pool.score.weight.detach().numpy()[0]
        self.score_b = float(model.pool.score.bias.detach().numpy()[0])

        self.int8 = int8
        if int8:
            if act_scales is None:
                raise ValueError("int8 needs activation scales; use calibrate_activation_scales()")
            self.act_scales = act_scales
            for L in self.layers:
                L.dw_w = self._fake_quant_w(L.dw_w)
                L.pw_w = self._fake_quant_w(L.pw_w)
        self.reset()

    @staticmethod
    def _fake_quant_w(w: np.ndarray) -> np.ndarray:
        q, s = _quant_sym(w, axis=0)
        return q.astype(np.float64) * s

    def _fake_quant_a(self, a: np.ndarray, i: int) -> np.ndarray:
        s = self.act_scales[i]
        return np.clip(np.round(a / s), -127, 127) * s

    def reset(self) -> None:
        self.buffers = [np.zeros((self.n_signals, self.d, (L.k - 1) * L.dilation + 1)) for L in self.layers]
        self.sum_h = np.zeros((self.n_signals, self.d))
        self.t = 0

    def step(self, sample: np.ndarray) -> np.ndarray:
        """Consume one sample (n_signals,) and return per-signal quality logits so far."""
        x = np.where(self.mask, sample, 0.0)
        h = x[:, None] * self.inp_w + self.inp_b + self.sig_emb          # (S, d)
        for i, (L, buf) in enumerate(zip(self.layers, self.buffers)):
            if self.int8:
                h = self._fake_quant_a(h, i)
            buf[:, :, :-1] = buf[:, :, 1:]
            buf[:, :, -1] = h
            taps = buf[:, :, :: L.dilation]                              # (S, d, k), oldest first
            u = np.maximum((taps * L.dw_w).sum(-1) + L.dw_b, 0.0)
            h = np.maximum(h + u @ L.pw_w.T + L.pw_b, 0.0)
        self.sum_h += h
        self.t += 1
        return self.quality_logits()

    def quality_logits(self) -> np.ndarray:
        return (self.sum_h / max(self.t, 1)) @ self.q_w + self.q_b

    def embedding(self) -> np.ndarray:
        z = self.sum_h / max(self.t, 1)
        a = np.where(self.mask, z @ self.score_w + self.score_b, -np.inf)
        a = np.exp(a - a.max())
        a /= a.sum()
        return (a[:, None] * z).sum(0)

    def memory_bytes(self) -> dict[str, int]:
        """Footprint on device: int8 weights and ring buffers when quantised, else float32.

        Biases and the running-mean accumulator stay 32-bit in both cases, as
        int8 kernels accumulate in int32.
        """
        wb = 1 if self.int8 else 4
        n_w = self.inp_w.size + self.sig_emb.size + sum(L.dw_w.size + L.pw_w.size for L in self.layers)
        n_b = self.inp_b.size + sum(L.dw_b.size + L.pw_b.size for L in self.layers)
        n_buf = sum(b.size for b in self.buffers)
        return {"weights": n_w * wb + n_b * 4, "state": n_buf * wb + self.sum_h.size * 4}


@torch.no_grad()
def calibrate_activation_scales(model: WNIRSNet, x: np.ndarray, mask: np.ndarray) -> list[float]:
    """Per-layer input activation scale (max |a| / 127) over a calibration set."""
    model.eval()
    enc = model.encoder
    xt, mt = torch.from_numpy(x), torch.from_numpy(mask)
    b, s, t = xt.shape
    h = enc.inp((xt * mt[..., None]).reshape(b * s, 1, t)).reshape(b, s, enc.d, t)
    h = (h + enc.signal_emb.weight[:s, :, None]).reshape(b * s, enc.d, t)
    scales = []
    for blk in enc.blocks:
        scales.append(float(h.abs().max()) / 127.0)
        h = blk(h)
    return scales


@torch.no_grad()
def compare_to_torch(model: WNIRSNet, stream: StreamingEncoder, x: np.ndarray) -> dict[str, float]:
    """Run one window through both; return max abs differences of quality logits and embedding."""
    model.eval()
    out = model(torch.from_numpy(x[None]), torch.from_numpy(stream.mask[None]))
    stream.reset()
    for t in range(x.shape[1]):
        q = stream.step(x[:, t])
    ref_q = out["quality"][0].numpy()
    ref_z = out["embedding"][0].numpy()
    m = stream.mask
    return {
        "quality_max_abs": float(np.abs(q[m] - ref_q[m]).max()),
        "embedding_max_abs": float(np.abs(stream.embedding() - ref_z).max()),
        "embedding_rel": float(np.linalg.norm(stream.embedding() - ref_z) / (np.linalg.norm(ref_z) + 1e-12)),
    }
