"""A small causal temporal-convolution encoder for wearable fNIRS, sized for an MCU.

Design choices, and why:

* **Channel-independent temporal encoder.** One shared stack of 1-D
  convolutions runs over every signal (HbO/HbR x channel) separately, plus a
  learned embedding saying which signal it is. Weights are shared across 62
  signals, so the parameter count stays in the low thousands, and a lost
  channel just removes one stream instead of corrupting a mixed input.
  Cross-signal information enters only at pooling, via masked attention.
* **Causal, dilated, depthwise-separable convolutions.** Causal so that the
  same weights run sample-by-sample on device (see ``wnirs.ml.streaming``);
  dilations 1..16 with kernel 5 give a ~25 s receptive field at 5 Hz, about
  one haemodynamic response plus its undershoot; depthwise-separable to cut
  multiply-accumulates roughly by the kernel size.
* **ReLU and BatchNorm only.** Both fold cleanly into int8 kernels (BN folds
  into the preceding conv; ReLU is a clamp), so the quantised model is the
  trained model, not an approximation of it.

Self-supervised pretraining is masked *causal* reconstruction: random spans
of random signals are hidden and the model must reproduce them from the past
of the same signal and the other signals' context. Being causal, this is
really short-horizon forecasting -- the same capability a streaming model uses.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """Conv1d that pads on the left only, so output t never sees input > t."""

    def __init__(self, c_in: int, c_out: int, k: int, dilation: int = 1, groups: int = 1):
        super().__init__(c_in, c_out, k, dilation=dilation, groups=groups, bias=False)
        self.left_pad = (k - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = F.pad(x, (self.left_pad, 0))
        if self.groups != self.in_channels or self.groups != self.out_channels:
            return super().forward(xp)
        # Depthwise case as a sum of k shifted, per-channel scaled copies.
        # PyTorch's CPU backward for grouped convs loops over groups x batch,
        # which with ~2000 sequences per batch is two orders of magnitude
        # slower than these k elementwise ops. Same weights, same result.
        k, r, t = self.kernel_size[0], self.dilation[0], x.shape[-1]
        w = self.weight[:, 0, :]                                      # (C, k), oldest tap first
        y = sum(w[:, j, None] * xp[..., j * r: j * r + t] for j in range(k))
        return y if self.bias is None else y + self.bias[:, None]


class Pointwise(nn.Conv1d):
    """1x1 conv computed as a matmul; the CPU conv kernels are slow for this shape."""

    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super().__init__(c_in, c_out, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("oc,nct->not", self.weight[:, :, 0], x)
        return y if self.bias is None else y + self.bias[:, None]


class SepBlock(nn.Module):
    """Depthwise causal conv -> BN -> ReLU -> pointwise conv -> BN, with a residual."""

    def __init__(self, d: int, k: int, dilation: int):
        super().__init__()
        self.dw = CausalConv1d(d, d, k, dilation, groups=d)
        self.bn1 = nn.BatchNorm1d(d)
        self.pw = Pointwise(d, d, bias=False)
        self.bn2 = nn.BatchNorm1d(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.dw(x)))
        return F.relu(x + self.bn2(self.pw(h)))


class SignalEncoder(nn.Module):
    """Shared temporal encoder applied to each signal independently.

    Input  (B, S, T) windows and (B, S) usable-mask.
    Output (B, S, d, T) per-signal feature sequences.
    """

    def __init__(self, n_signals: int, d: int = 16, k: int = 5,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16)):
        super().__init__()
        self.inp = Pointwise(1, d)
        self.signal_emb = nn.Embedding(n_signals, d)
        self.blocks = nn.Sequential(*[SepBlock(d, k, r) for r in dilations])
        self.d = d
        self.receptive_field = 1 + (k - 1) * sum(dilations)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, s, t = x.shape
        h = self.inp((x * mask[..., None]).reshape(b * s, 1, t)).reshape(b, s, self.d, t)
        # broadcast, not repeat(b): keeps the batch dimension dynamic in exported graphs
        h = h + self.signal_emb.weight[:s, :, None]
        return self.blocks(h.reshape(b * s, self.d, t)).reshape(b, s, self.d, t)


class MaskedPool(nn.Module):
    """Attention pooling over signals that ignores masked ones, then mean over time."""

    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = h.mean(-1)                                       # (B, S, d)
        a = self.score(z).squeeze(-1).masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, dim=-1).nan_to_num(0.0)         # all-masked window -> zeros
        return (a[..., None] * z).sum(1)                     # (B, d)


class WNIRSNet(nn.Module):
    """Encoder with three heads.

    recon   per-signal reconstruction, for self-supervised pretraining
    quality per-signal quality score (logit), the on-device "is this channel usable?" check
    cls     window-level classifier (task state, or subject for fingerprinting)
    """

    def __init__(self, n_signals: int, n_classes: int = 4, d: int = 16, **enc_kw):
        super().__init__()
        self.encoder = SignalEncoder(n_signals, d, **enc_kw)
        self.recon = Pointwise(d, 1)
        self.quality = nn.Linear(d, 1)
        self.pool = MaskedPool(d)
        self.cls = nn.Linear(d, n_classes)

    def embed(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encoder(x, mask), mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x, mask)
        b, s, d, t = h.shape
        z = self.pool(h, mask)
        return {
            "recon": self.recon(h.reshape(b * s, d, t)).reshape(b, s, t),
            "quality": self.quality(h.mean(-1)).squeeze(-1),
            "logits": self.cls(z),
            "embedding": z,
        }

    def reset_head(self, n_classes: int) -> None:
        """New classifier for a new downstream task, keeping the pretrained encoder."""
        self.cls = nn.Linear(self.encoder.d, n_classes)


def masked_recon_loss(x: torch.Tensor, mask: torch.Tensor, model: WNIRSNet,
                      span: int = 10, ratio: float = 0.3) -> torch.Tensor:
    """Hide random spans of usable signals; score reconstruction on the hidden samples only.

    Hidden samples are zeroed in the input (the same value a dropped channel
    takes), so the model learns one consistent meaning for "no data here".
    """
    b, s, t = x.shape
    n_spans = max(1, int(ratio * t / span))
    hide = torch.zeros_like(x, dtype=torch.bool)
    starts = torch.randint(0, t - span + 1, (b, s, n_spans), device=x.device)
    pos = starts[..., None] + torch.arange(span, device=x.device)
    hide.scatter_(-1, pos.reshape(b, s, -1), True)
    hide &= mask[..., None]
    out = model(x.masked_fill(hide, 0.0), mask)["recon"]
    return ((out - x) ** 2)[hide].mean()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: WNIRSNet, n_signals: int, t: int) -> int:
    """Multiply-accumulates for one window through the encoder (convs dominate)."""
    macs = 0
    for m in model.encoder.modules():
        if isinstance(m, nn.Conv1d):
            macs += m.in_channels // m.groups * m.out_channels * m.kernel_size[0] * t
    return macs * n_signals
