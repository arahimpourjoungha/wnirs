"""Hand-crafted DSP features: the classical baseline every learned model must beat.

Two families, computed per window:

* **Per-signal spectral/temporal features** -- band powers in the bands that
  matter physiologically for fNIRS (task/neurovascular, Mayer waves,
  respiration, cardiac), spectral entropy, kurtosis and the largest
  first difference (a cheap motion-artefact detector). Cardiac power in
  particular is the basis of the scalp-coupling index: a channel that
  couples to skin shows the pulse; one that does not is noise.
* **Connectivity** -- the upper triangle of the Fisher-z correlation matrix
  between usable signals, the feature the precision-imaging literature
  uses for "fingerprinting" individuals.

All functions are vectorised over windows and signals; nothing loops in Python
over samples.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis

BANDS = {
    "task": (0.01, 0.08),
    "mayer": (0.08, 0.15),
    "resp": (0.15, 0.5),
    "cardiac": (0.8, 2.0),
}
SPECTRAL_NAMES = [f"bp_{b}" for b in BANDS] + ["spec_entropy", "kurtosis", "max_diff", "acf1"]


def spectral_features(x: np.ndarray, fs: float) -> np.ndarray:
    """Per-signal features.

    x : (n_windows, n_signals, n_samples)
    returns (n_windows, n_signals, len(SPECTRAL_NAMES))

    Band powers are relative (fraction of total power), so they are
    insensitive to the per-recording z-scoring upstream.
    """
    nper = min(x.shape[-1], int(fs * 20))
    f, p = welch(x, fs=fs, nperseg=nper, axis=-1)
    total = p.sum(-1, keepdims=True) + 1e-12
    pn = p / total
    bp = [pn[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS.values()]
    ent = -(pn * np.log(pn + 1e-12)).sum(-1) / np.log(pn.shape[-1])
    kur = kurtosis(x, axis=-1, fisher=True)
    mdiff = np.abs(np.diff(x, axis=-1)).max(-1)
    xc = x - x.mean(-1, keepdims=True)
    acf1 = (xc[..., 1:] * xc[..., :-1]).sum(-1) / ((xc ** 2).sum(-1) + 1e-12)
    out = np.stack(bp + [ent, kur, mdiff, acf1], axis=-1)
    return np.nan_to_num(out).astype(np.float32)


def connectivity_features(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fisher-z correlation between every pair of signals, upper triangle.

    Pairs involving a masked signal are set to 0 (the mean of z under no
    coupling), which keeps the feature vector the same length for every
    session. The clamp before arctanh matters: identical signals give r = 1
    exactly and arctanh(1) = inf, which would poison any mean taken later.
    """
    xc = x - x.mean(-1, keepdims=True)
    sd = np.sqrt((xc ** 2).sum(-1, keepdims=True))
    xn = np.where(sd > 0, xc / np.where(sd > 0, sd, 1), 0.0)
    r = np.einsum("wit,wjt->wij", xn, xn)
    z = np.arctanh(np.clip(r, -1 + 1e-6, 1 - 1e-6))
    pair_ok = mask[:, :, None] & mask[:, None, :]
    z = np.where(pair_ok, z, 0.0)
    iu = np.triu_indices(x.shape[1], 1)
    return z[:, iu[0], iu[1]].astype(np.float32)


def window_features(x: np.ndarray, mask: np.ndarray, fs: float) -> np.ndarray:
    """Flat feature vector per window: spectral features of every signal + connectivity."""
    s = spectral_features(x, fs) * mask[..., None]
    return np.concatenate([s.reshape(len(x), -1), connectivity_features(x, mask)], axis=1)
