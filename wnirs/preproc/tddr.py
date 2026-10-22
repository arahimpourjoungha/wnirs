"""Temporal Derivative Distribution Repair (Fishburn et al., 2019)."""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal

from ._util import as_columns, fill_gaps, filtfilt

_TUKEY = 4.685          # biweight tuning constant: 95 % efficiency under normality
_MAX_ITER = 50
_SPLIT_HZ = 0.5         # published variant corrects only below this frequency


def tddr(x: np.ndarray, fs: float = 1.0,
         variant: Literal["published", "fullband"] = "published") -> np.ndarray:
    """Motion-correct optical density by robustly reweighting its derivative.

    Parameters
    ----------
    x : numpy.ndarray
        ``(T,)`` or ``(T, C)`` optical density (apply before Beer-Lambert).
    fs : float
        Sampling rate in Hz. Needed for the 0.5 Hz split of the published
        variant; with ``fs <= 1`` that split cannot be designed and the
        published variant runs full-band.
    variant : {"published", "fullband"}
        ``"published"`` reproduces the paper and nirs-toolbox: the trace is
        split at 0.5 Hz, only the low band is corrected, the high band is
        added back unchanged, and the derivative's scale is a median absolute
        deviation. ``"fullband"`` corrects everything and uses a weighted RMS
        scale.

    Returns
    -------
    numpy.ndarray
        Corrected signal, same shape as ``x``. Channels with fewer than ten
        finite samples, or a constant trace, are returned unchanged.

    Notes
    -----
    Motion shows up as outliers in the first derivative. Each derivative
    sample gets a Tukey biweight from iteratively reweighted least squares,
    and the signal is rebuilt by integrating the weighted, centred
    derivative. Baseline shifts are flattened and the slow haemodynamics are
    untouched. No samples are removed, so timing and length are preserved.

    The variants trade off: the published one passes high-frequency content
    straight through, and a sharp spike is mostly high-frequency, so
    ``"fullband"`` corrects spikes better. On baseline shifts, which TDDR
    exists for, the published variant wins (r = 0.98 against 0.91 on
    simulated shifts). Use wavelet or spline for spikes and keep this on
    ``"published"``.
    """
    if variant not in ("published", "fullband"):
        raise ValueError(f'unknown variant "{variant}"; use "published" or "fullband"')
    published = variant == "published"

    x2, restore = as_columns(x)
    y = x2.copy()
    cols = np.flatnonzero(np.isfinite(x2).sum(axis=0) >= 10)
    if cols.size == 0:
        return restore(y)

    v = np.column_stack([fill_gaps(x2[:, c]) for c in cols])
    mu0 = v.mean(axis=0)
    v = v - mu0

    low = v
    if published and fs > 1:
        b, a = signal.butter(3, _SPLIT_HZ * 2 / fs)
        try:
            low = filtfilt(b, a, v)
        except ValueError:
            pass        # too short to filter: correct the trace whole
    high = v - low

    dv = np.diff(low, axis=0)
    w, mu = _robust_weights(dv, published)

    corrected = np.vstack([np.zeros((1, cols.size)), np.cumsum(w * (dv - mu), axis=0)])
    corrected -= corrected.mean(axis=0)

    # A constant channel has nothing to repair and keeps its original samples.
    moving = dv.any(axis=0)
    y[:, cols[moving]] = (corrected + high + mu0)[:, moving]
    return restore(y)


def _robust_weights(dv: np.ndarray, published: bool) -> tuple[np.ndarray, np.ndarray]:
    """Tukey-biweight IRLS for the location of each column of ``dv``.

    Columns iterate together but stop independently, exactly as they would
    one at a time: a column that has converged, or whose scale collapses to
    zero, keeps the weights and location it had at that point.
    """
    w = np.ones_like(dv)
    mu = np.zeros(dv.shape[1])
    prev = np.full(dv.shape[1], np.inf)
    running = dv.any(axis=0)
    for _ in range(_MAX_ITER):
        idx = np.flatnonzero(running)
        if idx.size == 0:
            break
        d, wi = dv[:, idx], w[:, idx]
        m = (wi * d).sum(axis=0) / wi.sum(axis=0)
        dev = d - m
        if published:
            # A MAD cannot be inflated by the outliers it is meant to find; a
            # plain SD lets a large artefact partly hide itself.
            sig = 1.4826 * np.median(np.abs(dev), axis=0)
        else:
            sig = np.sqrt((wi * dev ** 2).sum(axis=0) / wi.sum(axis=0))
        ok = sig > 0
        r = np.minimum(np.abs(dev[:, ok]) / (_TUKEY * sig[ok]), 1.0)
        w[:, idx[ok]] = (1 - r ** 2) ** 2
        converged = np.abs(m - prev[idx]) < 1e-10 * np.maximum(1.0, np.abs(m))
        mu[idx], prev[idx] = m, m
        running[idx] = ok & ~converged
    return w, mu
