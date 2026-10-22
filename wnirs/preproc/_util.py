"""Small numerical helpers shared by the preprocessing steps.

Several of these pin a numerical convention exactly (rounding, linear
extrapolation, moving-window edge handling, ``filtfilt`` padding), because
the steps are validated sample-for-sample against reference outputs and the
defaults only mean the same thing if these agree.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal


class PreprocWarning(UserWarning):
    """A preprocessing step was skipped or adapted to what the data supports."""


def mround(x: float) -> int:
    """Round half away from zero.

    Python's ``round`` rounds half to even, which turns e.g. a 2.5-sample
    window into 2 rather than 3 and silently changes window lengths derived
    from ``seconds * fs``.
    """
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


def as_columns(x: np.ndarray) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """View a 1-D trace as a single column; return a function that undoes it."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[:, None], lambda y: y[:, 0]
    if x.ndim != 2:
        raise ValueError(f"expected a (T,) or (T, C) array, got shape {x.shape}")
    return x, lambda y: y


def fill_gaps(v: np.ndarray) -> np.ndarray:
    """Linearly interpolate non-finite samples, extrapolating past the ends.

    The filters and motion methods need a gap-free trace, and a linear bridge is
    the least informative thing that can be put in a gap.
    """
    ok = np.isfinite(v)
    if ok.all() or not ok.any():
        return v.copy()
    idx = np.flatnonzero(ok)
    if idx.size == 1:
        return np.full_like(v, v[idx[0]])
    t = np.arange(v.size)
    out = np.interp(t, idx, v[idx])
    head, tail = t < idx[0], t > idx[-1]
    slope0 = (v[idx[1]] - v[idx[0]]) / (idx[1] - idx[0])
    slope1 = (v[idx[-1]] - v[idx[-2]]) / (idx[-1] - idx[-2])
    out[head] = v[idx[0]] + (t[head] - idx[0]) * slope0
    out[tail] = v[idx[-1]] + (t[tail] - idx[-1]) * slope1
    return out


def _windows(v: np.ndarray, k: int) -> np.ndarray:
    """(T, k) NaN-padded windows for the moving-window statistics.

    An odd window is centred; an even one covers ``k/2`` samples back and
    ``k/2 - 1`` forward. Padding with NaN and reducing with the nan-aware
    statistic shrinks (or truncates) the window at the endpoints.
    """
    back = k // 2
    fwd = k - 1 - back
    padded = np.concatenate([np.full(back, np.nan), v, np.full(fwd, np.nan)])
    return sliding_window_view(padded, k)


def moving_std(v: np.ndarray, k: int) -> np.ndarray:
    """Moving sample standard deviation over k samples, shrinking at the ends."""
    return np.nanstd(_windows(v, k), axis=1, ddof=1)


def moving_median(v: np.ndarray, k: int) -> np.ndarray:
    """Running median over k samples, truncated at the ends."""
    return np.nanmedian(_windows(v, k), axis=1)


def filtfilt(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zero-phase filtering along axis 0, padded by ``3 * (order)`` samples.

    The padding is odd reflection of ``3 * (order)`` samples; SciPy's default
    is ``3 * (order + 1)``. The difference is visible at the edges, so the
    shorter length is used.
    """
    padlen = 3 * (max(len(a), len(b)) - 1)
    return signal.filtfilt(b, a, x, axis=0, padtype="odd", padlen=padlen)


def hazen_quantile(x: np.ndarray, q, axis=None) -> np.ndarray:
    """Quantiles by Hazen's rule.

    The sorted samples sit at probabilities ``(i - 0.5) / n`` and are
    interpolated linearly between, which is Hazen's definition. NumPy's
    default (linear, ``(i - 1) / (n - 1)``) gives different values on the
    short coefficient vectors the wavelet thresholds are set on.
    """
    return np.quantile(x, q, axis=axis, method="hazen")
