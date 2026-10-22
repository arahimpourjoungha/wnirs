"""Frequency filtering and detrending of haemoglobin traces."""
from __future__ import annotations

import warnings

import numpy as np
from scipy import signal

from ._util import PreprocWarning, as_columns, fill_gaps, filtfilt, moving_median, mround

#: Haemodynamic band in Hz: drift below it, cardiac and respiratory above it.
DEFAULT_BAND: tuple[float, float] = (0.01, 0.09)


def butter_filtfilt(x: np.ndarray, fc: float, fs: float, btype: str) -> np.ndarray:
    """Third-order Butterworth, applied forwards and backwards along axis 0.

    Zero phase, so block timing is preserved. The order is kept low because
    fNIRS cutoffs sit very close to DC, where high-order IIR designs in
    transfer-function form go numerically unstable.
    """
    wn = min(max(fc / (fs / 2), 1e-6), 0.999)
    b, a = signal.butter(3, wn, btype)
    return filtfilt(b, a, x)


def bandpass(x: np.ndarray, fs: float, band=DEFAULT_BAND, median_win: int = 0) -> np.ndarray:
    """Band-pass (or one-sided) filter, with an optional median despike.

    Parameters
    ----------
    x : numpy.ndarray
        ``(T,)`` or ``(T, C)``; columns are filtered independently.
    fs : float
        Sampling rate in Hz.
    band : (float, float)
        ``(low, high)`` cutoffs in Hz. NaN at either end makes the filter
        one-sided. A high cutoff at or above Nyquist is left open, with a
        warning.
    median_win : int
        Median-filter length in samples applied after the band-pass; 0 skips.

    Returns
    -------
    numpy.ndarray
        Filtered data, same shape as ``x``. Gaps are bridged linearly before
        filtering; all-NaN columns stay NaN.

    Notes
    -----
    The trace is mirrored by half its length (capped at 30 s) on each side
    before filtering and trimmed afterwards. A 0.01 Hz high-pass has a
    transient of tens of seconds, and without the mirror it would eat into the
    first and last blocks of the run.
    """
    x2, restore = as_columns(x)
    T = x2.shape[0]
    lo, hi = (float(b) for b in band)
    if np.isfinite(hi) and hi >= fs / 2:
        warnings.warn(f"high cutoff {hi:.3g} Hz is at or above Nyquist ({fs / 2:.3g} Hz); "
                      "leaving it open", PreprocWarning, stacklevel=2)
        hi = np.nan

    y = x2.copy()
    cols = np.isfinite(x2).any(axis=0)
    if not cols.any():
        return restore(y)

    pad = min(T - 1, max(1, mround(min(30 * fs, T / 2))))
    v = np.column_stack([fill_gaps(x2[:, c]) for c in np.flatnonzero(cols)])
    v = np.pad(v, ((pad, pad), (0, 0)), mode="reflect")
    if np.isfinite(lo) and lo > 0:
        v = butter_filtfilt(v, lo, fs, "highpass")
    if np.isfinite(hi):
        v = butter_filtfilt(v, hi, fs, "lowpass")
    if median_win > 1:
        v = np.column_stack([moving_median(col, median_win) for col in v.T])
    y[:, cols] = v[pad:pad + T]
    return restore(y)


def detrend(x: np.ndarray) -> np.ndarray:
    """Remove a least-squares line from each column, over its finite samples.

    Columns with fewer than three finite samples are returned unchanged;
    non-finite samples stay where they are.
    """
    x2, restore = as_columns(x)
    y = x2.copy()
    finite = np.isfinite(x2)
    full = finite.all(axis=0)
    if full.any():
        y[:, full] = signal.detrend(x2[:, full], axis=0)
    for c in np.flatnonzero(~full & (finite.sum(axis=0) >= 3)):
        ok = finite[:, c]
        y[ok, c] = signal.detrend(x2[ok, c])
    return restore(y)
