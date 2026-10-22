"""Signal-processing primitives with pinned numerical conventions.

The QC metrics are compared channel by channel against reference outputs, so
the details where SciPy's defaults would differ are pinned here:
``filtfilt`` edge padding, Welch segmenting in ``cpsd``, NaN-ignoring
``min``/``max`` and round-half-away-from-zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class Filter:
    b: np.ndarray
    a: np.ndarray


def design_butter(fs: float, band: float | tuple[float, float],
                  kind: Literal["bandpass", "highpass"]) -> Optional[Filter]:
    """Second-order Butterworth design, or None when the band is not realisable at ``fs``."""
    wn = np.atleast_1d(np.asarray(band, dtype=float)) / (fs / 2)
    if not np.all(np.isfinite(wn)) or np.any(wn <= 0) or np.any(wn >= 1):
        return None
    b, a = signal.butter(2, wn if wn.size > 1 else wn[0], btype=kind)
    return Filter(b, a)


def filtfilt(filt: Filter, x: np.ndarray) -> np.ndarray:
    """Zero-phase filtering padded by ``3 * (order)`` samples of odd reflection."""
    n_pad = 3 * (max(filt.a.size, filt.b.size) - 1)
    return signal.filtfilt(filt.b, filt.a, x, padtype="odd", padlen=n_pad)


def filt_safe(x: np.ndarray, filt: Optional[Filter]) -> Optional[np.ndarray]:
    """Filter a trace with gaps: NaNs are linearly interpolated (and extrapolated) first.

    None when there is no filter, fewer than 12 finite samples, or the trace
    is too short for the edge padding.
    """
    if filt is None:
        return None
    ok = np.isfinite(x)
    if ok.sum() < 12:
        return None
    x = np.asarray(x, dtype=float)
    if not ok.all():
        x = interp_extrap(np.flatnonzero(ok), x[ok], np.arange(x.size))
    try:
        return filtfilt(filt, x)
    except ValueError:
        return None


def interp_extrap(xp: np.ndarray, fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation, extrapolated linearly past the ends."""
    y = np.interp(x, xp, fp)
    if xp.size > 1:
        lo, hi = x < xp[0], x > xp[-1]
        y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y


def cpsd(x: np.ndarray, y: np.ndarray, nfft: int, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided cross power spectral density, Welch-segmented.

    Eight Hamming-windowed segments with 50 % overlap; a segment longer than
    ``nfft`` is wrapped (summed modulo ``nfft``) rather than rejected.
    """
    n = x.size
    seg = int(n // 4.5)
    overlap = seg // 2
    step = seg - overlap
    n_seg = (n - overlap) // step
    win = np.hamming(seg)
    acc = np.zeros(nfft, dtype=complex)
    for k in range(n_seg):
        sl = slice(k * step, k * step + seg)
        acc += np.conj(_wrapped_fft(x[sl] * win, nfft)) * _wrapped_fft(y[sl] * win, nfft)
    pxy = acc / n_seg / (fs * np.sum(win ** 2))
    n_one = nfft // 2 + 1
    pxy = pxy[:n_one]
    pxy[1:n_one - (1 if nfft % 2 == 0 else 0)] *= 2
    return pxy, np.arange(n_one) * fs / nfft


def _wrapped_fft(x: np.ndarray, nfft: int) -> np.ndarray:
    if x.size > nfft:
        x = np.pad(x, (0, -x.size % nfft)).reshape(-1, nfft).sum(axis=0)
    return np.fft.fft(x, nfft)


def round_half_away(x: float) -> int:
    """Round half away from zero, unlike Python's half-to-even."""
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


def nan_min(a: float, b: float) -> float:
    """``min(a, b)`` that ignores a NaN argument."""
    return float(np.fmin(a, b))


def nan_max(a: float, b: float) -> float:
    return float(np.fmax(a, b))


def robust_sigma(x: np.ndarray) -> float:
    """Normal-consistent median absolute deviation."""
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))
