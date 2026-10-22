"""Wavelet motion correction (Molavi & Dumont, 2012)."""
from __future__ import annotations

import numpy as np
import pywt

from ._util import as_columns, fill_gaps, hazen_quantile

_IQR_PER_SD = 1.349     # interquartile range of a standard normal


def wavelet_correct(x: np.ndarray, basis: str = "sym8", std_thresh: float = 5.0,
                    iqr_thresh: float | None = None, remove_scaling: bool = False,
                    level: int | None = None) -> tuple[np.ndarray, dict]:
    """Zero outlying wavelet detail coefficients, level by level.

    Parameters
    ----------
    x : numpy.ndarray
        ``(T,)`` or ``(T, C)`` optical density.
    basis : str
        Wavelet name understood by PyWavelets.
    std_thresh : float
        A detail coefficient further than this many robust SDs
        (``IQR / 1.349``) from its level's median is zeroed. Ignored when
        ``iqr_thresh`` is given.
    iqr_thresh : float, optional
        Use Homer's rule instead: zero coefficients outside
        ``[q1 - k IQR, q3 + k IQR]`` with ``k = iqr_thresh`` (Homer uses 1.5).
    remove_scaling : bool
        Also zero the coarsest approximation, removing slow drift with the
        motion. Leave off when band-passing afterwards anyway.
    level : int, optional
        Decomposition depth; default is the deepest the length supports.

    Returns
    -------
    y : numpy.ndarray
        Corrected signal, same shape as ``x``.
    info : dict
        ``removed_fraction``: fraction of detail coefficients zeroed, per
        channel.

    Notes
    -----
    A haemodynamic response spreads over many coefficients while a motion
    artefact concentrates in a few large ones, so zeroing the outliers removes
    the artefact and keeps the response. That holds for sharp spikes and less
    well for slow, large movements.

    The threshold is set per level and on robust statistics, and neither is
    optional. Coefficient spread falls steeply with scale, so a threshold
    pooled over levels is set by the coarse ones and is far too lenient at the
    fine levels where spikes land. And a plain SD is inflated by the very
    artefact it should flag. This is the most aggressive motion method: with
    ``std_thresh`` much below 5 it starts attenuating the response itself.

    Signal extension is half-point symmetric (PyWavelets ``"symmetric"``).
    """
    x2, restore = as_columns(x)
    T, C = x2.shape
    y = x2.copy()
    info = {"removed_fraction": np.zeros(C)}
    filt_len = pywt.Wavelet(basis).dec_len

    for c in range(C):
        if np.isfinite(x2[:, c]).sum() < 32:
            continue
        n_level = pywt.dwt_max_level(T, filt_len) if level is None else level
        if n_level < 1:
            continue
        coeffs = pywt.wavedec(fill_gaps(x2[:, c]), basis, mode="symmetric", level=n_level)

        n_dropped = 0
        for d in coeffs[1:]:            # detail bands, coarsest first
            drop = _outliers(d, std_thresh, iqr_thresh)
            d[drop] = 0.0
            n_dropped += drop.sum()
        n_detail = sum(d.size for d in coeffs[1:])
        if n_detail:
            info["removed_fraction"][c] = n_dropped / n_detail
        if remove_scaling:
            coeffs[0][:] = 0.0
        y[:, c] = pywt.waverec(coeffs, basis, mode="symmetric")[:T]
    return restore(y), info


def _outliers(d: np.ndarray, std_thresh: float, iqr_thresh: float | None) -> np.ndarray:
    if d.size < 4:
        return np.zeros(d.size, dtype=bool)
    q1, q2, q3 = hazen_quantile(d, [0.25, 0.5, 0.75])
    iq = q3 - q1
    if not np.isfinite(iq) or iq <= 0:
        return np.zeros(d.size, dtype=bool)
    if iqr_thresh is not None:
        return (d > q3 + iqr_thresh * iq) | (d < q1 - iqr_thresh * iq)
    return np.abs(d - q2) > std_thresh * (iq / _IQR_PER_SD)
