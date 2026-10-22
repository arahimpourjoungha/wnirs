"""Nuisance regression: low-power reference, short channels, global signal."""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from ._util import PreprocWarning

_EPS = np.finfo(float).eps


def _project_out(x: np.ndarray, reg: np.ndarray,
                 fit: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Remove each column's projection on its own centred regressor.

    ``x`` and ``reg`` are ``(T, K)``. The coefficient is estimated on the rows
    where ``fit`` is True (all rows by default) and the correction is applied
    to every row. Returns the corrected columns and the coefficients, NaN
    where the regressor has no variance to remove.
    """
    r = reg - reg.mean(axis=0)
    xc = x - x.mean(axis=0)
    m = np.ones_like(r, dtype=bool) if fit is None else fit
    den = np.sum(m * r * r, axis=0)
    usable = den > _EPS
    beta = np.full(x.shape[1], np.nan)
    beta[usable] = np.sum(m * r * xc, axis=0)[usable] / den[usable]
    y = x.copy()
    y[:, usable] -= beta[usable] * r[:, usable]
    return y, beta


def lp_regress(od: np.ndarray, od_ref: np.ndarray | None,
               mode: Literal["perchannel", "mean"] = "perchannel") -> tuple[np.ndarray, dict]:
    """Regress the low-power (LP) reference trace out of each channel.

    Parameters
    ----------
    od : numpy.ndarray
        ``(T, C, W)`` optical density of the regular-power signal.
    od_ref : numpy.ndarray or None
        ``(T, C, W)`` optical density of the low-power signal.
    mode : {"perchannel", "mean"}
        Regress each channel's own LP trace, or the LP mean across channels
        (steadier when individual LP traces are very noisy). A channel whose
        own LP trace is incomplete falls back to the mean.

    Returns
    -------
    y : numpy.ndarray
    info : dict
        ``applied`` and ``beta``, ``(C, W)``.

    Notes
    -----
    The headband records every channel twice, at regular and at low LED
    power. The LP trace follows the same optical path with far less emitted
    light, so what survives in it is mostly *not* emitted light: ambient
    light, dark current, thermal drift and electronic noise. These are shared
    with the regular trace, and projecting them out removes a nuisance no
    filter can separate by frequency. This is not short-separation
    regression, which removes scalp haemodynamics; the two can be combined.
    """
    C, W = od.shape[1:]
    y = od.copy()
    info = {"applied": np.zeros((C, W), dtype=bool), "beta": np.full((C, W), np.nan),
            "mode": mode}
    if od_ref is None or not np.isfinite(od_ref).any():
        warnings.warn("no usable low-power reference; LP regression skipped",
                      PreprocWarning, stacklevel=2)
        return y, info

    for w in range(W):
        ref = od_ref[:, :, w]
        mean_ref = np.nanmean(ref, axis=1)
        reg = np.repeat(mean_ref[:, None], C, axis=1) if mode == "mean" else ref.copy()
        incomplete = ~np.isfinite(reg).all(axis=0)
        reg[:, incomplete] = mean_ref[:, None]
        cols = np.isfinite(od[:, :, w]).all(axis=0) & np.isfinite(reg).all(axis=0)
        y[:, cols, w], beta = _project_out(od[:, cols, w], reg[:, cols])
        info["beta"][cols, w] = beta
        info["applied"][cols, w] = np.isfinite(beta)
    return y, info


def ssr(x: np.ndarray, sources: np.ndarray, is_short: np.ndarray,
        mode: Literal["samesource", "mean"] = "samesource",
        usable: np.ndarray | None = None,
        clip_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Short-separation regression: remove scalp haemodynamics.

    Parameters
    ----------
    x : numpy.ndarray
        ``(T, C)`` one chromophore (call once for HbO, once for HbR).
    sources : numpy.ndarray
        ``(C,)`` source number of each channel.
    is_short : numpy.ndarray
        ``(C,)`` True for short-separation channels.
    mode : {"samesource", "mean"}
        ``"samesource"`` pairs each long channel with the short on its own
        source, falling back to the mean short where that source has none;
        ``"mean"`` always uses the mean of all shorts.
    usable : numpy.ndarray, optional
        ``(C,)`` False for channels that failed the saturation / dark checks.
        Only usable, fully finite shorts act as regressors: a clipped short
        would inject a flat line into every channel.
    clip_mask : numpy.ndarray, optional
        ``(T, C)`` True where a sample was clipped and interpolated.

    Returns
    -------
    y : numpy.ndarray
        Long channels with their regressor projected out; shorts unchanged so
        they stay available for inspection.
    info : dict
        ``applied``, ``beta`` and ``source`` per channel (the short's source,
        0 for the mean short, NaN where not applied).

    Notes
    -----
    Each long channel loses ``beta * s`` with ``beta = s'x / s's`` on centred
    traces. Beta is estimated only on samples where the short was a real
    measurement: clipped stretches were interpolated, and letting the
    interpolation into the fit would bias beta towards what it invented. The
    correction itself is applied to the whole record. If too few clean
    samples remain (fewer than 20 or a quarter of the record), all are used.
    """
    T, C = x.shape
    usable = np.ones(C, dtype=bool) if usable is None else np.asarray(usable, dtype=bool)
    clip_mask = np.zeros((T, C), dtype=bool) if clip_mask is None else clip_mask
    is_short = np.asarray(is_short, dtype=bool)
    sources = np.asarray(sources)
    y = x.copy()
    info = {"applied": np.zeros(C, dtype=bool), "beta": np.full(C, np.nan),
            "source": np.full(C, np.nan)}

    shorts = np.flatnonzero(is_short & usable)
    if shorts.size == 0:
        warnings.warn("no usable short-separation channels; SSR skipped",
                      PreprocWarning, stacklevel=2)
        return y, info
    shorts = shorts[np.isfinite(x[:, shorts]).all(axis=0)]
    if shorts.size == 0:
        warnings.warn("every short channel is non-finite; SSR skipped",
                      PreprocWarning, stacklevel=2)
        return y, info

    longs = np.flatnonzero(~is_short & np.isfinite(x).all(axis=0))
    # Column j of the regressor matrices belongs to long channel longs[j]:
    # its first same-source short, or the mean short.
    first_short = {}
    for s in shorts[::-1]:
        first_short[sources[s]] = s
    pick = np.array([first_short.get(sources[c], -1) if mode == "samesource" else -1
                     for c in longs], dtype=int)
    own = pick >= 0
    reg = np.empty((T, longs.size))
    clipped = np.empty((T, longs.size), dtype=bool)
    reg[:, own] = x[:, pick[own]]
    clipped[:, own] = clip_mask[:, pick[own]]
    reg[:, ~own] = x[:, shorts].mean(axis=1, keepdims=True)
    clipped[:, ~own] = clip_mask[:, shorts].any(axis=1, keepdims=True)

    fit = ~clipped
    too_few = fit.sum(axis=0) < max(20, 0.25 * T)
    fit[:, too_few] = True

    y[:, longs], beta = _project_out(x[:, longs], reg, fit)
    info["beta"][longs] = beta
    info["applied"][longs] = np.isfinite(beta)
    info["source"][longs] = np.where(own, sources[np.maximum(pick, 0)], 0)
    return y, info


def global_regress(x: np.ndarray, include: np.ndarray) -> tuple[np.ndarray, dict]:
    """Regress the mean of the included channels out of every channel.

    Parameters
    ----------
    x : numpy.ndarray
        ``(T, C)`` one chromophore.
    include : numpy.ndarray
        ``(C,)`` channels that build the global signal: long channels that
        passed QC. A short or dead channel would put scalp signal or a flat
        line into the regressor meant to stand for the whole montage.

    Returns
    -------
    y : numpy.ndarray
    info : dict
        ``applied`` and ``beta`` per channel.

    Notes
    -----
    This removes systemic physiology shared across the montage when short
    channels are unavailable. The cost is real: a genuinely global response
    is indistinguishable from global noise by this criterion and is removed
    too, and the regression can induce negative correlations between
    channels. Right for connectivity following the published pipeline; wrong
    if a whole-montage activation is expected. Needs at least three usable
    channels.
    """
    C = x.shape[1]
    y = x.copy()
    info = {"applied": np.zeros(C, dtype=bool), "beta": np.full(C, np.nan)}
    finite = np.isfinite(x).all(axis=0)
    members = np.asarray(include, dtype=bool) & finite
    if members.sum() < 3:
        warnings.warn("fewer than 3 usable channels; global signal regression skipped",
                      PreprocWarning, stacklevel=2)
        return y, info
    g = x[:, members].mean(axis=1, keepdims=True)
    y[:, finite], beta = _project_out(x[:, finite], np.repeat(g, finite.sum(), axis=1))
    info["beta"][finite] = beta
    info["applied"][finite] = np.isfinite(beta)
    return y, info
