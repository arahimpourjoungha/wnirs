"""From detector intensity to optical density: channel flags, dark level, OD."""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np

from ._util import PreprocWarning, fill_gaps, hazen_quantile

#: Statuses that remove a channel from every analysis.
UNUSABLE = ("saturated", "missing", "dark")

DarkMethod = Literal["unused", "quantile", "none"] | float | Sequence[float]


def flag_channels(intensity: np.ndarray, sat_level: float = 5.0, sat_tol: float = 0.02,
                  sat_max_frac: float = 0.20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify channels by detector saturation and repair intermittent clipping.

    Parameters
    ----------
    intensity : numpy.ndarray
        ``(T, C, W)`` detector intensity.
    sat_level : float
        The detector ceiling, in the file's units.
    sat_tol : float
        Readings within this of the ceiling count as clipped.
    sat_max_frac : float
        A channel clipped on more than this fraction of samples (at any
        wavelength) is dropped; below it the clipped samples are repaired.

    Returns
    -------
    status : numpy.ndarray
        ``(C,)`` object array of ``"ok"``, ``"missing"`` (a wavelength with no
        finite sample), ``"saturated"``, ``"intermittentClip"`` or
        ``"nearCeiling"`` (mean within 10 % of the ceiling without clipping).
        Object dtype, so later stages can write longer QC labels into it.
    intensity : numpy.ndarray
        A copy with saturated and missing channels set to NaN and the clipped
        samples of intermittent channels linearly interpolated.
    clip_mask : numpy.ndarray
        ``(T, C)`` True where any wavelength was clipped.

    Notes
    -----
    A channel pinned at the ceiling carries no information -- its variance is
    near zero and its OD a constant -- so it is blanked rather than left to
    put a flat line into every average. Clipping on this headband is often
    intermittent, though: a channel sits just under the ceiling and crosses it
    now and then. Throwing that channel away because 2 % of its samples clip
    loses far more than it protects, so its clipped samples, which are known
    invalid readings rather than a flat signal, are interpolated instead.
    """
    I = np.array(intensity, dtype=float)
    finite = np.isfinite(I)
    n_finite = finite.sum(axis=0)                                  # (C, W)
    present = n_finite > 0
    hit = finite & (I >= sat_level - sat_tol)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(present, hit.sum(axis=0) / n_finite, 0.0)
        mean = np.where(present, np.nansum(I, axis=0) / n_finite, -np.inf)
    clip_mask = hit.any(axis=2)
    clip_frac = frac.max(axis=1)

    status = np.full(I.shape[1], "ok", dtype=object)
    status[(mean >= 0.90 * sat_level).any(axis=1)] = "nearCeiling"
    status[clip_frac > 0] = "intermittentClip"
    status[clip_frac > sat_max_frac] = "saturated"
    status[~present.all(axis=1)] = "missing"

    I[:, np.isin(status, ("saturated", "missing"))] = np.nan
    for c in np.flatnonzero(status == "intermittentClip"):
        for w in range(I.shape[2]):
            bad = clip_mask[:, c] | ~np.isfinite(I[:, c, w])
            if bad.any() and not bad.all():
                v = I[:, c, w].copy()
                v[bad] = np.nan
                I[:, c, w] = fill_gaps(v)
    return status, I, clip_mask


def dark_level(intensity: np.ndarray, method: DarkMethod = "unused",
               unused: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Estimate the detector's zero-light level for each wavelength.

    Parameters
    ----------
    intensity : numpy.ndarray
        ``(T, C, W)`` intensity of the montage channels.
    method : {"unused", "quantile", "none"} or float or array_like
        ``"unused"`` takes the median, over source-detector pairs the montage
        never illuminates, of each pair's mean reading; it needs at least four
        such pairs and falls back to ``"quantile"`` otherwise. ``"quantile"``
        is the 1st percentile over all channels and samples. ``"none"``
        returns zeros (only for exports that are already dark-corrected). A
        number is used as given, per wavelength or once for all.
    unused : numpy.ndarray, optional
        ``(T, P, W)`` intensity of the unused pairs, for ``"unused"``.

    Returns
    -------
    dark : numpy.ndarray
        ``(W,)`` level to subtract.
    info : dict
        ``method`` (as actually used), ``n_unused`` and ``spread`` (robust SD
        of the unused-pair means).

    Notes
    -----
    The detector output does not start at zero: an unilluminated channel on
    this headband reads about 1.511 and a good one 1.6-2.5, so the
    light-dependent part is a small fraction of what is recorded. OD is the
    log of an intensity *ratio*, and an additive offset crushes it::

        -log10((dark + s) / (dark + mean(s)))     almost flat
        -log10(s / mean(s))                       the real modulation

    Leaving the offset in attenuates the haemodynamic response towards zero
    and makes weak channels look clean (a tiny CV on a big constant), so this
    is the step that most affects the results.
    """
    n_wl = intensity.shape[2]
    info = {"method": method, "n_unused": 0, "spread": np.full(n_wl, np.nan)}
    if not isinstance(method, str):
        info["method"] = "user-supplied"
        return np.broadcast_to(np.asarray(method, dtype=float), (n_wl,)).copy(), info
    method = method.lower()
    if method == "none":
        return np.zeros(n_wl), info
    if method == "unused":
        dark = _from_unused(unused, n_wl, info)
        if np.isfinite(dark).all():
            return dark, info
        warnings.warn("could not estimate the dark level from unused source-detector "
                      "pairs; falling back to the 1st percentile", PreprocWarning, stacklevel=2)
    elif method != "quantile":
        raise ValueError(f'unknown dark-level method "{method}"')
    info["method"] = "quantile"
    dark = np.full(n_wl, np.nan)
    for w in range(n_wl):
        v = intensity[:, :, w]
        v = v[np.isfinite(v)]
        if v.size:
            dark[w] = hazen_quantile(v, 0.01)
    return dark, info


def _from_unused(unused: np.ndarray | None, n_wl: int, info: dict) -> np.ndarray:
    dark = np.full(n_wl, np.nan)
    if unused is None or unused.shape[1] < 4:
        return dark
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN pairs
        means = np.nanmean(unused, axis=0)                     # (P, W)
    for w in range(n_wl):
        m = means[:, w][np.isfinite(means[:, w])]
        if m.size < 4:
            continue
        # An unused pair can still catch stray light from a neighbouring
        # optode; the median is robust to those few, and the spread reports
        # whether the rest agree.
        dark[w] = np.median(m)
        info["spread"][w] = 1.4826 * np.median(np.abs(m - dark[w]))
    info["n_unused"] = unused.shape[1]
    info["method"] = "unused"
    return dark


def optical_density(intensity: np.ndarray) -> np.ndarray:
    """``OD = -log10(I / mean(I))`` per channel and wavelength (base 10).

    Base 10 matches Prahl's extinction coefficients used by
    :func:`~wnirs.preproc.mbll`. A channel whose mean intensity is not
    positive -- after dark subtraction, one that saw no light -- has no
    defined OD and is returned as NaN; :func:`dark_channels` reports them.
    Samples at or below zero in an otherwise lit channel are clamped to
    machine epsilon, and missing samples stay missing.
    """
    I = np.asarray(intensity, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(I, axis=0)
    lit = np.isfinite(mu) & (mu > 0)
    od = np.full(I.shape, np.nan)
    with np.errstate(invalid="ignore"):
        od[:, lit] = -np.log10(np.maximum(I[:, lit], np.finfo(float).eps) / mu[lit])
    return od


def dark_channels(intensity: np.ndarray) -> np.ndarray:
    """``(C,)`` True where any wavelength's mean intensity is not positive."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(intensity, axis=0)
    return (~np.isfinite(mu) | (mu <= 0)).any(axis=1)
