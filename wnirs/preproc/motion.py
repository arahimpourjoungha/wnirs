"""Motion-correction dispatcher, plus the two simplest methods."""
from __future__ import annotations

import inspect
import warnings
from collections.abc import Sequence

import numpy as np
from scipy import signal

from ._util import PreprocWarning, as_columns, fill_gaps, mround
from .spline import spline_correct
from .tddr import tddr
from .wavelet import wavelet_correct

#: Motion-correction methods by name, with when to reach for each.
MOTION_METHODS: dict[str, str] = {
    "none": "No correction. Use to see the artefacts, or when a later step "
            "(AR-IRLS) already downweights them.",
    "tddr": "Temporal Derivative Distribution Repair (Fishburn et al. 2019). Needs no "
            "threshold and never removes samples, which is why it is the default.",
    "spline": "Spline interpolation (Scholkmann et al. 2010). Strong on discrete spikes "
              "and steps, but only corrects what its moving-SD detector finds.",
    "wavelet": "Wavelet filtering (Molavi & Dumont 2012). Zeroes outlying sym8 detail "
               "coefficients per level. The most aggressive option.",
    "splinesg": "Spline, then Savitzky-Golay (Jahani et al. 2018): the spline removes "
                "the large detected artefacts, the smoother the residual spikes.",
    "sgolay": "Savitzky-Golay smoothing alone, for small frequent artefacts rather "
              "than large discrete ones.",
    "pca": "Remove the leading spatial principal components, on the assumption that "
           "motion is shared across the montage. Also removes any global response.",
}


def savgol_correct(x: np.ndarray, fs: float, sg_order: int = 3,
                   sg_window: float = 1.0) -> np.ndarray:
    """Savitzky-Golay smoothing with the window given in seconds.

    The window is made odd and longer than the polynomial order. Edges are
    handled by fitting the polynomial to the first and last windows.
    Channels with fewer finite samples than the
    window are left unchanged.
    """
    x2, restore = as_columns(x)
    T = x2.shape[0]
    n = max(mround(sg_window * fs) | 1, sg_order + 2) | 1     # "| 1": next odd number
    y = x2.copy()
    if n >= T:
        warnings.warn("recording is shorter than the Savitzky-Golay window; step skipped",
                      PreprocWarning, stacklevel=2)
        return restore(y)
    for c in np.flatnonzero(np.isfinite(x2).sum(axis=0) >= n):
        y[:, c] = signal.savgol_filter(fill_gaps(x2[:, c]), n, sg_order, mode="interp")
    return restore(y)


def pca_correct(x: np.ndarray, pca_components: int = 1) -> tuple[np.ndarray, dict]:
    """Remove the leading spatial principal components across channels.

    Only fully finite channels take part. At least one component is always
    kept, so a two-channel montage is never reduced to its mean.

    Returns
    -------
    y : numpy.ndarray
    info : dict
        ``removed`` (components requested) and ``explained`` (percent variance
        of every component, before removal).
    """
    x2, restore = as_columns(x)
    y = x2.copy()
    info = {"removed": pca_components, "explained": np.zeros(0)}
    ok = np.isfinite(x2).all(axis=0)
    if ok.sum() < 2 or pca_components < 1:
        return restore(y), info
    X = x2[:, ok]
    mu = X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    info["explained"] = 100 * s ** 2 / np.sum(s ** 2)
    k = min(pca_components, Vt.shape[0] - 1)
    y[:, ok] = X - (U[:, :k] * s[:k]) @ Vt[:k]
    return restore(y), info


def _call(fn, /, *args, **opts):
    """Call ``fn`` with only the options it accepts.

    One option set is shared by every method, so
    e.g. ``std_thresh`` steers both the spline detector and the wavelet
    threshold when they are chained.
    """
    accepted = inspect.signature(fn).parameters
    return fn(*args, **{k: v for k, v in opts.items() if k in accepted})


def motion_correct(x: np.ndarray, fs: float, method: str | Sequence[str] = "tddr", /,
                   **opts) -> tuple[np.ndarray, dict]:
    """Apply one motion-correction method, or several in sequence.

    Parameters
    ----------
    x : numpy.ndarray
        ``(T,)`` or ``(T, C)`` optical density.
    fs : float
        Sampling rate in Hz.
    method : str or sequence of str
        A key of :data:`MOTION_METHODS`, or several applied in the order given.
    **opts
        Options for the methods (e.g. ``variant``, ``t_motion``,
        ``std_thresh``, ``iqr_thresh``, ``sg_window``, ``pca_components``);
        each method receives the ones it accepts.

    Returns
    -------
    y : numpy.ndarray
    info : dict
        ``method`` (the names applied) and ``detail`` (each method's own
        diagnostics, or None).

    Notes
    -----
    Correcting OD rather than haemoglobin matters: an artefact is a coupling
    change that hits both wavelengths, and correcting before Beer-Lambert
    keeps them consistent with each other.
    """
    methods = [method] if isinstance(method, str) else list(method)
    unknown = [m for m in methods if m.lower() not in MOTION_METHODS]
    if unknown:
        raise ValueError(f"unknown motion method(s) {unknown}; available: {list(MOTION_METHODS)}")

    y = np.asarray(x, dtype=float)
    detail: list[dict | None] = []
    for m in (m.lower() for m in methods):
        d = None
        if m == "tddr":
            y = _call(tddr, y, fs, **opts)
        elif m == "spline":
            y, d = _call(spline_correct, y, fs, **opts)
        elif m == "wavelet":
            y, d = _call(wavelet_correct, y, **opts)
        elif m == "splinesg":
            y, d = _call(spline_correct, y, fs, **opts)
            y = _call(savgol_correct, y, fs, **opts)
        elif m == "sgolay":
            y = _call(savgol_correct, y, fs, **opts)
        elif m == "pca":
            y, d = _call(pca_correct, y, **opts)
        detail.append(d)
    return y, {"method": methods, "detail": detail}
