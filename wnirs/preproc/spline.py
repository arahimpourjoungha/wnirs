"""Spline-interpolation motion correction (Scholkmann et al., 2010)."""
from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.ndimage import maximum_filter1d

from ._util import as_columns, fill_gaps, moving_std, mround


def smoothing_spline(y: np.ndarray, p: float) -> np.ndarray:
    """Values at unit-spaced sites of the cubic smoothing spline with parameter p.

    Minimises ``p * sum((y - f)**2) + (1 - p) * integral(f''**2)``
    (``p = 1`` interpolates, ``p = 0`` is
    the least-squares line). Solved with Reinsch's algorithm: the second
    derivatives ``u`` at the interior knots satisfy the pentadiagonal system
    ``(6 (1 - p) Q'Q + p R) u = Q'y``, and the fit is ``y - 6 (1 - p) Q u``,
    where ``Q'`` is the second-difference operator and ``R`` is
    ``tridiag(1, 4, 1)`` for unit spacing.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return y.copy()
    m = n - 2
    lam = 6.0 * (1.0 - p)
    # Q'Q for the second-difference operator has constant diagonals 1, -4, 6, -4, 1.
    main = np.full(m, 6.0 * lam + 4.0 * p)
    off1 = np.full(m - 1, -4.0 * lam + p)
    off2 = np.full(max(m - 2, 0), lam)
    ab = np.zeros((5, m))
    ab[0, 2:] = off2
    ab[1, 1:] = off1
    ab[2] = main
    ab[3, :-1] = off1
    ab[4, :-2] = off2
    u = linalg.solve_banded((2, 2), ab, y[:-2] - 2 * y[1:-1] + y[2:])
    qu = np.zeros(n)
    qu[:-2] += u
    qu[1:-1] -= 2 * u
    qu[2:] += u
    return y - lam * qu


def spline_correct(x: np.ndarray, fs: float, t_motion: float = 0.5, t_mask: float = 1.0,
                   std_thresh: float = 5.0, amp_thresh: float = 0.5,
                   p: float = 0.99) -> tuple[np.ndarray, dict]:
    """Detect motion artefacts and subtract a smoothing spline from each one.

    Parameters
    ----------
    x : numpy.ndarray
        ``(T,)`` or ``(T, C)`` optical density.
    fs : float
        Sampling rate in Hz.
    t_motion : float
        Window of the moving standard deviation, in seconds.
    t_mask : float
        Seconds marked either side of each detection.
    std_thresh : float
        A sample is artefact when its moving SD exceeds this multiple of the
        channel's median moving SD.
    amp_thresh : float
        A sample-to-sample OD change larger than this is artefact whatever the
        SD says.
    p : float
        Spline smoothing parameter in [0, 1] (see :func:`smoothing_spline`).

    Returns
    -------
    y : numpy.ndarray
        Corrected signal, same shape as ``x``.
    info : dict
        ``n_segments`` (artefact segments per channel) and ``fraction``
        (fraction of samples inside a segment).

    Notes
    -----
    Each detected segment is fitted with a smoothing spline, which follows the
    artefact but not the slower haemodynamics, and the fit is subtracted.
    That leaves a step at each end of the segment, so the trace is then
    re-levelled to join up. Samples outside the segments are untouched: clean
    data is never distorted, but an artefact the detector misses is never
    corrected either.
    """
    x2, restore = as_columns(x)
    T, C = x2.shape
    y = x2.copy()
    info = {"n_segments": np.zeros(C, dtype=int), "fraction": np.zeros(C)}

    win = max(3, mround(t_motion * fs))
    mask = max(1, mround(t_mask * fs))

    for c in range(C):
        if np.isfinite(x2[:, c]).sum() < 4 * win:
            continue
        v = fill_gaps(x2[:, c])
        bad = _detect(v, win, mask, std_thresh, amp_thresh)
        if not bad.any():
            continue
        seg = _segments(bad)
        info["n_segments"][c] = len(seg)
        info["fraction"][c] = bad.mean()

        w = v.copy()
        for i1, i2 in seg:
            if i2 - i1 + 1 < 4:
                continue
            fit = smoothing_spline(v[i1:i2 + 1], p)
            # Subtracting the fit also removes the segment's level; adding its
            # mean back keeps the segment where it was until re-levelling.
            w[i1:i2 + 1] = v[i1:i2 + 1] - fit + fit.mean()
        y[:, c] = _relevel(w, seg)
    return restore(y), info


def _detect(v: np.ndarray, win: int, mask: int, std_thresh: float,
            amp_thresh: float) -> np.ndarray:
    msd = moving_std(v, win)
    ref = np.nanmedian(msd)
    jump = np.concatenate([[0.0], np.abs(np.diff(v))])
    bad = (msd > std_thresh * max(ref, np.finfo(float).eps)) | (jump > amp_thresh)
    # Widen each detection so the spline covers the whole excursion, not just
    # its steepest part.
    return maximum_filter1d(bad.astype(np.uint8), size=2 * mask + 1, mode="constant") > 0


def _segments(bad: np.ndarray) -> np.ndarray:
    """(n, 2) inclusive start/end indices of the runs of True in ``bad``."""
    edges = np.diff(np.concatenate([[0], bad.astype(np.int8), [0]]))
    return np.column_stack([np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1])


def _relevel(w: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Shift every segment, and everything after it, to remove its edge steps.

    Each segment start is moved to continue from the sample before it, and the
    sample after its end to continue from its last sample. Runs are separated
    by at least one sample, so every shift is a difference of two samples that
    all earlier shifts moved together; the shifts can therefore be read off
    the unshifted trace and applied as one cumulative offset.
    """
    T = w.size
    step = np.zeros(T)
    i1, i2 = seg[:, 0], seg[:, 1]
    starts = i1[i1 > 0]
    ends = i2[i2 < T - 1]
    np.add.at(step, starts, w[starts - 1] - w[starts])
    np.add.at(step, ends + 1, w[ends] - w[ends + 1])
    return w + np.cumsum(step)
