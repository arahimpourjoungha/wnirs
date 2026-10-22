"""Baseline (DC) shift removal by robust autoregressive fitting."""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal

from ._util import as_columns, fill_gaps, mround

_TUKEY = 4.685
_MAD_TO_SD = 0.6745


def baseline_shift(x: np.ndarray, fs: float, method: Literal["ari", "step"] = "ari",
                   tune: float = _TUKEY, pmax: int | None = None) -> tuple[np.ndarray, dict]:
    """Remove step shifts in optical density.

    Parameters
    ----------
    x : numpy.ndarray
        ``(T,)`` or ``(T, C)`` optical density.
    fs : float
        Sampling rate in Hz.
    method : {"ari", "step"}
        ``"ari"`` is the nirs-toolbox ``BaselineCorrection`` algorithm;
        ``"step"`` thresholds the first difference.
    tune : float
        Outlier tuning constant, in robust SDs.
    pmax : int, optional
        Maximum AR order for ``"ari"``; default ``round(4 * fs)``, the
        published guidance.

    Returns
    -------
    y : numpy.ndarray
        Corrected signal, same shape as ``x``.
    info : dict
        ``n_shifts`` (steps removed per channel, ``"step"`` only),
        ``mean_weight`` (mean robust weight per channel, ``"ari"``; lower
        means more of the record was treated as artefact) and ``ar_order``.

    Notes
    -----
    A baseline shift is a step: an optode moves, the coupling changes, and the
    trace settles at a new level. No frequency filter removes that -- a step
    contains every frequency, and high-passing it leaves a decaying tail right
    where the response is expected. Despite the usual name this regresses
    nothing against a baseline period; it is a conservative motion correction
    run on OD before Beer-Lambert.

    ``"ari"`` differences the signal (a step becomes a spike), fits an AR
    model to the difference, derives robust weights from its residual, refits
    on the weighted difference, then applies differencing and AR whitening
    together and reweights in that domain before integrating back through the
    inverse filter. Reweighting after whitening is what distinguishes it from
    TDDR: a large excursion consistent with the channel's own autocorrelation
    is kept, a genuine discontinuity is removed.

    ``"step"`` takes no account of autocorrelation and so needs a higher
    threshold to avoid removing real signal, but is easy to check by eye.
    """
    if method not in ("ari", "step"):
        raise ValueError(f'unknown method "{method}"; use "ari" or "step"')
    if pmax is None:
        pmax = max(1, mround(4 * fs))

    x2, restore = as_columns(x)
    C = x2.shape[1]
    y = x2.copy()
    info = {"method": method, "n_shifts": np.zeros(C, dtype=int),
            "mean_weight": np.full(C, np.nan), "ar_order": np.full(C, np.nan)}

    for c in range(C):
        if np.isfinite(x2[:, c]).sum() < 20:
            continue
        v = fill_gaps(x2[:, c])
        if method == "ari":
            y[:, c], info["mean_weight"][c], info["ar_order"][c] = _ari(v, tune, pmax)
        else:
            y[:, c], info["n_shifts"][c] = _step(v, tune)
    return restore(y), info


def _ari(v: np.ndarray, tune: float, pmax: int) -> tuple[np.ndarray, float, float]:
    """Robust ARI(p, 1) correction of one trace."""
    yd = np.diff(v, prepend=0.0)
    if not yd.any():
        return v, np.nan, np.nan

    a0 = arfit(yd, pmax)
    w = _robust_weight(signal.lfilter(a0, 1.0, yd), tune)
    a = arfit(w * yd, pmax)

    # Differencing and whitening together ask whether an excursion is
    # surprising given this channel's autocorrelation, not merely large.
    f = np.convolve([1.0, -1.0], a)
    yf = signal.lfilter(f, 1.0, v)
    w = _robust_weight(yf, tune)
    with np.errstate(over="ignore", invalid="ignore"):
        rebuilt = signal.lfilter([1.0], f, w * yf)
    if not np.isfinite(rebuilt).all():
        return v, float(w.mean()), float(a.size - 1)     # inverse filter went unstable
    # The reconstruction is defined up to a constant; restore the original level.
    return rebuilt - rebuilt.mean() + v.mean(), float(w.mean()), float(a.size - 1)


def _robust_weight(r: np.ndarray, tune: float) -> np.ndarray:
    """Weights ``(1 - z^2)`` with a separate robust scale for each tail.

    Motion artefacts in OD are asymmetric -- an optode lifting off and
    settling back give excursions of different size -- so scaling each tail by
    its own spread stops the larger one setting the threshold for both.
    """
    r = r - np.nanmedian(r)
    z = np.zeros_like(r)
    for tail in (r < 0, r > 0):
        s = _mad_scale(r[tail])
        if s > 0:
            z[tail] = r[tail] / (s * tune)
    w = (1 - z ** 2) * (np.abs(z) < 1)
    w[~np.isfinite(w)] = 0.0
    return w


def _mad_scale(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    return float(np.median(np.abs(v - np.median(v))) / _MAD_TO_SD)


def _step(v: np.ndarray, tune: float) -> tuple[np.ndarray, int]:
    d = np.diff(v)
    sd = 1.4826 * np.median(np.abs(d - np.median(d)))
    if sd <= 0:
        return v, 0
    jump = np.where(np.abs(d) > tune * sd, d, 0.0)
    if not jump.any():
        return v, 0
    # The offset at each sample is everything that has jumped so far.
    return v - np.concatenate([[0.0], np.cumsum(jump)]), int(np.count_nonzero(jump))


def arfit(x: np.ndarray, pmax: int = 20, criterion: Literal["bic", "aic"] = "bic") -> np.ndarray:
    """Autoregressive whitening filter with the order chosen by BIC or AIC.

    Parameters
    ----------
    x : numpy.ndarray
        1-D series; non-finite samples are dropped.
    pmax : int
        Largest order considered (also capped at ``len(x) // 4``).
    criterion : {"bic", "aic"}
        BIC penalises extra parameters more and so errs towards the lower
        order, the safer error: too high an order starts fitting the signal.

    Returns
    -------
    numpy.ndarray
        ``[1, -phi_1, ..., -phi_p]``, usable directly as
        ``lfilter(a, 1, x)``; ``[1.0]`` when no AR structure is found.

    Notes
    -----
    Levinson-Durbin on the biased autocovariance. Dividing by ``n`` rather
    than ``n - k`` keeps the sequence positive semi-definite, which guarantees
    a stable filter; an unstable whitening filter would amplify rather than
    remove autocorrelation.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    pmax = min(pmax, n // 4)
    if n < 10 or pmax < 1:
        return np.ones(1)
    x = x - x.mean()
    g = np.array([x[:n - k] @ x[k:] for k in range(pmax + 1)]) / n
    if g[0] <= 0:
        return np.ones(1)

    best_score, best = np.inf, np.zeros(0)
    phi = np.zeros(0)
    err = g[0]
    for p in range(1, pmax + 1):
        k = (g[p] - phi @ g[p - 1:0:-1]) / err
        if not np.isfinite(k) or abs(k) >= 1:
            break
        phi = np.concatenate([phi - k * phi[::-1], [k]])
        err *= 1 - k ** 2
        if err <= 0:
            break
        penalty = 2 * p if criterion == "aic" else p * np.log(n)
        score = n * np.log(err) + penalty
        if score < best_score:
            best_score, best = score, phi
    return np.concatenate([[1.0], -best])
