"""Autoregressive model fitting for prewhitening."""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import lfilter


def autocovariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased autocovariance g[0..max_lag] of an already-centred series.

    Dividing by n rather than n - k keeps the sequence positive semi-definite,
    which is what guarantees a stable Levinson-Durbin recursion.
    """
    n = x.size
    return np.array([x[: n - k] @ x[k:] for k in range(max_lag + 1)]) / n


def arfit(x: np.ndarray, pmax: int = 20, criterion: Literal["bic", "aic"] = "bic") -> tuple[np.ndarray, int]:
    """Autoregressive fit with the model order chosen by an information criterion.

    The order is chosen rather than fixed. Too low leaves autocorrelation in
    the residual, which is the problem prewhitening exists to solve; too high
    starts fitting the signal itself and eats the response. BIC penalises
    extra parameters more than AIC, so it errs towards the lower order, which
    is the safer error here.

    Coefficients come from the Levinson-Durbin recursion on the biased
    autocovariance, which is guaranteed to give a stable filter -- an unstable
    whitening filter would amplify rather than remove the autocorrelation.

    Parameters
    ----------
    x : array_like
        Series to model; non-finite samples are dropped.
    pmax : int
        Largest order considered (capped at n/4).
    criterion : {'bic', 'aic'}

    Returns
    -------
    a : numpy.ndarray
        Whitening filter ``[1, -phi_1, ..., -phi_p]``: ``lfilter(a, 1, x)``
        removes the modelled autocorrelation. ``[1]`` when no order helps.
    order : int
        The selected order p (0 for none).
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    none = np.array([1.0]), 0
    if n < 10:
        return none
    pmax = min(int(pmax), n // 4)
    if pmax < 1:
        return none
    g = autocovariance(x - x.mean(), pmax)
    if g[0] <= 0:
        return none

    penalty = 2.0 if criterion.lower() == "aic" else np.log(n)
    best_score, best_a, best_order = np.inf, None, 0
    a = np.array([1.0])          # accumulates +phi: a[j] is phi_j
    err = g[0]
    for p in range(1, pmax + 1):
        k = (g[p] - a[1:p] @ g[p - 1:0:-1]) / err
        if not np.isfinite(k) or abs(k) >= 1:
            break
        a = np.concatenate((a[:1], a[1:p] - k * a[p - 1:0:-1], [k]))
        err *= 1.0 - k * k
        if err <= 0:
            break
        score = n * np.log(err) + p * penalty
        if score < best_score:
            best_score, best_a, best_order = score, a, p
    if best_order == 0:
        return none
    return np.concatenate(([1.0], -best_a[1:])), best_order


def whiten(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply the whitening filter ``a`` along the first axis (``lfilter(a, 1, x)``)."""
    return lfilter(a, [1.0], x, axis=0)
