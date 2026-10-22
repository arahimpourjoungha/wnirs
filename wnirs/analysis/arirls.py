"""Autoregressive iteratively reweighted least squares (Barker, Aarabi & Huppert 2013)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d

from ._stats import EPS, round_half_away, t_pvalue
from .ar import arfit, whiten
from .robust import BISQUARE_TUNE, RobustFit, robust_fit


@dataclass
class ARIRLSFit:
    """One channel's AR-IRLS fit. All fields are NaN when the fit was refused."""
    beta: np.ndarray
    covb: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    pval: np.ndarray
    dfe: float
    order: float
    a: np.ndarray
    w: np.ndarray
    sigma2: float
    r2: float
    converged: bool = field(default=False)

    @classmethod
    def empty(cls, n: int, p: int) -> "ARIRLSFit":
        nan_p = np.full(p, np.nan)
        return cls(nan_p, np.full((p, p), np.nan), nan_p.copy(), nan_p.copy(), nan_p.copy(),
                   np.nan, np.nan, np.array([]), np.full(n, np.nan), np.nan, np.nan)


def default_pmax(fs: float | None) -> int:
    """Largest AR order searched: 4x the sampling rate, the published guidance."""
    return max(1, round_half_away(4 * fs)) if fs else 20


def ar_irls(X: np.ndarray, y: np.ndarray, *, fs: float | None = None, pmax: int | None = None,
            tune: float = BISQUARE_TUNE, max_iter: int = 10) -> ARIRLSFit:
    """Fit one channel by AR prewhitening plus robust bisquare regression.

    fNIRS residuals are heavily autocorrelated -- systemic physiology is not
    white noise -- and ordinary least squares standard errors assume they are,
    which inflates t several-fold. AR-IRLS alternates two steps until the
    coefficients settle:

    1. fit an AR model (order by BIC, up to ``pmax``) to the current residual
       and use it to whiten both y and X, which fixes the autocorrelation;
    2. refit by robust IRLS with a Tukey bisquare weight, which stops motion
       artefacts dominating the fit.

    Whitening alone leaves the fit sensitive to artefacts; robust fitting alone
    leaves the standard errors wrong. Do not low-pass filter data before this:
    whitening flattens the residual spectrum and needs all of it.

    Parameters
    ----------
    X : (T, p) array
        Design matrix.
    y : (T,) array
        One channel. Short non-finite gaps are interpolated for the AR filter
        but excluded from the fit.
    fs : float, optional
        Sampling rate, used only to choose ``pmax`` (= round(4 fs)).
    pmax : int, optional
        Largest AR order. Default 4 fs, or 20 without ``fs``.
    tune : float
        Bisquare tuning constant.
    max_iter : int
        Outer (whiten-refit) iterations.

    Returns
    -------
    ARIRLSFit
        ``w`` holds the robust weight per sample (NaN for samples the fit did
        not use); ``order`` is the AR order of the final whitening filter.

    Notes
    -----
    Convergence compares each new estimate with the one from *two* iterations
    earlier; the reference outputs the fit is tested against depend on this
    rule, so it is kept.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    T, p = X.shape
    pmax = default_pmax(fs) if pmax is None else pmax
    out = ARIRLSFit.empty(T, p)

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if ok.sum() < max(10, 2 * p) or not np.any(y[ok] != y[ok][0]):
        return out
    yy = y.copy()
    if not ok.all():
        idx = np.arange(T)
        yy[~ok] = interp1d(idx[ok], y[ok], fill_value="extrapolate")(idx[~ok])

    B = np.linalg.lstsq(X[ok], yy[ok], rcond=None)[0]
    B_prev = np.full(p, np.inf)
    fit: RobustFit | None = None
    a, use, converged = np.array([1.0]), ok, False
    for _ in range(max_iter):
        a_new = arfit((yy - X @ B)[ok], pmax)[0]
        use_new = ok.copy()
        use_new[: a_new.size - 1] = False          # filter start-up depends on assumed zeros
        if use_new.sum() < max(10, 2 * p):
            break
        a, use = a_new, use_new
        fit = robust_fit(whiten(a, X)[use], whiten(a, yy)[use], tune)
        B_new = fit.beta
        with np.errstate(invalid="ignore"):
            converged = np.linalg.norm(B_new - B_prev) / max(np.linalg.norm(B_prev), EPS) < 1e-2
        B_prev, B = B, B_new
        if converged:
            break
    if fit is None:
        return out

    se = np.sqrt(np.maximum(np.diag(fit.covb), 0))
    tstat = B / np.maximum(se, EPS)
    w = np.full(T, np.nan)
    w[use] = fit.w
    return ARIRLSFit(B, fit.covb, se, tstat, t_pvalue(tstat, max(fit.dfe, 1)), fit.dfe,
                     float(a.size - 1), a, w, fit.sigma2, fit.r2, bool(converged))
