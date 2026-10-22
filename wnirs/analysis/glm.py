"""Subject-level general linear model on one recording."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ..types import Hb
from ._stats import EPS, round_half_away, t_pvalue
from .ar import arfit, whiten
from .arirls import ar_irls
from .design import design_matrix
from .hrf import HRFKind
from .robust import BISQUARE_TUNE

GLMMethod = Literal["arirls", "ar", "ols"]
CHROMOPHORES = ("HbO", "HbR")


@dataclass
class ChannelFits:
    """Per-channel estimates for one chromophore; NaN where a channel was skipped."""
    beta: np.ndarray        # (p, n_ch)
    se: np.ndarray          # (p, n_ch)
    tstat: np.ndarray       # (p, n_ch)
    pval: np.ndarray        # (p, n_ch)
    cov: np.ndarray         # (p, p, n_ch)
    r2: np.ndarray          # (n_ch,)
    dfe: np.ndarray         # (n_ch,)
    ar_order: np.ndarray    # (n_ch,), NaN unless method is arirls
    weights: np.ndarray     # (T, n_ch) robust weight per sample, NaN unless arirls

    @classmethod
    def empty(cls, T: int, p: int, n_ch: int) -> "ChannelFits":
        shapes = dict(beta=(p, n_ch), se=(p, n_ch), tstat=(p, n_ch), pval=(p, n_ch), cov=(p, p, n_ch),
                      r2=(n_ch,), dfe=(n_ch,), ar_order=(n_ch,), weights=(T, n_ch))
        return cls(**{name: np.full(shape, np.nan) for name, shape in shapes.items()})


@dataclass
class GLMResult:
    """GLM estimates for every channel of one recording, HbO and HbR.

    Arrays carry a trailing chromophore axis of length 2 (0 = HbO, 1 = HbR).
    ``dfe`` is stored per channel: AR-IRLS drops the filter start-up, so the
    error degrees of freedom depend on each channel's AR order.
    """
    subject: str
    task: str
    conditions: list[str]
    names: list[str]
    n_condition_regressors: int
    layout: pd.DataFrame
    beta: np.ndarray        # (p, n_ch, 2)
    se: np.ndarray
    tstat: np.ndarray
    pval: np.ndarray
    cov: np.ndarray         # (p, p, n_ch, 2)
    r2: np.ndarray          # (n_ch, 2)
    dfe: np.ndarray         # (n_ch, 2)
    X: np.ndarray           # the unwhitened design that was fitted
    fs: float
    method: str
    ar_order: np.ndarray    # (n_ch, 2)
    weights: np.ndarray     # (T, n_ch, 2)
    options: dict = field(default_factory=dict)


def fit_channels(X: np.ndarray, Y: np.ndarray, *, fs: float, method: GLMMethod = "arirls",
                 ar: int | None = None, tune: float = BISQUARE_TUNE) -> ChannelFits:
    """Fit the same design to every column of ``Y`` (T x n_channels).

    Channels with any non-finite sample are skipped and left NaN; a channel
    excluded by quality control arrives that way.
    """
    method = method.lower()
    T, n_ch = Y.shape
    p = X.shape[1]
    out = ChannelFits.empty(T, p, n_ch)
    for ch in range(n_ch):
        y = Y[:, ch]
        if not np.all(np.isfinite(y)):
            continue
        if method == "arirls":
            s = ar_irls(X, y, fs=fs, pmax=ar, tune=tune)
            if not np.isfinite(s.dfe):
                continue
            out.beta[:, ch], out.se[:, ch], out.tstat[:, ch], out.pval[:, ch] = s.beta, s.se, s.tstat, s.pval
            out.cov[:, :, ch], out.r2[ch], out.dfe[ch] = s.covb, s.r2, s.dfe
            out.ar_order[ch], out.weights[:, ch] = s.order, s.w
        else:
            order = 0 if method == "ols" else (max(1, round_half_away(4 * fs)) if ar is None else ar)
            b, se, t, pv, V, r2, dfe = _fit_prewhitened(X, y, order)
            out.beta[:, ch], out.se[:, ch], out.tstat[:, ch], out.pval[:, ch] = b, se, t, pv
            out.cov[:, :, ch], out.r2[ch], out.dfe[ch] = V, r2, dfe
    return out


def glm(hb: Hb, *, method: GLMMethod = "arirls", hrf: HRFKind = "canonical", derivative: bool = False,
        drift: int = 1, ar: int | None = None, tune: float = BISQUARE_TUNE,
        short_regressor: bool = False) -> GLMResult:
    """Fit y = X beta + e per channel, separately for HbO and HbR.

    X holds one HRF-convolved boxcar per condition plus the nuisance terms
    asked for. Serial correlation in fNIRS residuals inflates t statistics
    badly if ignored, so by default the model is fitted by AR-IRLS.

    Parameters
    ----------
    hb : Hb
        Preprocessed recording. The analysis copies (``hbo``/``hbr``) are used,
        never the low-passed display copies.
    method : {'arirls', 'ar', 'ols'}
        ``arirls``: AR whitening plus robust IRLS (Barker, Aarabi & Huppert
        2013). Handles autocorrelation and motion; the recommended choice and
        what nirs-toolbox uses.
        ``ar``: AR prewhitening then ordinary least squares. Fixes
        autocorrelation but not outliers.
        ``ols``: plain least squares. Its t values will be inflated several-fold
        because fNIRS residuals are strongly autocorrelated; use it to compare
        against, not to report.
    hrf : {'canonical', 'gamma'}
    derivative : bool
        Add a temporal derivative regressor per condition.
    drift : int
        Polynomial drift order; 0 = intercept only.
    ar : int, optional
        AR order. For ``arirls`` the maximum order searched (chosen per channel
        by BIC). Default 4 x fs, the published guidance.
    tune : float
        Bisquare constant for ``arirls``.
    short_regressor : bool
        Add the mean short channel as a nuisance regressor. Leave False if
        short-separation regression already ran in preprocessing.
    """
    method = method.lower()
    if method not in ("arirls", "ar", "ols"):
        raise ValueError(f'Unknown GLM method "{method}". Use arirls, ar, or ols.')
    label = f"{hb.subject} / {hb.task}"
    _warn_if_lowpassed(hb, method)

    short = None
    if short_regressor:
        is_short = hb.layout["Type"].to_numpy() == 2
        if is_short.any():
            short = np.nanmean(hb.hbo[:, is_short], axis=1)
        else:
            warnings.warn("short_regressor requested but this layout has no short channels.", stacklevel=2)
    design = design_matrix(hb.time, hb.conditions, hb.fs, hrf_kind=hrf, derivative=derivative,
                           drift=drift, short_regressor=short, label=label)

    fits = [fit_channels(design.X, Y, fs=hb.fs, method=method, ar=ar, tune=tune) for Y in (hb.hbo, hb.hbr)]
    per_chromophore = {name: np.stack([getattr(f, name) for f in fits], axis=-1) for name in vars(fits[0])}
    return GLMResult(
        subject=hb.subject, task=hb.task, conditions=list(map(str, hb.conditions.name)),
        names=design.names, n_condition_regressors=design.n_condition_regressors, layout=hb.layout,
        X=design.X, fs=hb.fs, method=method, **per_chromophore,
        options=dict(method=method, hrf=hrf, derivative=derivative, drift=drift, ar=ar, tune=tune,
                     short_regressor=short_regressor),
    )


def _fit_prewhitened(X: np.ndarray, y: np.ndarray, ar_order: int):
    """OLS, then (if ar_order > 0) a refit on AR-whitened data.

    Whitening both sides with the same filter leaves beta unbiased while giving
    standard errors that account for the residual autocorrelation. The order is
    chosen by BIC up to ``ar_order`` rather than fixed at it: a fixed high order
    over-whitens, the AR model starts absorbing the haemodynamic response
    itself and the effect disappears into the innovations.
    """
    yw, Xw = y, X
    if ar_order > 0 and y.size > 5 * ar_order:
        r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        a = arfit(r, ar_order)[0]
        yw, Xw = whiten(a, y)[a.size:], whiten(a, X)[a.size:]

    n, k = Xw.shape
    dfe = max(n - k, 1)
    Q, R = np.linalg.qr(Xw)
    RtR = R.T @ R
    if 1.0 / np.linalg.cond(RtR, 1) < 1e-12:
        b = np.linalg.pinv(Xw) @ yw
        Rinv = np.linalg.pinv(Xw.T @ Xw)
    else:
        b = np.linalg.solve(R, Q.T @ yw)
        Rinv = np.linalg.inv(RtR)
    res = yw - Xw @ b
    V = (res @ res) / dfe * Rinv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    t = b / np.maximum(se, EPS)
    r2 = 1 - (res @ res) / max(np.sum((yw - yw.mean()) ** 2), EPS)
    return b, se, t, t_pvalue(t, dfe), V, r2, float(dfe)


def _warn_if_lowpassed(hb: Hb, method: str) -> None:
    """AR whitening flattens the residual spectrum, so it needs the spectrum intact."""
    band = hb.info.get("band", hb.meta.get("band"))
    if method == "ols" or band is None or np.size(band) != 2 or not np.isfinite(band[1]):
        return
    warnings.warn(
        f"This data was low-pass filtered at {band[1]:g} Hz and the {method!r} estimator whitens the residual "
        "spectrum. With part of the spectrum missing the AR coefficients are unstable. Either turn the low-pass "
        "off and let the drift regressors handle slow terms, or use method 'ols' and accept that its t values "
        "are optimistic.", stacklevel=3)
