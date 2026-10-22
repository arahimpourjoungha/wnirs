"""Brain-behaviour association: across subjects per channel, and within subjects (rmcorr)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ._stats import correlate as corr_pair
from ._stats import t_pvalue, unique_stable
from .fdr import fdr_bh


def channel_behaviour(tables: Sequence[pd.DataFrame], y, *, kind: str = "spearman",
                      covariates: np.ndarray | None = None, n_perm: int = 0, name: str = "behaviour",
                      rng: np.random.Generator | int | None = None) -> pd.DataFrame:
    """Correlate each channel's subject-level beta with a behavioural measure.

    With small n these correlations are unstable whatever the p value says.
    Treat a channel-wise brain-behaviour map from under about 20 subjects as
    exploratory, and say so in the write-up.

    Parameters
    ----------
    tables : sequence of DataFrame
        One contrast table per subject (from :func:`~wnirs.analysis.contrast`).
        Rows are matched to the first table by Label and Chromophore.
    y : array_like
        One behavioural value per subject, same order.
    kind : {'spearman', 'pearson', 'kendall'}
    covariates : (n_subjects, n_cov) array, optional
        Regressed out of both sides, so rho is a partial correlation.
    n_perm : int
        Use a permutation p value with this many shuffles instead of the
        parametric one. Below about 15 subjects it is the honest choice.
    """
    rng = np.random.default_rng(rng)
    y = np.asarray(y, float).ravel()
    n = len(tables)
    if y.size != n:
        raise ValueError(f"y has {y.size} entries but there are {n} subjects.")
    ref = tables[0]
    key = ref["Label"].astype(str) + "|" + ref["Chromophore"].astype(str)
    B = np.column_stack([
        (t.assign(_k=t["Label"].astype(str) + "|" + t["Chromophore"].astype(str))
          .drop_duplicates("_k").set_index("_k")["Beta"].reindex(key).to_numpy(float))
        for t in tables])
    Z = None
    if covariates is not None:
        covariates = np.asarray(covariates, float).reshape(n, -1)
        Z = np.column_stack((np.ones(n), covariates))

    rho, pv, N = np.full(len(ref), np.nan), np.full(len(ref), np.nan), np.zeros(len(ref), int)
    for r in range(len(ref)):
        ok = np.isfinite(B[r]) & np.isfinite(y)
        if Z is not None:
            ok &= np.all(np.isfinite(Z), axis=1)
        N[r] = ok.sum()
        if N[r] < 4:
            continue
        xr, yr = B[r, ok], y[ok]
        if Z is not None:
            xr, yr = _residualise(xr, Z[ok]), _residualise(yr, Z[ok])
        rho[r], pv[r] = corr_pair(xr, yr, kind)
        if n_perm > 0:
            null = np.array([corr_pair(rng.permutation(xr), yr, kind)[0] for _ in range(n_perm)])
            # +1 so a finite permutation test can never return exactly zero.
            pv[r] = (np.sum(np.abs(null) >= abs(rho[r])) + 1) / (n_perm + 1)

    chromo = ref["Chromophore"].astype(str).to_numpy()
    q = np.full(len(ref), np.nan)
    for c in unique_stable(chromo):
        q[chromo == c] = fdr_bh(pv[chromo == c])
    out = pd.DataFrame({"Channel": ref["Channel"].to_numpy(), "Source": ref["Source"].to_numpy(),
                        "Detector": ref["Detector"].to_numpy(), "Label": ref["Label"].to_numpy(),
                        "Chromophore": chromo, "N": N, "Rho": rho, "Pval": pv, "Qval": q})
    out.attrs = dict(type=kind, name=name, n_perm=n_perm, has_covariates=covariates is not None)
    return out


def _residualise(v: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]


@dataclass
class RMCorr:
    """Repeated-measures correlation; ``df = n - n_subjects - 1``."""
    r: float
    p: float
    df: int
    n: int
    n_subjects: int


def rmcorr(x, y, subject) -> RMCorr:
    """Repeated-measures correlation (Bakdash & Marusich 2017).

    Each person's series is centred on their own means, so what is correlated
    is session-to-session covariation WITHIN people: a person with higher
    values on both measures overall contributes nothing. Pooling the raw values
    instead would let a few between-person differences dominate. The test has
    n - k - 1 degrees of freedom for n observations from k people, one slope
    shared by everyone and one intercept per person.

    Parameters
    ----------
    x, y : array_like
        Paired observations; pairs with a non-finite value are dropped.
    subject : array_like
        Who each observation belongs to.
    """
    x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(subject)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], g[ok]
    codes = np.unique(g, return_inverse=True)[1]
    k = int(codes.max()) + 1 if codes.size else 0
    counts = np.bincount(codes, minlength=k)
    xc = x - (np.bincount(codes, x, k) / counts)[codes]
    yc = y - (np.bincount(codes, y, k) / counts)[codes]
    n, df = x.size, x.size - k - 1
    if k < 1 or df < 1:
        return RMCorr(np.nan, np.nan, df, n, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(np.clip(xc @ yc / np.sqrt((xc @ xc) * (yc @ yc)), -1, 1))
        t = r * np.sqrt(df / max(1 - r * r, np.finfo(float).eps))
    return RMCorr(r, float(t_pvalue(t, df)), df, n, k)
