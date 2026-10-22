"""Small statistical primitives shared across the analysis modules.

Several of these exist only because the NumPy/SciPy default would change a
published number: rounding (half away from zero, not half to even),
percentiles (Hazen plotting positions, not linear), and the p value of
Spearman's rho (exact / Edgeworth / Student depending on n and ties, where
SciPy always uses Student). Pinning them is what makes the outputs match
the reference values exactly rather than merely resemble them.
"""
from __future__ import annotations

from itertools import permutations
from math import factorial

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erfc

EPS = np.finfo(float).eps


def round_half_away(x):
    """Round half away from zero (NumPy rounds halves to even)."""
    x = np.asarray(x, dtype=float)
    out = np.sign(x) * np.floor(np.abs(x) + 0.5)
    return int(out) if out.ndim == 0 else out.astype(int)


def percentile(x, q, axis=None):
    """Percentiles with Hazen plotting positions, NaNs ignored."""
    return np.nanpercentile(x, q, axis=axis, method="hazen")


def t_pvalue(t, df):
    """Two-sided p value of a t statistic."""
    return 2.0 * stats.t.cdf(-np.abs(t), df)


def channel_labels(layout: pd.DataFrame) -> np.ndarray:
    """Channel labels as the layout builds them: 'S1-D10', shorts '(sc)'."""
    if "Label" in layout:
        return layout["Label"].astype(str).to_numpy()
    lab = "S" + layout["S"].astype(int).astype(str) + "-D" + layout["D"].astype(int).astype(str)
    if "Type" in layout:
        lab = lab.where(layout["Type"].to_numpy() != 2, lab + " (sc)")
    return lab.to_numpy()


def unique_stable(values) -> np.ndarray:
    """Unique values in order of first appearance."""
    values = np.asarray(values)
    _, first = np.unique(values, return_index=True)
    return values[np.sort(first)]


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
def pairwise_corr(X: np.ndarray) -> np.ndarray:
    """Pearson correlation of the columns of X using pairwise-complete rows.

    Columns are centred on their own means first so that the expanded-sum
    formula below does not lose precision, and every pair then uses only the
    rows both have.
    """
    X = np.asarray(X, dtype=float)
    M = np.isfinite(X)
    Xc = np.where(M, X - nanmean(X, axis=0), 0.0)
    Mf = M.astype(float)
    n = Mf.T @ Mf
    sx = Xc.T @ Mf                       # sx[i, j]: sum of x_i over rows shared with j
    sxx = (Xc**2).T @ Mf
    sxy = Xc.T @ Xc
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sx.T / n
        var_i = sxx - sx**2 / n
        r = cov / np.sqrt(var_i * var_i.T)
    r[n < 2] = np.nan
    return np.clip(r, -1.0, 1.0)


def corr_matrix(X: np.ndarray, kind: str = "pearson") -> np.ndarray:
    """Pearson or Spearman correlation of the columns of X, pairwise-complete."""
    kind = kind.lower()
    if kind == "pearson":
        return pairwise_corr(X)
    if kind != "spearman":
        raise ValueError(f"Unknown correlation type {kind!r}; use pearson or spearman.")
    if np.all(np.isfinite(X)):
        return pairwise_corr(stats.rankdata(X, axis=0))
    # Pairwise Spearman ranks each pair on the rows both columns have.
    n = X.shape[1]
    R = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            ok = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
            if ok.sum() >= 2:
                R[i, j] = R[j, i] = pairwise_corr(stats.rankdata(X[ok][:, [i, j]], axis=0))[0, 1]
    return R


def nanmean(x: np.ndarray, axis=None) -> np.ndarray:
    """Mean ignoring NaN, NaN (without a warning) where nothing is finite."""
    ok = np.isfinite(x)
    n = ok.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.where(ok, x, 0.0).sum(axis=axis) / n, np.nan)


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r with its t-distribution p value."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    r = float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))
    with np.errstate(divide="ignore"):
        t = r * np.sqrt((n - 2) / max(1.0 - r * r, 0.0))
    return r, float(t_pvalue(t, n - 2))


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rho and its p value.

    The exact permutation distribution is used below n = 10, the AS 89
    Edgeworth approximation for larger untied samples, and a Student t on rho
    when there are ties. SciPy uses the t approximation throughout, which can
    move a p value near 0.05 by enough to change a reported q.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n < 2:
        return np.nan, np.nan
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return np.nan, np.nan
    rho = float(np.clip(np.corrcoef(rx, ry)[0, 1], -1.0, 1.0))
    ties = np.unique(x).size < n or np.unique(y).size < n
    n3 = (n**3 - n) / 3.0
    if n < 10:
        d = np.sum((rx - ry) ** 2)
        if not ties:
            d = n3 / 2.0 - round_half_away(rho * n3 / 2.0)
        return rho, _spearman_exact(rx, ry, d)
    if ties:
        with np.errstate(divide="ignore"):
            t = np.inf * np.sign(rho) if abs(rho) >= 1 else rho * np.sqrt((n - 2) / (1 - rho**2))
        return rho, float(t_pvalue(t, n - 2))
    d = n3 / 2.0 - round_half_away(rho * n3 / 2.0)
    return rho, float(min(2.0 * _as89(max(d, n3 - d), n, n3), 1.0))


def _spearman_exact(rx: np.ndarray, ry: np.ndarray, d: float) -> float:
    """Two-sided exact p for the rank-difference statistic D over all n! orders."""
    n = rx.size
    perms = np.array(list(permutations(ry)))
    dperm = np.sum((rx[None, :] - perms) ** 2, axis=1)
    n3 = (n**3 - n) / 3.0
    edges = np.linspace(-0.25, n3 + 0.25, int(round(2 * n3)) + 2)   # bins centred on half-integers
    freq, _ = np.histogram(dperm, bins=edges)
    left = np.cumsum(freq)
    right = factorial(n) - np.concatenate(([0], left[:-1]))
    tail = np.minimum(np.minimum(left, right) / factorial(n) * 2.0, 1.0)
    return float(tail[int(round(2 * d))])


def _as89(d: float, n: int, n3: float) -> float:
    """Upper tail of Spearman's D, algorithm AS 89 (Best & Roberts 1975)."""
    c = [0.2274, 0.2531, 0.1745, 0.0758, 0.1033, 0.3932, 0.0879, 0.0151, 0.0072, 0.0831, 0.0131, 0.00046]
    x = (2 * (d - 1) / n3 - 1) * np.sqrt(n - 1)
    y = x * x
    u = x * (c[0] + (c[1] + c[2] / n) / n
             + y * (-c[3] + (c[4] + c[5] / n) / n
                    - y * (c[6] + c[7] / n
                           - y * (c[8] - c[9] / n + y * (c[10] - c[11] * y) / n)) / n)) / n
    p = u / np.exp(0.5 * y) + 0.5 * erfc(x / np.sqrt(2))
    return float(np.clip(p, 0.0, 1.0))


def correlate(x: np.ndarray, y: np.ndarray, kind: str = "pearson") -> tuple[float, float]:
    """Dispatch to :func:`pearson` or :func:`spearman` by name."""
    kind = kind.lower()
    if kind == "pearson":
        return pearson(x, y)
    if kind == "spearman":
        return spearman(x, y)
    if kind == "kendall":
        res = stats.kendalltau(x, y)
        return float(res.statistic), float(res.pvalue)
    raise ValueError(f"Unknown correlation type {kind!r}; use pearson, spearman or kendall.")


def is_outlier_mean(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Outliers by the mean rule: more than three standard deviations out."""
    mu = np.nanmean(x, axis=axis, keepdims=True)
    sd = np.nanstd(x, axis=axis, ddof=1, keepdims=True)
    return np.abs(x - mu) > 3.0 * sd
