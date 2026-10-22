"""Robust regression with Tukey's bisquare, as used inside AR-IRLS."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr, solve_triangular

from ._stats import EPS

#: Bisquare constant giving 95% efficiency on clean Gaussian data.
BISQUARE_TUNE = 4.685


@dataclass
class RobustFit:
    """Result of :func:`robust_fit` for one response."""
    beta: np.ndarray
    covb: np.ndarray
    w: np.ndarray
    dfe: float
    sigma2: float
    r2: float
    iterations: int


def bisquare(u: np.ndarray) -> np.ndarray:
    """Tukey bisquare weight: near 1 inside the window, exactly 0 beyond it."""
    return (np.abs(u) < 1) * (1.0 - u**2) ** 2


def robust_fit(X: np.ndarray, y: np.ndarray, tune: float = BISQUARE_TUNE, max_iter: int = 50) -> RobustFit:
    """Bisquare M-estimate by iteratively reweighted least squares.

    Each iteration reweights samples by how far their residual sits from the
    robust scale. Samples inside the tuning window are weighted near 1 and those
    beyond it get zero weight, so a motion artefact stops influencing the fit
    rather than dragging it.

    Getting the robust *covariance* right is the hard part. A bisquare weight
    gives the largest residuals zero weight, so the naive weighted residual
    variance underestimates sigma and inflates t. The estimate therefore uses
    leverage-adjusted residuals (DuMouchel & O'Brien 1989), the MAD scale
    omitting the p - 1 smallest residuals, the Street/Huber K correction for
    the covariance, and shrinkage of the robust sigma towards the OLS sigma;
    no constant is added to X. A null simulation with this estimator controls
    the false-positive rate near 5% where a simpler one ran above 50%.

    Parameters
    ----------
    X : (n, p) array
        Design matrix; include an intercept column yourself.
    y : (n,) array
    tune : float
        Bisquare tuning constant. 4.685 is 95% efficient on Gaussian data;
        lower is more robust to outliers at a cost when there are none.
    max_iter : int
        Iteration limit.

    Returns
    -------
    RobustFit
        ``dfe`` is n - rank(X); ``sigma2`` the squared robust scale; ``r2`` is
        computed with the robust weights.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, p = X.shape
    if n <= p:
        return RobustFit(np.full(p, np.nan), np.full((p, p), np.nan), np.ones(n), max(n - p, 1), np.nan, np.nan, 0)

    Q, R, perm = qr(X, mode="economic", pivoting=True)
    tol = abs(R[0, 0]) * max(n, p) * EPS
    rank = int(np.sum(np.abs(np.diag(R)) > tol))
    perm = perm[:rank]
    Rr = R[:rank, :rank]
    b = np.zeros(p)
    b[perm] = solve_triangular(Rr, Q[:, :rank].T @ y)

    E = solve_triangular(Rr, X[:, perm].T, trans="T").T
    h = np.minimum(0.9999, np.sum(E * E, axis=1))
    adj = 1.0 / np.sqrt(1.0 - h)
    dfe = n - rank
    ols_s = np.linalg.norm(y - X @ b) / np.sqrt(dfe)
    tiny_s = 1e-6 * np.std(y, ddof=1) or 1.0

    D = np.sqrt(EPS)
    w = np.ones(n)
    b0 = np.zeros(p)
    it, wrank = 0, rank
    while it == 0 or np.any(np.abs(b - b0) > D * np.maximum(np.abs(b), np.abs(b0))):
        it += 1
        if it > max_iter:
            break
        radj = (y - X @ b) * adj
        s = _mad_sigma(radj, wrank)
        w = bisquare(radj / (max(s, tiny_s) * tune))
        b0 = b.copy()
        b[perm], wrank = _weighted_lstsq(X[:, perm], y, w)

    r = y - X @ b
    radj = r * adj
    mad_s = _mad_sigma(radj, rank)
    if np.all((w < D) | (w > 1 - D)):
        inc = w > 1 - D
        robust_s = np.linalg.norm(r[inc]) / np.sqrt(inc.sum() - rank)
    else:
        robust_s = _robust_sigma(radj, rank, max(mad_s, tiny_s), tune, h)
    sigma = max(robust_s, np.sqrt((ols_s**2 * rank**2 + robust_s**2 * n) / (rank**2 + n)))

    Ri = solve_triangular(Rr, np.eye(rank))
    covb = np.zeros((p, p))
    covb[np.ix_(perm, perm)] = Ri @ Ri.T * sigma**2

    wm = np.sum(w * y) / max(np.sum(w), EPS)
    r2 = 1.0 - np.sum(w * r**2) / max(np.sum(w * (y - wm) ** 2), EPS)
    return RobustFit(b, covb, w, float(dfe), float(sigma**2), float(r2), it)


def _mad_sigma(r: np.ndarray, p: int) -> float:
    """MAD scale from zero, omitting the p - 1 smallest absolute residuals."""
    rs = np.sort(np.abs(r))
    return float(np.median(rs[max(1, p) - 1:]) / 0.6745)


def _weighted_lstsq(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, int]:
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    Q, R, perm = qr(Xw, mode="economic", pivoting=True)
    tol = abs(R[0, 0]) * max(Xw.shape) * EPS
    rank = int(np.sum(np.abs(np.diag(R)) > tol))
    b = np.zeros(X.shape[1])
    b[perm[:rank]] = solve_triangular(R[:rank, :rank], Q[:, :rank].T @ (y * sw))
    return b, rank


def _robust_sigma(r: np.ndarray, p: int, s: float, tune: float, h: np.ndarray) -> float:
    """Robust sigma such that sigma^2 (X'X)^-1 is a fair covariance (Street et al. 1988).

    The derivative of psi is taken numerically, by a central difference with
    step 1e-4.
    """
    st = s * tune
    n = r.size
    u = r / st
    delta = 1e-4
    phi = u * bisquare(u)
    dphi = ((u + delta) * bisquare(u + delta) - (u - delta) * bisquare(u - delta)) / (2 * delta)
    m1 = np.mean(dphi)
    m2 = np.sum((1 - h) * phi**2) / (n - p)
    K = 1 + (p / n) * (1 - m1) / m1
    return float(K * np.sqrt(m2) * st / m1)
