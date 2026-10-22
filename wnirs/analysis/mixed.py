"""Linear mixed models by profiled (RE)ML, for variance decompositions and growth curves.

The parameterisation is the one R's ``lme4`` uses:
the random effects are b = Lambda(theta) u with u ~ N(0, sigma^2 I), the fixed
effects and sigma are profiled out, and only the relative Cholesky factors
theta are optimised. That makes the criterion a function of a handful of
parameters however many random-effect levels there are, and it lets a
crossed design (person, person x session, channel, person x channel) be fitted
directly, which is what the generalizability analysis needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy import sparse
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize

from ._stats import t_pvalue


@dataclass
class RandomTerm:
    """Random effects of one grouping factor.

    ``Z`` has ``n_levels * n_effects`` columns ordered level-major, so the
    effects of one level are adjacent and share an ``n_effects`` square
    covariance.
    """
    name: str
    Z: sparse.csr_matrix
    n_effects: int


def random_intercept(name: str, groups) -> RandomTerm:
    """(1 | groups)."""
    codes = np.unique(np.asarray(groups), return_inverse=True)[1]
    n = codes.size
    return RandomTerm(name, sparse.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, codes.max() + 1)), 1)


def random_slope(name: str, groups, x) -> RandomTerm:
    """(1 + x | groups) with a full 2 x 2 covariance."""
    codes = np.unique(np.asarray(groups), return_inverse=True)[1]
    n, g = codes.size, codes.max() + 1
    rows = np.repeat(np.arange(n), 2)
    cols = np.column_stack((2 * codes, 2 * codes + 1)).ravel()
    vals = np.column_stack((np.ones(n), np.asarray(x, float))).ravel()
    return RandomTerm(name, sparse.csr_matrix((vals, (rows, cols)), shape=(n, 2 * g)), 2)


@dataclass
class MixedFit:
    beta: np.ndarray
    cov_beta: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    pval: np.ndarray
    dfe: int
    sigma2: float                    # residual variance
    psi: dict[str, np.ndarray]       # covariance of each term's random effects
    loglik: float
    bic: float
    method: str
    converged: bool


def fit_mixed(y: np.ndarray, X: np.ndarray, terms: Sequence[RandomTerm],
              method: Literal["reml", "ml"] = "reml") -> MixedFit:
    """Fit y = X beta + sum_k Z_k b_k + e.

    Fixed-effect p values use the residual degrees of freedom n - p. BIC
    counts the fixed effects, the covariance parameters and the residual
    variance, with n - p observations under REML.
    """
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, p = X.shape
    reml = method.lower() == "reml"
    Z = sparse.hstack([t.Z for t in terms]).tocsc()
    tri = [np.tril_indices(t.n_effects) for t in terms]
    on_diagonal = np.concatenate([r == c for r, c in tri])
    n_theta = on_diagonal.size
    XtX, Xty = X.T @ X, X.T @ y

    def templates(theta):
        """Each term's lower-triangular relative Cholesky factor."""
        out = np.split(theta, np.cumsum([r.size for r, _ in tri])[:-1])
        Ts = []
        for t, (r, c), vals in zip(terms, tri, out):
            T = np.zeros((t.n_effects, t.n_effects))
            T[r, c] = vals
            Ts.append(T)
        return Ts

    def solve(theta):
        Lam = sparse.block_diag([sparse.kron(sparse.identity(t.Z.shape[1] // t.n_effects), T)
                                 for t, T in zip(terms, templates(theta))], format="csc")
        U = Z @ Lam
        L = cholesky((U.T @ U).toarray() + np.eye(U.shape[1]), lower=True)
        cu = solve_triangular(L, U.T @ y, lower=True)
        RZX = solve_triangular(L, (U.T @ X), lower=True)
        Rx = cholesky(XtX - RZX.T @ RZX, lower=False)
        beta = cho_solve((Rx, False), Xty - RZX.T @ cu)
        delta = solve_triangular(L.T, cu - RZX @ beta, lower=False)
        resid = y - X @ beta - U @ delta
        r2 = resid @ resid + delta @ delta
        logdet = np.sum(np.log(np.diag(L))) + (np.sum(np.log(np.diag(Rx))) if reml else 0.0)
        n_eff = n - p if reml else n
        ll = -n_eff / 2 * (1 + np.log(2 * np.pi * r2 / n_eff)) - logdet
        return ll, beta, Rx, r2 / n_eff

    # Start from independent unit-variance effects; Cholesky diagonals stay
    # non-negative, which is what lets a variance land exactly on zero.
    res = minimize(lambda th: -solve(th)[0], on_diagonal.astype(float), method="L-BFGS-B",
                   bounds=[(0.0, None) if d else (None, None) for d in on_diagonal],
                   options=dict(ftol=1e-13, gtol=1e-9, maxiter=2000))
    theta = res.x
    ll, beta, Rx, sigma2 = solve(theta)
    Rinv = solve_triangular(Rx, np.eye(p))
    cov_beta = sigma2 * Rinv @ Rinv.T
    se = np.sqrt(np.diag(cov_beta))
    psi = {t.name: sigma2 * T @ T.T for t, T in zip(terms, templates(theta))}
    n_eff = n - p if reml else n
    return MixedFit(beta=beta, cov_beta=cov_beta, se=se, tstat=beta / se, pval=t_pvalue(beta / se, n - p),
                    dfe=n - p, sigma2=float(sigma2), psi=psi, loglik=float(ll),
                    bic=float(-2 * ll + (n_theta + p + 1) * np.log(n_eff)), method=method.lower(),
                    converged=bool(res.success))
