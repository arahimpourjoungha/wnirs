"""GLM design matrix: HRF-convolved boxcars plus nuisance terms."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..types import Conditions
from .hrf import HRFKind, hrf


@dataclass
class Design:
    """A design matrix and the names of its columns.

    The first ``n_condition_regressors`` columns are the task regressors (one
    per condition, plus a derivative after each when requested); nuisance
    terms follow.
    """
    X: np.ndarray
    names: list[str]
    n_condition_regressors: int


def boxcar(time: np.ndarray, onsets, durations) -> np.ndarray:
    """1 while any block is on (onset <= t < onset + duration), else 0."""
    on = np.asarray(onsets, float)[None, :]
    off = on + np.asarray(durations, float)[None, :]
    t = np.asarray(time, float)[:, None]
    return np.any((t >= on) & (t < off), axis=1).astype(float)


def _convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.convolve(x, h)[: x.size]


def design_matrix(time: np.ndarray, conditions: Conditions, fs: float, *, hrf_kind: HRFKind = "canonical",
                  derivative: bool = False, drift: int = 1, short_regressor: np.ndarray | None = None,
                  label: str = "") -> Design:
    """Build the subject-level design matrix for one recording.

    Each condition regressor is scaled to unit peak. Beta then reads directly
    as the peak response in micromolar, and it no longer depends on the
    sampling rate: convolution sums scale with fs, so unnormalised betas from a
    4.71 Hz and a 4.72 Hz recording would not be comparable at the group level.

    Parameters
    ----------
    time : (T,) array
        Sample times, on the same clock as the condition onsets.
    conditions : Conditions
    fs : float
    hrf_kind : {'canonical', 'gamma'}
    derivative : bool
        Add a temporal-derivative regressor after each condition, scaled by
        the same factor so the pair stays on one footing.
    drift : int
        Polynomial drift order on time rescaled to [-1, 1]; 0 = intercept only.
    short_regressor : (T,) array, optional
        Mean short-channel HbO, added (z-scored) as a nuisance column. Leave it
        out if short-separation regression already ran in preprocessing, or the
        same signal is removed twice.
    label : str
        Used in warnings only.
    """
    if len(conditions) == 0:
        raise ValueError(f"{label}: no stimulus blocks -- there is nothing to model.")
    T = len(time)
    h = hrf(fs, hrf_kind)[0]
    hd = hrf(fs, "derivative")[0] if derivative else None
    cols, names = [], []
    for name, onsets, durations in zip(conditions.name, conditions.onsets, conditions.durations):
        box = boxcar(time, onsets, durations)
        reg = _convolve(box, h)
        peak = np.max(np.abs(reg))
        if peak <= 0:
            warnings.warn(f'Condition "{name}" produced an all-zero regressor.', stacklevel=2)
            peak = 1.0
        cols.append(reg / peak)
        names.append(str(name))
        if derivative:
            cols.append(_convolve(box, hd) / peak)
            names.append(f"{name}:deriv")
    n_cond = len(cols)

    if short_regressor is not None:
        cols.append(_zscore_safe(short_regressor))
        names.append("shortMeanHbO")
    tn = np.linspace(-1, 1, T)
    for p in range(drift + 1):
        cols.append(tn**p)
        names.append("intercept" if p == 0 else f"drift{p}")
    X = np.column_stack(cols)
    _check_design(X, n_cond, label)
    return Design(X, names, n_cond)


def _zscore_safe(x: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(x), x, 0.0)
    s = np.std(x, ddof=1)
    return x - x.mean() if s == 0 else (x - x.mean()) / s


def _check_design(X: np.ndarray, n_cond: int, label: str) -> None:
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        warnings.warn(f"{label}: the design matrix is rank deficient (rank {rank} of {X.shape[1]} columns). "
                      "Conditions may overlap perfectly or a drift term may be redundant.", stacklevel=3)
    # Rapid event-related designs can leave two conditions nearly
    # indistinguishable after HRF convolution; the contrast between them then
    # rests on very little independent information.
    if n_cond > 1:
        rc = np.corrcoef(X[:, :n_cond], rowvar=False)
        off = np.abs(rc[np.triu_indices(n_cond, 1)])
        if np.any(off > 0.9):
            warnings.warn(f"{label}: two condition regressors correlate at r = {off.max():.2f} after HRF "
                          "convolution. Contrasts between them are poorly determined -- check the trial "
                          "ordering and the inter-stimulus intervals.", stacklevel=3)
