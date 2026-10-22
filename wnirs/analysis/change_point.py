"""Is this person's signal drifting, stepping, or just noisy?"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._stats import EPS, unique_stable
from .fdr import fdr_bh
from .sessions import SessionMeasure, chronological, stack

COLUMNS = ["Subject", "Task", "Element", "nSessions", "Model", "Slope", "StepAt", "StepSize",
           "pTrend", "pStep", "qTrend", "qStep"]


def change_point(sessions: Sequence[SessionMeasure], *, n_perm: int = 2000, min_shift: int = 3,
                 rng: np.random.Generator | int | None = None) -> pd.DataFrame:
    """Constant vs. linear trend vs. step, per subject, task and element.

    Every reliability measure assumes it is measuring a STABLE quantity
    repeatedly; under that assumption genuine within-person change is
    indistinguishable from unreliability. For a progressive condition the
    change IS the signal. This fits all three explanations and lets BIC choose:

    constant  one level, sessions vary around it as noise
    trend     the level moves steadily -- gradual decline or practice
    step      the level jumps at some session and stays (location found by
              exhaustive search, at least ``min_shift`` sessions either side)

    Significance is by permutation, and has to be: the models are not nested
    in a way that gives an analytic null, the step location is chosen by
    looking at the data, and session series are short. Shuffling session order
    destroys temporal structure while keeping the values, which is the null of
    "no temporal organisation". Read a null result as a result: reproducible
    sessions with no trend or step mean the person is not moving over this
    window, which is a finding about the study duration.

    Parameters
    ----------
    sessions : sequence of SessionMeasure
        Ordered by ``order``, else ``datetime``, else numeric session label;
        sessions that cannot be ordered are refused.
    n_perm : int
    min_shift : int
        Smallest run of sessions either side of a step; a "step" one session
        from the end is an outlier, not a shift.

    Returns
    -------
    pandas.DataFrame
        One row per subject x task x element: Model, Slope (per session),
        StepAt (session index of the last pre-step session, NaN when no split
        beats a constant), StepSize, permutation pTrend/pStep, and BH qTrend/qStep
        within subject and task (correcting across subjects would let one
        person's flat data buy another's significance).
    """
    if len(sessions) < 8:
        raise ValueError(f"Separating a trend from a step from noise needs a reasonable run of sessions; got "
                         f"{len(sessions)}. Below about 8 the three models are not distinguishable.")
    rng = np.random.default_rng(rng)
    st = stack(sessions)
    rows = []
    for sb in unique_stable(st.subject):
        for tk in unique_stable(st.task):
            sel = np.flatnonzero((st.subject == sb) & (st.task == tk))
            if sel.size < 8 or sel.size < 2 * min_shift + 2:
                continue
            sel = sel[chronological([sessions[i] for i in sel], strict=True)]
            block = []
            for e, y in enumerate(st.V[:, sel]):
                ok = np.isfinite(y)
                if ok.sum() < 2 * min_shift + 2:
                    continue
                fit = fit_series(y[ok], np.flatnonzero(ok) + 1.0, n_perm=n_perm, min_shift=min_shift, rng=rng)
                block.append([sb, tk, e + 1, int(ok.sum()), *fit])
            if block:
                b = pd.DataFrame(block, columns=COLUMNS[:10])
                b["qTrend"], b["qStep"] = fdr_bh(b["pTrend"]), fdr_bh(b["pStep"])
                rows.append(b)
    if not rows:
        raise ValueError("No subject/task had enough ordered sessions to test for change.")
    return pd.concat(rows, ignore_index=True).sort_values(["Subject", "Task", "Element"], kind="stable",
                                                          ignore_index=True)


def fit_series(y: np.ndarray, t: np.ndarray, *, n_perm: int = 2000, min_shift: int = 3,
               rng: np.random.Generator | None = None) -> tuple:
    """Fit constant / trend / step to one series and test trend and step by permutation.

    Returns ``(model, slope, step_at, step_size, p_trend, p_step)``.
    """
    rng = np.random.default_rng(rng)
    n = y.size
    tc = (t - t.mean()) / max(np.std(t, ddof=1), EPS)          # scaled so the slope is readable
    ss_const = np.sum((y - y.mean()) ** 2)
    ss_trend, slope = _trend_ss(y[None], tc)
    ss_step, at, size = _step_ss(y[None], min_shift)
    bic = [n * np.log(max(ss, EPS) / n) + k * np.log(n) for ss, k in ((ss_const, 1), (ss_trend[0], 2),
                                                                     (ss_step[0], 3))]
    model = ("constant", "trend", "step")[int(np.argmin(bic))]
    step_at = float(at[0]) if np.isfinite(size[0]) else np.nan

    perm = rng.permuted(np.tile(y, (n_perm, 1)), axis=1)
    ss_c = np.sum((perm - perm.mean(axis=1, keepdims=True)) ** 2, axis=1)
    p_trend = (1 + np.sum(ss_c - _trend_ss(perm, tc)[0] >= ss_const - ss_trend[0])) / (n_perm + 1)
    p_step = (1 + np.sum(ss_c - _step_ss(perm, min_shift)[0] >= ss_const - ss_step[0])) / (n_perm + 1)
    return model, slope[0] / max(np.std(t, ddof=1), EPS), step_at, size[0], p_trend, p_step


def _trend_ss(Y: np.ndarray, tc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residual sum of squares and slope of [1, tc] regressions, one per row of Y."""
    X = np.column_stack((np.ones_like(tc), tc))
    beta = np.linalg.lstsq(X, Y.T, rcond=None)[0]
    return np.sum((Y.T - X @ beta) ** 2, axis=0), beta[1]


def _step_ss(Y: np.ndarray, min_shift: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best two-level split of each row of Y by exhaustive search.

    A split only counts if it beats the constant model strictly; the first
    best split wins ties. Returns the residual SS, the split (number of
    sessions before the step, NaN if none) and mean(after) - mean(before).
    """
    n = Y.shape[1]
    cuts = np.arange(min_shift, n - min_shift + 1)
    before = np.arange(n)[None, :] < cuts[:, None]                       # (n_cuts, n)
    ss = np.empty((Y.shape[0], cuts.size))
    jump = np.empty_like(ss)
    for j, m in enumerate(before):
        a, b = Y[:, m], Y[:, ~m]
        ma, mb = a.mean(axis=1, keepdims=True), b.mean(axis=1, keepdims=True)
        ss[:, j] = np.sum((a - ma) ** 2, axis=1) + np.sum((b - mb) ** 2, axis=1)
        jump[:, j] = (mb - ma)[:, 0]
    ss_const = np.sum((Y - Y.mean(axis=1, keepdims=True)) ** 2, axis=1)
    best = np.argmin(ss, axis=1)
    rows = np.arange(Y.shape[0])
    better = ss[rows, best] < ss_const
    return (np.where(better, ss[rows, best], ss_const), np.where(better, cuts[best], np.nan),
            np.where(better, jump[rows, best], np.nan))
