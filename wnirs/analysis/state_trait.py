"""Split a repeated measure into trait, systematic change and state, within each person."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ._stats import EPS, unique_stable
from .mixed import fit_mixed, random_intercept

BY_SUBJECT = ["Subject", "Measure", "nSessions", "TraitLevel", "PctTrend", "PctCondition", "PctState",
              "SDState", "AR1", "SessionsFor", "Target", "Conditions"]
BY_MEASURE = ["Measure", "nSubjects", "PctTrend", "PctCondition", "PctState", "AR1", "ICC_between",
              "SessionsForWorstSubject"]


def state_trait(T: pd.DataFrame, *, min_obs: int = 8,
                target: float | Mapping[str, float] = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within-person decomposition of session-to-session variation.

    LST, LST-MSST, TSO, CUTS and DSEM identify a trait factor from variance
    BETWEEN people, so they need many persons and a few occasions each. A
    precision study is the mirror image -- a few people, many occasions -- and
    a between-person trait factor estimated from three points is not a
    measurement. What it can do well is decompose variation WITHIN a person:
    a stable level, systematic drift or treatment, and genuine
    occasion-specific state.

    Slow drift across weeks looks like state variance to a simple
    within-person variance split, but it is trait change. Removing the linear
    trend first and reporting the lag-1 autocorrelation of what remains tells
    the two apart: AR1 near zero means the residual really is
    occasion-specific; a high AR1 means something slower is still in there.

    Parameters
    ----------
    T : DataFrame
        Long table with Subject, SessionNo, Measure, Value and optionally
        Condition (treatment arm), one row per subject x session x measure.
    min_obs : int
        Sessions needed to model a subject.
    target : float or mapping
        Precision the session count aims at (in the measure's units), either
        one value for all measures or per measure name; a measure missing
        from the mapping gets a fifth of its own SD.

    Returns
    -------
    by_subject : DataFrame
        TraitLevel (the person's mean level), PctTrend / PctCondition /
        PctState (shares of within-person variance), SDState, AR1 (over
        consecutive sessions only), SessionsFor (sessions for the person-mean
        SE to reach ``target``).
    by_measure : DataFrame
        Pooled per measure, with a between-person ICC from a random-intercept
        model -- DESCRIPTIVE ONLY with a handful of people. Sorted by PctState,
        descending.
    """
    T = T.copy()
    T["Subject"], T["Measure"] = T["Subject"].astype(str), T["Measure"].astype(str)
    T["Condition"] = T["Condition"].fillna("").astype(str) if "Condition" in T else ""
    rows = []
    for m in unique_stable(T["Measure"]):
        for s in unique_stable(T["Subject"]):
            q = T[(T["Subject"] == s) & (T["Measure"] == m) & np.isfinite(T["Value"])]
            q = q.sort_values("SessionNo", kind="stable")
            if len(q) >= min_obs:
                row = _one_person(q, m, target)
                if row is not None:
                    rows.append([s, m, len(q), *row])
    if not rows:
        raise ValueError(f"No subject-measure pair had {min_obs} or more sessions.")
    V = pd.DataFrame(rows, columns=BY_SUBJECT)

    mrows = []
    for m in unique_stable(V["Measure"]):
        q = V[V["Measure"] == m]
        tq = T[(T["Measure"] == m) & np.isfinite(T["Value"])]
        icc = np.nan
        if tq["Subject"].nunique() >= 2:
            fit = fit_mixed(tq["Value"].to_numpy(float), np.ones((len(tq), 1)),
                            [random_intercept("Subject", tq["Subject"].to_numpy())], method="ml")
            vb = fit.psi["Subject"][0, 0]
            icc = vb / (vb + fit.sigma2)
        ar1 = q["AR1"].to_numpy(float)
        mrows.append([m, len(q), q["PctTrend"].mean(), q["PctCondition"].mean(), q["PctState"].mean(),
                      np.nanmean(ar1) if np.isfinite(ar1).any() else np.nan, icc, q["SessionsFor"].max()])
    M = pd.DataFrame(mrows, columns=BY_MEASURE).sort_values("PctState", ascending=False, kind="stable",
                                                           ignore_index=True)
    return V, M


def _one_person(q: pd.DataFrame, measure: str, target) -> list | None:
    y = q["Value"].to_numpy(float)
    sess = q["SessionNo"].to_numpy(float)
    x = sess - sess.mean()                  # centred, so the intercept is the person's mean level
    tot_var = np.var(y, ddof=1)
    if tot_var <= 0:
        return None
    cols, kinds = [np.ones_like(y)], ["const"]
    if np.std(x, ddof=1) > 0:
        cols.append(x)
        kinds.append("trend")
    cond = q["Condition"].to_numpy(str)
    levels = unique_stable(cond[cond != ""])
    for lv in levels[1:] if levels.size >= 2 else []:
        cols.append((cond == lv).astype(float))
        kinds.append("cond")
    X = np.column_stack(cols)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    kinds = np.array(kinds)
    v_trend = np.var(X[:, kinds == "trend"] @ b[kinds == "trend"], ddof=1)
    v_cond = np.var(X[:, kinds == "cond"] @ b[kinds == "cond"], ddof=1)
    v_state = np.var(res, ddof=1)
    denom = v_trend + v_cond + v_state
    if denom <= 0:
        return None
    # Lag-1 autocorrelation over CONSECUTIVE sessions only: sessions are
    # unequally spaced, and a lag across a gap is not a one-session lag.
    adj = np.flatnonzero(np.diff(sess) == 1)
    ar1 = np.corrcoef(res[adj], res[adj + 1])[0, 1] if adj.size >= 5 else np.nan
    if isinstance(target, Mapping):
        target = target.get(measure, 0.2 * np.sqrt(tot_var))     # a fifth of this measure's own spread
    sd_state = np.sqrt(v_state)
    k_need = int(np.ceil((1.96 * sd_state / max(target, EPS)) ** 2))
    return [b[0], 100 * v_trend / denom, 100 * v_cond / denom, 100 * v_state / denom, sd_state, ar1, k_need,
            float(target), "/".join(levels)]
