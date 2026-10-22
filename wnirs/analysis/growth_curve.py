"""Latent growth curves: does the group change, and do people differ in HOW they change?"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from ._stats import EPS, unique_stable
from .fdr import fdr_bh
from . import mixed
from .mixed import MixedFit, fit_mixed, random_intercept
from .sessions import SessionMeasure, session_numbers, stack

TABLE_COLUMNS = ["Element", "Shape", "Intercept", "Slope", "SlopeSE", "SlopeP", "SlopeQ", "SlopeVar", "ICC",
                 "nSubjects", "nObs"]


@dataclass
class GrowthCurve:
    """Result of :func:`growth_curve`.

    ``per_subject`` (OLS slope per subject and element) is always filled, and
    is the only output when there are too few subjects for a population model;
    ``adequate`` is False then and ``note`` says why.
    """
    per_subject: pd.DataFrame
    table: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TABLE_COLUMNS))
    fixed: dict = field(default_factory=dict)
    models: list[MixedFit] = field(default_factory=list)
    adequate: bool = False
    note: str = ""


def growth_curve(sessions: Sequence[SessionMeasure], *,
                 shape: Literal["linear", "quadratic", "compare"] = "linear", random_slope: bool | None = None,
                 elements: Sequence[int] | None = None, time_unit: Literal["session", "days"] = "session",
                 alpha: float = 0.05, min_subjects_fixed: int = 3, min_subjects_random: int = 8) -> GrowthCurve:
    """Multilevel growth curve per element, y(i,t) = (b0 + u0i) + (b1 + u1i) t + e.

    The change-point analysis asks whether ONE person's series moves. A growth
    curve asks the population question: is there an average trajectory, and do
    individuals differ reliably in their own slopes? var(u1i) is the
    statistical form of "these people are not all on the same path". This is
    the multilevel form of a latent growth curve model; it handles unbalanced
    session counts and missing observations directly.

    IDENTIFIABILITY. A random-slope variance is a between-person variance and
    needs between-person replication: with two subjects it is the variance of
    two numbers. Below ``min_subjects_random`` (8) a random slope is refused
    (with a warning if one was explicitly asked for) and only a random
    intercept fitted; below ``min_subjects_fixed`` (3) no model is fitted and
    the per-subject slopes are returned instead. Adding PEOPLE buys far more
    power for a population effect than adding occasions, so dense sampling is
    the right design for within-person questions and the wrong one for this.

    Parameters
    ----------
    sessions : sequence of SessionMeasure
        Time is the numeric session label (or ``datetime`` in days with
        ``time_unit='days'``) from each subject's first session; text labels
        fall back to arrival order.
    shape : {'linear', 'quadratic', 'compare'}
        ``compare`` fits intercept-only, linear and quadratic and keeps the
        one with the lowest BIC.
    random_slope : bool, optional
        Let people differ in their rate of change. Default: when there are at
        least ``min_subjects_random`` subjects.
    elements : sequence of int, optional
        1-based elements to model; default all.
    """
    st = stack(sessions)
    V = st.V
    ids = np.arange(1, V.shape[0] + 1) if elements is None else np.asarray(elements, int)
    V = V[ids - 1]
    t = _time(sessions, st.subject, time_unit)
    n_subj = unique_stable(st.subject).size
    result = GrowthCurve(per_subject=_per_subject_slopes(V, st.subject, t, ids))
    if n_subj < min_subjects_fixed:
        result.note = (f"Only {n_subj} subject(s). A growth curve estimates a POPULATION trajectory, which needs at "
                       f"least {min_subjects_fixed}, so none was fitted; per_subject holds descriptive slopes. Use "
                       "change_point for the within-person question, which is what this design supports.")
        return result

    use_slope = n_subj >= min_subjects_random if random_slope is None else random_slope
    if use_slope and n_subj < min_subjects_random:
        warnings.warn(f"A random slope variance is a between-person variance and {n_subj} subject(s) cannot "
                      "estimate one; fitting a random intercept only. Pass min_subjects_random to override.",
                      stacklevel=2)
        use_slope = False
    shapes = ["intercept", "linear", "quadratic"] if shape == "compare" else [shape]

    rows, models = [], []
    for e, y in zip(ids, V):
        fit = _fit_element(y, st.subject, t, shapes, use_slope)
        if fit is not None:
            rows.append([e, *fit[:-1]])
            models.append(fit[-1])
    if not rows:
        result.note = "No element had enough finite observations to fit."
        return result
    T = pd.DataFrame(rows, columns=[c for c in TABLE_COLUMNS if c != "SlopeQ"])
    T.insert(TABLE_COLUMNS.index("SlopeQ"), "SlopeQ", fdr_bh(T["SlopeP"].to_numpy()))
    result.table, result.models, result.adequate = T, models, True
    result.fixed = dict(mean_slope=float(np.nanmedian(T["Slope"])) if T["Slope"].notna().any() else np.nan,
                        n_significant=int(np.sum(T["SlopeQ"] < alpha)), n_elements=len(T),
                        random_slope_fitted=use_slope)
    result.note = f"{'/'.join(shapes)} growth curve, random slope {str(use_slope).lower()}, {n_subj} subject(s)."
    return result


def _fit_element(y, subject, t, shapes, use_slope):
    ok = np.isfinite(y)
    if ok.sum() < 8 or np.unique(subject[ok]).size < 2:
        return None
    y, g = y[ok], subject[ok]
    # Centre time so the intercept is the level at the average occasion and the
    # quadratic term is not collinear with the linear one.
    tc = t[ok] - t[ok].mean()
    design = {"intercept": [np.ones_like(tc)], "linear": [np.ones_like(tc), tc],
              "quadratic": [np.ones_like(tc), tc, tc**2]}
    best, best_shape = None, ""
    for s in shapes:
        if s not in design:
            raise ValueError(f'Unknown shape "{s}".')
        slope = use_slope and s != "intercept"
        term = mixed.random_slope("subject", g, tc) if slope else random_intercept("subject", g)
        try:
            fit = fit_mixed(y, np.column_stack(design[s]), [term], method="reml")
        except np.linalg.LinAlgError:
            continue
        if best is None or fit.bic < best.bic:
            best, best_shape = fit, s
    if best is None:
        return None
    slope = (best.beta[1], best.se[1], best.pval[1]) if best_shape != "intercept" else (np.nan,) * 3
    P = best.psi["subject"]
    slope_var = P[1, 1] if use_slope and P.shape[0] >= 2 else np.nan
    icc = P[0, 0] / max(P[0, 0] + best.sigma2, EPS)
    return (best_shape, best.beta[0], *slope, slope_var, icc, np.unique(g).size, y.size, best)


def _per_subject_slopes(V, subject, t, ids) -> pd.DataFrame:
    rows = []
    for sb in unique_stable(subject):
        sel = subject == sb
        for e, y in zip(ids, V[:, sel]):
            ok = np.isfinite(y) & np.isfinite(t[sel])
            if ok.sum() >= 3:
                X = np.column_stack((np.ones(ok.sum()), t[sel][ok]))
                rows.append([sb, e, np.linalg.lstsq(X, y[ok], rcond=None)[0][1], int(ok.sum())])
    return pd.DataFrame(rows, columns=["Subject", "Element", "Slope", "nSessions"])


def _time(sessions, subject, unit) -> np.ndarray:
    """Time from each subject's first session: growth is modelled from study entry."""
    t = np.full(len(sessions), np.nan)
    if unit == "days":
        if any(s.datetime is None for s in sessions):
            raise ValueError('time_unit "days" needs a datetime on every session.')
        dt = np.array([s.datetime for s in sessions], dtype="datetime64[ns]")
        for sb in unique_stable(subject):
            m = subject == sb
            t[m] = (dt[m] - dt[m].min()) / np.timedelta64(1, "D")
        return t
    num = session_numbers([s.session for s in sessions])
    for sb in unique_stable(subject):
        m = subject == sb
        t[m] = num[m] - num[m].min() if np.all(np.isfinite(num)) else np.arange(m.sum())
    return t
