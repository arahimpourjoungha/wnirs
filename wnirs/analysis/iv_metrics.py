"""Intraindividual variability metrics across dense sessions."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._stats import EPS, unique_stable
from .sessions import SessionMeasure, chronological, stack

COLUMNS = ["Subject", "Task", "Element", "nSessions", "iMean", "iSD", "iCV", "iSkew", "iKurtosis",
           "MSSD", "RMSSD", "RMSSD_iSD_ratio", "AR1", "PAC"]


def iv_metrics(sessions: Sequence[SessionMeasure], *, elements: Sequence[int] | None = None,
               acute_sd: float = 2.0) -> pd.DataFrame:
    """Describe what each person's signal does across sessions.

    The reliability tools ask whether a measurement is trustworthy; these ask
    how much a person moves around their own level, how fast, how jumpily, and
    whether one session predicts the next -- the standard descriptive
    vocabulary of the intensive-longitudinal literature (Bolger & Laurenceau
    2013), and the natural first pass on a densely sampled person.

    Two families, and the difference matters. Unstructured with respect to
    time (shuffle the sessions and they do not change): iMean, iSD, iCV =
    iSD/|iMean|, iSkew, iKurtosis. Time-structured (destroyed by shuffling):

    MSSD   sum of squared successive differences / (n - 1)
    RMSSD  its square root, on the scale of the measure
    ratio  RMSSD / iSD: near sqrt(2) when successive sessions are unrelated,
           well below for slow drift, above for session-to-session oscillation
    AR1    lag-1 autocorrelation -- inertia, whose rise is a documented
           early-warning sign of an approaching transition
    PAC    probability of acute change: fraction of successive jumps more than
           ``acute_sd`` SDs of the successive differences from their mean

    iSD says how far a person ranges, RMSSD how fast they get there: slow drift
    and rapid alternation between the same extremes have the same iSD and very
    different RMSSD.

    Parameters
    ----------
    sessions : sequence of SessionMeasure
        Ordered by ``order``, else ``datetime``, else numeric session label,
        else as given -- order matters for half of the metrics.
    elements : sequence of int, optional
        1-based element indices to summarise (default all).
    acute_sd : float
    """
    if len(sessions) < 3:
        raise ValueError(f"Successive-difference metrics need at least 3 sessions; got {len(sessions)}.")
    st = stack(sessions)
    V = st.V
    ids = np.arange(1, V.shape[0] + 1)
    if elements is not None:
        ids = np.asarray(elements, int)
        if np.any(ids < 1) or np.any(ids > V.shape[0]):
            raise ValueError(f"element must be between 1 and {V.shape[0]}.")
        V = V[ids - 1]
    rows = []
    for sb in unique_stable(st.subject):
        for tk in unique_stable(st.task):
            sel = np.flatnonzero((st.subject == sb) & (st.task == tk))
            if sel.size < 3:
                continue
            sel = sel[chronological([sessions[i] for i in sel], strict=False)]
            for e, x in zip(ids, V[:, sel]):
                x = x[np.isfinite(x)]
                if x.size >= 3:
                    rows.append([sb, tk, e, x.size, *series_metrics(x, acute_sd)])
    if not rows:
        raise ValueError("No subject/task had 3 or more sessions with finite data.")
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(["Subject", "Task", "Element"], kind="stable",
                                                           ignore_index=True)


def series_metrics(x: np.ndarray, acute_sd: float = 2.0) -> tuple[float, ...]:
    """(iMean, iSD, iCV, iSkew, iKurtosis, MSSD, RMSSD, ratio, AR1, PAC) of one ordered series."""
    n = x.size
    d = np.diff(x)
    mean, sd = x.mean(), np.std(x, ddof=1)
    mssd = np.sum(d**2) / (n - 1)
    ar1 = (np.corrcoef(x[:-1], x[1:])[0, 1]
           if np.std(x[:-1], ddof=1) > 0 and np.std(x[1:], ddof=1) > 0 else np.nan)
    sd_d = np.std(d, ddof=1) if d.size > 1 else 0.0
    pac = np.mean(np.abs(d - d.mean()) > acute_sd * sd_d) if sd_d > 0 else 0.0
    return (mean, sd, sd / max(abs(mean), EPS), _moment(x, 3, 3), _moment(x, 4, 4), mssd, np.sqrt(mssd),
            np.sqrt(mssd) / max(sd, EPS), ar1, pac)


def _moment(x: np.ndarray, order: int, min_n: int) -> float:
    """Standardised central moment (population SD), NaN for short or constant series."""
    if x.size < min_n or np.std(x) == 0:
        return np.nan
    return float(np.mean(((x - x.mean()) / np.std(x)) ** order))
