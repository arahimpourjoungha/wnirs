"""Within-person analysis across many sessions of one person."""
from __future__ import annotations

import warnings
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from ._stats import nanmean, t_pvalue, unique_stable
from .fdr import fdr_bh
from .sessions import session_numbers


def dense_sampling(session_stats: Sequence[Mapping[str, Any]], *,
                   model: Literal["mean", "trend", "predictor"] = "mean", predictor: str | None = None,
                   covariates: Sequence[str] = (), weight: Literal["equal", "inverse"] = "equal") -> pd.DataFrame:
    """Per-channel model across the sessions of ONE person.

    The usual group model treats each person as one observation and asks what
    is true on average across people. Dense sampling asks whether, within this
    person, the response varies with something measured repeatedly. The unit
    of replication is the session; the between-person variance that normally
    dominates fNIRS is absent by construction.

    It carries a different risk: sessions of one person are not independent
    the way people are. Cap placement, hair and slow physiological state
    persist and drift over weeks, so a significant channel may reflect that
    drift. Check results against a ``'sessionNumber'`` covariate whenever a
    trend is not the thing being tested, and state conclusions as within-person.

    Parameters
    ----------
    session_stats : sequence of mapping
        One per session with ``'session'`` (id or number), ``'table'`` (a
        contrast table) and any per-session covariates as further keys.
    model : {'mean', 'trend', 'predictor'}
        ``mean``: is the response non-zero across sessions (one-sample t).
        ``trend``: does it change linearly with session number.
        ``predictor``: does it covary with ``session_stats[i][predictor]``.
    predictor : str
        Key to regress on for ``model='predictor'``.
    covariates : sequence of str
        Keys partialled out first; ``'sessionNumber'`` is always available.
    weight : {'equal', 'inverse'}
        For ``mean``: weight sessions by 1/SE^2 from the first level.

    Returns
    -------
    pandas.DataFrame
        One row per channel per chromophore: Channel, Source, Detector, Label,
        Chromophore, N, Estimate, SE, Tstat, Pval, Qval. ``attrs`` carries the
        model, the effect label, the session ids, the beta matrix, the
        session numbers and ``reliability`` (see :func:`pattern_reliability`).
    """
    n = len(session_stats)
    if n < 3:
        raise ValueError(f"Dense sampling needs at least 3 sessions to say anything about within-person "
                         f"variation; got {n}. Use a group analysis for a between-person question.")
    ref = session_stats[0]["table"]
    key = ref["Label"].astype(str) + "|" + ref["Chromophore"].astype(str)
    B = np.column_stack([_lookup(s["table"], key, "Beta") for s in session_stats])
    S = np.column_stack([_lookup(s["table"], key, "SE") for s in session_stats])
    sess_no = session_numbers([s["session"] for s in session_stats])
    sess_no = np.where(np.isfinite(sess_no), sess_no, np.arange(1, n + 1))   # unparseable ids: arrival order
    Z = _covariates(session_stats, covariates, sess_no)
    rel = pattern_reliability(B, S)

    model = model.lower()
    if model == "mean":
        label = "within-person mean across sessions"
        res = [_one_sample(B[r], S[r], Z, weight) for r in range(len(ref))]
    elif model == "trend":
        label = "change per session"
        res = [_regress_on(B[r], sess_no, Z) for r in range(len(ref))]
    elif model == "predictor":
        if not predictor or predictor not in session_stats[0]:
            raise ValueError(f'predictor must name a key of session_stats; "{predictor}" is not one.')
        x = np.array([float(s[predictor]) for s in session_stats])
        label = f"slope on {predictor}"
        res = [_regress_on(B[r], x, Z) for r in range(len(ref))]
    else:
        raise ValueError(f'Unknown model "{model}". Use mean, trend, or predictor.')
    est, se, t, p, N = map(np.array, zip(*res))

    chromo = ref["Chromophore"].astype(str).to_numpy()
    q = np.full(len(ref), np.nan)
    for c in unique_stable(chromo):
        q[chromo == c] = fdr_bh(p[chromo == c])
    out = pd.DataFrame({"Channel": ref["Channel"].to_numpy(), "Source": ref["Source"].to_numpy(),
                        "Detector": ref["Detector"].to_numpy(), "Label": ref["Label"].to_numpy(),
                        "Chromophore": chromo, "N": N.astype(int), "Estimate": est, "SE": se, "Tstat": t,
                        "Pval": p, "Qval": q})
    out.attrs = dict(model=model, effect=label, sessions=[str(s["session"]) for s in session_stats],
                     betas=B, reliability=rel, session_number=sess_no)
    return out


def pattern_reliability(B: np.ndarray, S: np.ndarray) -> dict:
    """How repeatable one person's map is across sessions.

    An ICC per channel is deliberately NOT used: ICC is between-session
    variance over total, so a channel giving the same value every session --
    the most reliable thing it could do -- scores near zero. For repeated
    measures of one person that is backwards. Instead:

    ``pattern``
        Mean pairwise correlation between sessions of the whole channel vector
        (on channels present in every non-empty session): is the spatial map
        the same each time?
    ``per_channel``
        Session-to-session SD of a channel divided by its mean within-session
        SE. Near 1 the channel varies no more than its own measurement noise,
        so there is nothing for a trend or predictor model to explain.

    A session is dropped only if it is entirely empty: requiring every
    channel would discard every session as soon as QC excluded one channel,
    and silently return NaN.
    """
    rel = dict(pattern=np.nan, per_channel=np.full(B.shape[0], np.nan), n_pairs=0)
    Bu = B[:, np.any(np.isfinite(B), axis=0)]
    n_s = Bu.shape[1]
    if n_s >= 2:
        good = np.all(np.isfinite(Bu), axis=1)
        if good.sum() >= 3:
            with np.errstate(invalid="ignore", divide="ignore"):     # a constant session has no correlation
                C = np.corrcoef(Bu[good], rowvar=False)
            iu = np.triu_indices(n_s, 1)
            rel["pattern"] = float(nanmean(C[iu]))
            rel["n_pairs"] = iu[0].size
    for r in range(B.shape[0]):
        ok = np.isfinite(B[r]) & np.isfinite(S[r]) & (S[r] > 0)
        if ok.sum() >= 3:
            rel["per_channel"][r] = np.std(B[r, ok], ddof=1) / np.mean(S[r, ok])
    return rel


def _lookup(table: pd.DataFrame, key: pd.Series, column: str) -> np.ndarray:
    k = table["Label"].astype(str) + "|" + table["Chromophore"].astype(str)
    return pd.Series(table[column].to_numpy(float), index=k)[~k.duplicated().to_numpy()].reindex(key).to_numpy()


def _covariates(stats: Sequence[Mapping[str, Any]], names: Sequence[str], sess_no: np.ndarray) -> np.ndarray | None:
    cols = []
    for nm in names:
        if nm.lower() == "sessionnumber":
            cols.append(sess_no)
        elif nm in stats[0]:
            cols.append(np.array([_to_number(s.get(nm)) for s in stats]))
        else:
            warnings.warn(f'Covariate "{nm}" is not a key of session_stats -- skipped.', stacklevel=3)
    return np.column_stack(cols) if cols else None


def _to_number(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _one_sample(b, s, Z, weight):
    ok = np.isfinite(b)
    if Z is not None:
        ok &= np.all(np.isfinite(Z), axis=1)
    N = int(ok.sum())
    nan = (np.nan, np.nan, np.nan, np.nan, N)
    if N < 3:
        return nan
    bb, df = b[ok], N - 1
    # Partial out covariates before testing the mean, so a session-number
    # drift is not read as a response.
    if Z is not None:
        Zr = np.column_stack((np.ones(N), Z[ok]))
        if np.linalg.matrix_rank(Zr) < Zr.shape[1]:
            Zr = np.ones((N, 1))
        bb = bb - Zr @ np.linalg.lstsq(Zr, bb, rcond=None)[0] + bb.mean()
        df = N - Zr.shape[1]
    if df < 1:
        return nan
    if weight == "inverse" and np.all(np.isfinite(s[ok])) and np.all(s[ok] > 0):
        w = 1 / s[ok] ** 2
        w /= w.sum()
        est = np.sum(w * bb)
        se = np.sqrt(np.sum(w * (bb - est) ** 2) * N / (N - 1) * np.sum(w**2))
    else:
        est, se = bb.mean(), np.std(bb, ddof=1) / np.sqrt(N)
    if se <= 0:
        return nan
    return est, se, est / se, float(t_pvalue(est / se, df)), N


def _regress_on(b, x, Z):
    ok = np.isfinite(b) & np.isfinite(x)
    if Z is not None:
        ok &= np.all(np.isfinite(Z), axis=1)
    N = int(ok.sum())
    nan = (np.nan, np.nan, np.nan, np.nan, N)
    if N < 4:
        return nan
    X = np.column_stack([np.ones(N), x[ok]] + ([Z[ok]] if Z is not None else []))
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return nan
    y = b[ok]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    df = N - X.shape[1]
    if df < 1:
        return nan
    se = np.sqrt(max((r @ r) / df * np.linalg.inv(X.T @ X)[1, 1], 0))
    if se <= 0:
        return nan
    est = beta[1]                       # the slope on x, which is the effect asked for
    return est, se, est / se, float(t_pvalue(est / se, df)), N
