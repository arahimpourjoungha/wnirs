"""Dense-sampling reliability and individual specificity.

Implements the three analyses of Rahimpour Jounghani et al. (2025), npj
Digital Medicine 8:271, "Wearable fNIRS platform for dense sampling and
precision functional neuroimaging":

1. Reliability curve -- how much data does one person need before their map
   stops changing? Single-session reliability in the paper was as low as 0.25
   and rose to 0.92 once several sessions were combined. A single session is
   not a measurement of a person, and the curve says how many are needed.
2. Fingerprinting -- are two sessions of one person more alike than two
   sessions of different people? Without it, a reliable measure could be
   reliably measuring something common to everybody.
3. ICC across sessions -- ICC(2,1) of two disjoint k-session averages per
   subject, i.e. the reliability of a k-session average measured directly
   rather than extrapolated with Spearman-Brown. This is not the Shrout-Fleiss
   ICC(2,k) formula, which is a different calculation.

Channel selection is by finiteness, which is NOT the same criterion as passing
quality control: a channel at the detector's dark floor produces finite betas
and is kept even though QC failed it. Exclude failing channels upstream (so
they arrive as NaN) when the two outputs must describe the same channels.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ._stats import EPS, corr_matrix, is_outlier_mean, nanmean, percentile, round_half_away, unique_stable
from .sessions import SessionMeasure, stack

CURVE_COLUMNS = ["Subject", "Task", "nSessions", "meanR", "sdR", "ciLo", "ciHi", "nIter", "nChannels",
                 "meanOverlap", "nReference", "nullMean", "nullLo", "nullHi", "pPerm"]
ICC_COLUMNS = ["Task", "nSessions", "ICC", "sdICC", "iqrLo", "iqrHi"]


@dataclass
class Fingerprint:
    """Within- versus between-person similarity of session maps.

    Every field exists even when fingerprinting did not run; scalars are then
    NaN, which propagates and fails ``isfinite`` instead of raising or reading
    as a real number.
    """
    similarity: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    within: np.ndarray = field(default_factory=lambda: np.empty(0))
    between: np.ndarray = field(default_factory=lambda: np.empty(0))
    mean_within: float = np.nan
    mean_between: float = np.nan
    separation: float = np.nan          # d' between the two similarity distributions
    id_accuracy: float = np.nan
    n_compared: int = 0
    chance: float = np.nan
    pval: float = np.nan
    n_channels_per_task: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["Task", "nChannels"]))


@dataclass
class ReliabilityResult:
    curve: pd.DataFrame
    fingerprint: Fingerprint
    icc: pd.DataFrame
    channels: np.ndarray            # elements finite in every session (information only)
    n_sessions: int


def reliability(sessions: Sequence[SessionMeasure], *, analyses: Iterable[str] = ("curve", "fingerprint", "icc"),
                n_iter: int = 500, common_only: bool | float = True, kind: str = "pearson",
                max_sessions: int | None = None, ref_frac: float = 0.5, null_between: bool = False,
                fisher: bool = True, rm_outliers: bool = True, n_perm: int = 2000,
                rng: np.random.Generator | int | None = None) -> ReliabilityResult:
    """Reliability curve, fingerprinting and ICC for a set of session maps.

    Parameters
    ----------
    sessions : sequence of SessionMeasure
        One per subject-session; connectivity matrices or per-channel vectors.
    analyses : iterable of {'curve', 'fingerprint', 'icc'}
    n_iter : int
        Random splits per data quantity (500 as published).
    common_only : bool or float
        Which elements a comparison uses: True = finite in every session it
        involves; False = finite in at least half; a fraction f = finite in at
        least f of them. Use a fraction once failing channels are excluded,
        since the surviving set then differs session to session and the
        strict intersection is usually empty.
    kind : {'pearson', 'spearman'}
    max_sessions : int, optional
        Largest k on the curve. Default half the subject's sessions, so the two
        halves never overlap.
    ref_frac : float
        Share of sessions in the fixed reference map (0.5 as published). A
        larger reference is steadier but leaves fewer sessions to draw k from;
        once those run out k is filled from the reference itself, and the
        curve reports ``meanOverlap`` so the resulting inflation can be shown.
    null_between : bool
        Add a between-person permutation null to the curve: the same
        reference, but the k sessions drawn from other people doing the task.
        Connectivity maps share much structure across everyone, so r is well
        above zero even between people; this asks whether a person's own
        sessions reproduce their map better than other people's do. Adds
        nullMean, nullLo/nullHi (2.5/97.5th percentiles) and
        pPerm = (1 + #{null r >= observed mean r}) / (n_null + 1).
    fisher : bool
        Fisher-z correlation measures before averaging and correlating;
        averaging raw r is biased towards zero.
    rm_outliers : bool
        Drop jointly outlying elements (> 3 SD in either map) before each
        correlation, so a handful of extreme edges cannot dominate.
    n_perm : int
        Label permutations for the fingerprinting p value.
    rng : numpy.random.Generator or seed
    """
    analyses = set(analyses)
    if len(sessions) < 4:
        raise ValueError(f"Reliability needs several sessions to say anything; got {len(sessions)}. "
                         "The published design used 10 per person.")
    rng = np.random.default_rng(rng)
    st = stack(sessions)
    V = st.V
    if fisher and st.is_matrix and _is_correlation(np.asarray(sessions[0].measure)):
        V = fisher_clamped(V)
    keep = np.all(np.isfinite(V), axis=1)
    if common_only is True and not keep.any():
        warnings.warn("No element is finite in every session of every subject, so no global common set "
                      "exists. Each analysis restricts to the elements its own comparison needs instead.",
                      stacklevel=2)
    opts = dict(common_only=common_only, kind=kind, rm_outliers=rm_outliers)
    curve = (reliability_curve(V, st.subject, st.task, n_iter=n_iter, max_sessions=max_sessions,
                               ref_frac=ref_frac, null_between=null_between, rng=rng, **opts)
             if "curve" in analyses else pd.DataFrame(columns=CURVE_COLUMNS))
    fp = (fingerprint(V, st.subject, st.task, common_only=common_only, kind=kind, n_perm=n_perm, rng=rng)
          if "fingerprint" in analyses else Fingerprint())
    icc = (icc_curve(V, st.subject, st.task, common_only=common_only, max_sessions=max_sessions, rng=rng)
           if "icc" in analyses else pd.DataFrame(columns=ICC_COLUMNS))
    return ReliabilityResult(curve, fp, icc, keep, len(sessions))


def fisher_clamped(V: np.ndarray, limit: float = 1e-12) -> np.ndarray:
    """Fisher z of correlation values, NaN kept NaN (see ``connectivity.fisher_z``)."""
    Z = np.full(V.shape, np.nan)
    ok = np.isfinite(V)
    Z[ok] = np.arctanh(np.clip(V[ok], -1 + limit, 1 - limit))
    return Z


def _is_correlation(M: np.ndarray) -> bool:
    """A square matrix whose finite off-diagonal entries all lie in [-1, 1]."""
    off = M[~np.eye(M.shape[0], dtype=bool)]
    off = off[np.isfinite(off)]
    return off.size > 0 and bool(np.all(np.abs(off) <= 1 + 1e-9))


def rows_finite(V: np.ndarray, common_only: bool | float = True) -> np.ndarray:
    """Elements usable across the columns given (see ``common_only`` in :func:`reliability`)."""
    if isinstance(common_only, (bool, np.bool_)):
        f = 1.0 if common_only else 0.5
    else:
        f = float(np.clip(common_only, 0, 1)) or 0.5
    finite = np.isfinite(V)
    return finite.all(axis=1) if f >= 1 else finite.mean(axis=1) >= f


# ---------------------------------------------------------------------------
# 1. reliability curve
# ---------------------------------------------------------------------------
def reliability_curve(V: np.ndarray, subject: np.ndarray, task: np.ndarray, *, n_iter: int = 500,
                      common_only: bool | float = True, kind: str = "pearson", max_sessions: int | None = None,
                      ref_frac: float = 0.5, null_between: bool = False, rm_outliers: bool = True,
                      rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Split-half reliability against the number of sessions, per subject and task.

    Sessions are split at random into a fixed reference and the rest; k of the
    rest are averaged and correlated with the reference average. All
    ``n_iter`` splits for one k are drawn and evaluated at once.
    """
    rng = np.random.default_rng(rng)
    rows = []
    for tk in unique_stable(task):
        for sb in unique_stable(subject):
            idx = np.flatnonzero((subject == sb) & (task == tk))
            n_s = idx.size
            if n_s < 4:
                continue
            # Elements this subject kept in all of THEIR sessions: another
            # subject's dead channel says nothing about whether this map repeats.
            sel = rows_finite(V[:, idx], common_only)
            if sel.sum() < 3:
                warnings.warn(f"{sb} / {tk}: only {sel.sum()} element(s) survive in all {n_s} sessions -- too "
                              "few to correlate two maps. Skipped.", stacklevel=2)
                continue
            Vs = V[sel]
            half = n_s // 2
            max_k = half if max_sessions is None else min(max_sessions, half)
            n_ref = max(1, min(n_s - 1, round_half_away(ref_frac * n_s)))
            n_out = n_s - n_ref
            others = np.flatnonzero((task == tk) & (subject != sb))
            for k in range(1, max_k + 1):
                # Any block of a random ordering is a uniform random subset, so
                # the reference, the k sessions and (when the non-reference
                # sessions run out) the overlap all come from one shuffle.
                perm = _permute_rows(idx, n_iter, rng)
                ref = perm[:, :n_ref]
                pick = perm[:, n_ref:n_ref + k] if n_out >= k else np.hstack((perm[:, n_ref:], ref[:, :k - n_out]))
                overlap = max(0, k - n_out)
                r = _split_corr(Vs, ref, pick, kind, rm_outliers)
                null = np.empty(0)
                if null_between and others.size >= k:
                    ref0 = _permute_rows(idx, n_iter, rng)[:, :n_ref]
                    null = _split_corr(Vs, ref0, _permute_rows(others, n_iter, rng)[:, :k], kind, rm_outliers)
                    null = null[np.isfinite(null)]
                r = r[np.isfinite(r)]
                if r.size == 0:
                    continue
                stats_null = ([null.mean(), percentile(null, 2.5), percentile(null, 97.5),
                               (1 + np.sum(null >= r.mean())) / (null.size + 1)] if null.size else [np.nan] * 4)
                rows.append([sb, tk, k, r.mean(), np.std(r, ddof=1) if r.size > 1 else 0.0,
                             percentile(r, 2.5), percentile(r, 97.5), r.size, int(sel.sum()), float(overlap), n_ref,
                             *stats_null])
    return pd.DataFrame(rows, columns=CURVE_COLUMNS)


def _permute_rows(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` independent random orderings of ``values``, one per row."""
    return rng.permuted(np.tile(values, (n, 1)), axis=1)


def _split_corr(Vs: np.ndarray, ref: np.ndarray, pick: np.ndarray, kind: str, rm_outliers: bool) -> np.ndarray:
    """Correlation between the reference average and the k-session average, per split."""
    return column_corr(nanmean(Vs[:, ref], axis=2), nanmean(Vs[:, pick], axis=2), kind, rm_outliers)


def column_corr(A: np.ndarray, B: np.ndarray, kind: str = "pearson", rm_outliers: bool = True) -> np.ndarray:
    """Correlation between matching columns of A and B (elements x maps).

    Elements non-finite in either map are dropped. With ``rm_outliers`` and at
    least 10 elements, elements more than 3 SD out in either map are dropped
    too, unless that would leave fewer than half (or fewer than 3).
    """
    ok = np.isfinite(A) & np.isfinite(B)
    n_ok = ok.sum(axis=0)
    use = ok
    if rm_outliers:
        Am, Bm = np.where(ok, A, np.nan), np.where(ok, B, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            clean = ok & ~(is_outlier_mean(Am) | is_outlier_mean(Bm))
        n_clean = clean.sum(axis=0)
        trim = (n_ok >= 10) & (n_clean >= np.maximum(3, 0.5 * n_ok))
        use = np.where(trim, clean, ok)
    if kind.lower() == "spearman":
        A = stats.rankdata(np.where(use, A, np.nan), axis=0, nan_policy="omit")
        B = stats.rankdata(np.where(use, B, np.nan), axis=0, nan_policy="omit")
    r = _masked_pearson(A, B, use)
    r[use.sum(axis=0) < 3] = np.nan
    return r


def _masked_pearson(A: np.ndarray, B: np.ndarray, M: np.ndarray) -> np.ndarray:
    n = M.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        da = np.where(M, A - np.where(M, A, 0).sum(0) / n, 0.0)
        db = np.where(M, B - np.where(M, B, 0).sum(0) / n, 0.0)
        r = (da * db).sum(0) / np.sqrt((da**2).sum(0) * (db**2).sum(0))
    return np.clip(r, -1, 1)


# ---------------------------------------------------------------------------
# 2. fingerprinting
# ---------------------------------------------------------------------------
def fingerprint(V: np.ndarray, subject: np.ndarray, task: np.ndarray, *, common_only: bool | float = True,
                kind: str = "pearson", n_perm: int = 2000, rng: np.random.Generator | None = None) -> Fingerprint:
    """Within- versus between-person similarity, identification accuracy and its chance level.

    Needs at least two people and refuses one: with a single subject every
    nearest neighbour is trivially the same person, giving 100% accuracy
    against 100% chance and a small p from permuting one label -- a
    spectacular-looking result that means nothing.

    The similarity matrix is built per task on that task's own common
    elements, since only same-task pairs are ever compared; cross-task cells
    stay NaN.

    Chance is NOT 1/(n_subjects - 1). The nearest neighbour is picked among all
    other sessions, and a subject with ten of them is ten times easier to hit
    by luck than one with a single session. So chance is the proportion of each
    session's candidate pool that belongs to the same person, averaged over
    sessions. With 2 subjects x 4 sessions the subject-counting formula says
    100%; the true value is 3/7.
    """
    rng = np.random.default_rng(rng)
    n = V.shape[1]
    n_subj = np.unique(subject).size
    if n_subj < 2:
        raise ValueError("Fingerprinting compares sessions of the same person with sessions of DIFFERENT people, "
                         f"so it needs at least 2 subjects; got {n_subj}. With one person it would report 100% "
                         "identification accuracy against a chance level of 100%.")
    S = np.full((n, n), np.nan)
    tasks = unique_stable(task)
    n_used = np.zeros(tasks.size, int)
    for t, tk in enumerate(tasks):
        cols = np.flatnonzero(task == tk)
        if cols.size < 2:
            continue
        sel = rows_finite(V[:, cols], common_only)
        n_used[t] = sel.sum()
        if sel.sum() < 3:
            warnings.warn(f"{tk}: only {sel.sum()} element(s) are shared by every session of every subject, so "
                          "this task cannot enter the fingerprinting.", stacklevel=2)
            continue
        S[np.ix_(cols, cols)] = corr_matrix(V[sel][:, cols], kind)
    if np.all(np.isnan(S)):
        raise ValueError("No task has enough elements shared across every session of every subject for "
                         "fingerprinting.")
    np.fill_diagonal(S, np.nan)

    same_subj = subject[:, None] == subject[None, :]
    # Like with like: a rest session and an n-back session of the same person
    # differ for reasons that have nothing to do with identity.
    usable = (task[:, None] == task[None, :]) & ~np.eye(n, dtype=bool) & np.isfinite(S)
    within, between = S[usable & same_subj], S[usable & ~same_subj]
    m_w, m_b = _mean(within), _mean(between)
    pooled_sd = np.sqrt((_var(within) + _var(between)) / 2)

    has = usable.any(axis=1)
    best = np.argmax(np.where(usable, S, -np.inf), axis=1)
    hits = same_subj[np.arange(n), best] & has
    chance = (usable & same_subj).sum(axis=1)[has] / usable.sum(axis=1)[has]

    # Permuting subject labels keeps the similarity structure and destroys
    # only identity, which is exactly the null being tested.
    null = _label_permutation_null(np.where(usable, S, 0.0), usable.astype(float), subject, n_perm, rng)
    return Fingerprint(
        similarity=S, within=within, between=between, mean_within=m_w, mean_between=m_b,
        separation=(m_w - m_b) / max(pooled_sd, EPS), id_accuracy=hits.sum() / max(has.sum(), 1),
        n_compared=int(has.sum()), chance=float(chance.mean()) if chance.size else np.nan,
        pval=(np.sum(null >= m_w - m_b) + 1) / (n_perm + 1),
        n_channels_per_task=pd.DataFrame({"Task": tasks, "nChannels": n_used}))


def _label_permutation_null(W: np.ndarray, U: np.ndarray, subject: np.ndarray, n_perm: int,
                            rng: np.random.Generator, chunk: int = 250) -> np.ndarray:
    """mean(within) - mean(between) under shuffled subject labels.

    Each permutation's same-subject mask is P P' for a one-hot label matrix P,
    so the within sums are traces computed in one einsum per chunk.
    """
    codes = np.unique(subject, return_inverse=True)[1]
    total_w, total_u = W.sum(), U.sum()
    out = np.empty(n_perm)
    for start in range(0, n_perm, chunk):
        m = min(chunk, n_perm - start)
        perm = rng.permuted(np.tile(codes, (m, 1)), axis=1)
        P = np.eye(codes.max() + 1)[perm]                         # (m, n, n_subj)
        sw = np.einsum("pis,ij,pjs->p", P, W, P)
        nw = np.einsum("pis,ij,pjs->p", P, U, P)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start:start + m] = sw / nw - (total_w - sw) / (total_u - nw)
    return out


def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else np.nan


def _var(x: np.ndarray) -> float:
    return float(np.var(x, ddof=1)) if x.size > 1 else (0.0 if x.size else np.nan)


# ---------------------------------------------------------------------------
# 3. ICC of k-session averages
# ---------------------------------------------------------------------------
def icc_curve(V: np.ndarray, subject: np.ndarray, task: np.ndarray, *, common_only: bool | float = True,
              max_sessions: int | None = None, n_rep: int = 20,
              rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Test-retest ICC(2,1) of a k-session average, against k, per task.

    Targets are SUBJECTS and the repeated measurements are SESSIONS. With
    channels as targets the ICC would ask whether channels rank the same
    across subjects -- a question about the group map -- and it goes negative
    whenever subjects do not share a spatial pattern. Each subject's sessions
    are split into two disjoint sets of k, each averaged; the two averages are
    two independent measurements of that subject. Computed per element (a
    property of a channel, not of the montage) and summarised by the median
    and interquartile range across elements, averaged over ``n_rep`` splits.
    """
    rng = np.random.default_rng(rng)
    rows = []
    for tk in unique_stable(task):
        in_task = task == tk
        subs = unique_stable(subject[in_task])
        if subs.size < 3:
            continue
        sel = rows_finite(V[:, in_task], common_only)
        if sel.sum() < 3:
            warnings.warn(f"{tk}: only {sel.sum()} element(s) common to every session -- ICC skipped.",
                          stacklevel=2)
            continue
        Vt = V[sel]
        members = [np.flatnonzero(in_task & (subject == s)) for s in subs]
        max_k = min(m.size for m in members) // 2
        if max_sessions is not None:
            max_k = min(max_k, max_sessions)
        for k in range(1, max_k + 1):
            med, lo, hi = np.empty(n_rep), np.empty(n_rep), np.empty(n_rep)
            for rep in range(n_rep):
                picks = [rng.permutation(m)[: 2 * k] for m in members]
                A = np.column_stack([nanmean(Vt[:, p[:k]], axis=1) for p in picks])
                B = np.column_stack([nanmean(Vt[:, p[k:]], axis=1) for p in picks])
                per_elem = icc21_batch(np.stack((A, B), axis=2))
                med[rep] = np.nanmedian(per_elem) if np.isfinite(per_elem).any() else np.nan
                lo[rep], hi[rep] = percentile(per_elem, 25), percentile(per_elem, 75)
            rows.append([tk, k, np.nanmean(med), np.nanstd(med, ddof=1), np.nanmean(lo), np.nanmean(hi)])
    return pd.DataFrame(rows, columns=ICC_COLUMNS)


def icc21(M: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measures.

    ``M`` is targets x repeated measurements; targets with any missing value
    are dropped. Absolute agreement keeps the rater term in the denominator,
    so a systematic offset between sessions counts against reliability rather
    than being removed as in the consistency form.
    """
    return float(icc21_batch(np.asarray(M, float)[None])[0])


def icc21_batch(M: np.ndarray) -> np.ndarray:
    """:func:`icc21` for a stack of (n_items, n_targets, n_raters) matrices at once."""
    valid = np.all(np.isfinite(M), axis=2)
    n = valid.sum(axis=1).astype(float)
    k = M.shape[2]
    Mz = np.where(valid[..., None], M, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        grand = Mz.sum(axis=(1, 2)) / (n * k)
        row_m = Mz.mean(axis=2)
        col_m = Mz.sum(axis=1) / n[:, None]
        ssr = k * np.sum(valid * (row_m - grand[:, None]) ** 2, axis=1)
        ssc = n * np.sum((col_m - grand[:, None]) ** 2, axis=1)
        sst = np.sum(np.where(valid[..., None], M - grand[:, None, None], 0.0) ** 2, axis=(1, 2))
        mse = (sst - ssr - ssc) / ((n - 1) * (k - 1))
        msr, msc = ssr / (n - 1), ssc / (k - 1)
        den = msr + (k - 1) * mse + k * (msc - mse) / n
        v = (msr - mse) / den
    bad = (n < 2) | (k < 2) | ~np.isfinite(mse) | (mse <= 0) | ~(den > 0)
    return np.where(bad, np.nan, np.clip(v, -1, 1))
