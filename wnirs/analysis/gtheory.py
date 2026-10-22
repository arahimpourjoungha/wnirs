"""Generalizability theory: reliability of CHANGE as well as of people."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ._stats import EPS, unique_stable
from .mixed import MixedFit, fit_mixed, random_intercept
from .sessions import SessionMeasure, stack

COMPONENTS = ("person", "personSession", "channel", "personChannel", "residual")


@dataclass
class GTheory:
    """Variance decomposition and the two generalizability coefficients for one task.

    ``Rc``/``R1F`` are evaluated at k = n_elements (the whole-montage average),
    ``Rc_single``/``R1F_single`` at k = 1 (one channel). Read the single-channel
    figures first: both coefficients divide the residual by k, so at large k
    they saturate towards 1 whenever the numerator is non-zero at all, and two
    measures that behave completely differently can both report 0.99.
    """
    task: str
    var: dict[str, float]
    pct: dict[str, float]
    Rc: float
    R1F: float
    Rc_single: float
    R1F_single: float
    k: int
    projection: pd.DataFrame
    n_subjects: int
    n_sessions: int
    n_elements: int
    model: MixedFit


def gtheory(sessions: Sequence[SessionMeasure], *, task: str | None = None, max_elements: int = 500,
            k_projection: Sequence[int] | None = None, rng: np.random.Generator | int = 0) -> list[GTheory]:
    """Decompose session maps into person, change, channel and fingerprint variance.

    The reliability curve, fingerprinting and ICC all ask whether a measure can
    tell PEOPLE apart, and so count genuine within-person movement as error.
    Generalizability theory (Cronbach 1972; Bolger & Laurenceau 2013, ch. 7)
    separates the two questions. Each measurement is modelled as

        y(person, session, channel) = mu + p + ps + c + pc + e

    with sessions nested in person (subject A's session 3 and subject B's
    session 3 are not the same occasion), fitted by REML, and two coefficients
    formed over k channels:

        Rc  = var(ps) / (var(ps) + var(e)/k)                  within-person change
        R1F = (var(p) + var(pc)/k) / (var(p) + var(pc)/k + var(e)/k)
                                                              between-person differences

    They are independent: a measure can identify people flawlessly while being
    unable to track any of them over time. Rc is estimable from a single
    densely sampled subject; R1F needs two or more and is NaN otherwise --
    "could not estimate" and "was low" are different findings.

    Missing elements are NOT required to be complete: a mixed model does not
    need a balanced design, so any element seen in at least 3 sessions is kept
    and individual missing observations are dropped row by row.

    Parameters
    ----------
    sessions : sequence of SessionMeasure
    task : str, optional
        Restrict to one task; default each task separately.
    max_elements : int
        Random subsample of elements fitted, for speed. The components are
        population quantities, so a subsample changes precision, not the answer.
    k_projection : sequence of int, optional
        k values for the projection table (default 1, 3, ..., 41 and n_elements).
    rng : numpy.random.Generator or seed
        For the element subsample; seeded so results are reproducible.
    """
    if len(sessions) < 4:
        raise ValueError(f"Variance components need several sessions to be identified; got {len(sessions)}.")
    rng = np.random.default_rng(rng)
    st = stack(sessions)
    tasks = unique_stable(st.task)
    if task is not None:
        tasks = tasks[tasks == str(task)]
        if tasks.size == 0:
            raise ValueError(f'No sessions for task "{task}".')
    out = []
    for tk in tasks:
        sel = st.task == tk
        g = _one_task(st.V[:, sel], st.subject[sel], st.session[sel], tk, max_elements, k_projection, rng)
        if g is not None:
            out.append(g)
    if not out:
        raise ValueError("No task had enough complete data to identify the variance components.")
    return out


def coefficients(var: dict[str, float], k: float, n_subjects: int) -> tuple[float, float]:
    """Rc and R1F at k elements (R1F is NaN for a single subject)."""
    rc = var["personSession"] / max(var["personSession"] + var["residual"] / k, EPS)
    if n_subjects < 2:
        return rc, np.nan
    num = var["person"] + var["personChannel"] / k
    return rc, num / max(num + var["residual"] / k, EPS)


def _one_task(V, subject, session, task, max_elements, k_projection, rng) -> GTheory | None:
    min_sess = min(3, V.shape[1])
    keep = np.sum(np.isfinite(V), axis=1) >= min_sess
    if keep.sum() < 4:
        warnings.warn(f"{task}: only {keep.sum()} element(s) seen in {min_sess} or more sessions; skipped.",
                      stacklevel=3)
        return None
    V = V[keep]
    if V.shape[0] > max_elements:
        V = V[np.sort(rng.choice(V.shape[0], max_elements, replace=False))]
    n_elem, n_sess = V.shape
    n_subj = unique_stable(subject).size

    elem, sess = np.meshgrid(np.arange(n_elem), np.arange(n_sess), indexing="ij")
    y, elem, sess = V.ravel(order="F"), elem.ravel(order="F"), sess.ravel(order="F")
    subj = subject[sess]
    ok = np.isfinite(y)
    if ok.sum() < 20:
        warnings.warn(f"{task}: only {ok.sum()} finite observations; skipped.", stacklevel=3)
        return None
    y, elem, subj, sess_lab = y[ok], elem[ok], subj[ok], session[sess[ok]]
    subj_sess = np.char.add(np.char.add(subj, "|"), sess_lab)
    subj_elem = np.char.add(np.char.add(subj, "|"), elem.astype(str))

    # With one subject, person has no variance to estimate and person x channel
    # is the channel effect under another name; fitting both would hand back
    # an arbitrary split of one quantity between two labels.
    terms = [random_intercept("personSession", subj_sess), random_intercept("channel", elem)]
    if n_subj >= 2:
        terms = [random_intercept("person", subj), terms[0], terms[1], random_intercept("personChannel", subj_elem)]
    fit = fit_mixed(y, np.ones((y.size, 1)), terms, method="reml")

    var = {c: 0.0 for c in COMPONENTS}
    var.update({name: float(v[0, 0]) for name, v in fit.psi.items()})
    var["residual"] = fit.sigma2
    total = sum(var.values())
    pct = {c: 100 * v / max(total, EPS) for c, v in var.items()}

    rc, r1f = coefficients(var, n_elem, n_subj)
    rc1, r1f1 = coefficients(var, 1, n_subj)
    kk = (np.unique(np.r_[np.arange(1, min(n_elem, 41) + 1, 2), n_elem]) if k_projection is None
          else np.asarray(k_projection))
    proj = np.array([coefficients(var, k, n_subj) for k in kk])
    return GTheory(task=str(task), var=var, pct=pct, Rc=rc, R1F=r1f, Rc_single=rc1, R1F_single=r1f1,
                   k=n_elem, projection=pd.DataFrame({"k": kk, "Rc": proj[:, 0], "R1F": proj[:, 1]}),
                   n_subjects=n_subj, n_sessions=n_sess, n_elements=n_elem, model=fit)
