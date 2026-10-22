"""Per-session measures, the input to the dense-sampling analyses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SessionMeasure:
    """One subject-session's map.

    Parameters
    ----------
    subject, session : str
    measure : numpy.ndarray
        Either a C x C connectivity matrix (only its upper triangle is used,
        since it is symmetric and both halves would double-count every pair)
        or a length-C vector of per-channel statistics.
    task : str
        Analyses run per task.
    order : float, optional
        Explicit chronological position. Wins over ``datetime`` and over a
        numeric session label wherever time order matters.
    datetime : numpy.datetime64, optional
    """
    subject: str
    session: str
    measure: np.ndarray
    task: str = "all"
    order: float | None = None
    datetime: np.datetime64 | None = None


@dataclass
class Stacked:
    """Sessions flattened to one column each, with their labels."""
    V: np.ndarray            # (n_elements, n_sessions)
    subject: np.ndarray
    session: np.ndarray
    task: np.ndarray
    is_matrix: bool


def upper_triangle(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Row/column indices of the strict upper triangle, in column-major order."""
    j, i = np.tril_indices(n, -1)
    return i, j


def stack(sessions: Sequence[SessionMeasure]) -> Stacked:
    """One column per session holding its map as a vector."""
    first = np.asarray(sessions[0].measure)
    is_matrix = first.ndim == 2 and first.shape[0] == first.shape[1] and first.shape[0] > 1
    if is_matrix:
        i, j = upper_triangle(first.shape[0])
        V = np.column_stack([np.asarray(s.measure, float)[i, j] for s in sessions])
    else:
        V = np.column_stack([np.asarray(s.measure, float).ravel() for s in sessions])
    return Stacked(V=V, subject=np.array([str(s.subject) for s in sessions]),
                   session=np.array([str(s.session) for s in sessions]),
                   task=np.array([str(s.task) for s in sessions]), is_matrix=is_matrix)


def chronological(sessions: Sequence[SessionMeasure], *, strict: bool) -> np.ndarray:
    """Indices that put ``sessions`` in time order.

    ``order`` wins, then ``datetime``, then a numeric session label. When none
    applies, ``strict`` raises (for analyses where order is the whole point)
    and otherwise the arrival order is kept, since a text label has no
    defensible ordering.
    """
    if all(s.order is not None for s in sessions):
        return np.argsort([s.order for s in sessions], kind="stable")
    if all(s.datetime is not None for s in sessions):
        return np.argsort(np.array([s.datetime for s in sessions], dtype="datetime64[ns]"), kind="stable")
    num = session_numbers([s.session for s in sessions])
    if np.all(np.isfinite(num)):
        return np.argsort(num, kind="stable")
    if strict:
        raise ValueError("These sessions cannot be put in time order: the labels are not numeric and neither "
                         "order nor datetime is set. Ordering is the whole analysis, so this is refused rather "
                         "than guessed.")
    return np.arange(len(sessions))


def session_numbers(labels) -> np.ndarray:
    """Session labels as numbers, NaN where a label is not numeric."""
    out = np.full(len(labels), np.nan)
    for k, lab in enumerate(labels):
        try:
            out[k] = float(lab)
        except (TypeError, ValueError):
            pass
    return out
