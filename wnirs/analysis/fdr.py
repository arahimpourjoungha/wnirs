"""Benjamini-Hochberg false discovery rate."""
from __future__ import annotations

import numpy as np


def fdr_bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values (q values).

    Thresholding q at 0.05 controls the expected proportion of false positives
    among the channels called significant. With 20-30 channels this is much
    less punishing than Bonferroni and is the usual correction in fNIRS.

    Parameters
    ----------
    p : array_like
        p values of any shape. NaNs come back as NaN and are left out of the
        correction, which is what is wanted for channels that were never tested.

    Returns
    -------
    numpy.ndarray
        q values, same shape as ``p``.
    """
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok], kind="stable")
    adj = p[ok][order] * m / np.arange(1, m + 1)
    # Monotone from the largest p downwards, then capped at 1.
    adj = np.minimum(1.0, np.minimum.accumulate(adj[::-1])[::-1])
    qv = np.empty(m)
    qv[order] = adj
    q[ok] = qv
    return q
