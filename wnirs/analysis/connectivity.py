"""Channel-by-channel functional connectivity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ..types import Hb
from ._stats import EPS, channel_labels, corr_matrix, round_half_away
from .ar import arfit, whiten


@dataclass
class Connectivity:
    """Correlation matrix of one recording.

    ``R`` is NaN on the diagonal and on excluded channels; ``Z`` is its Fisher
    transform with NaN kept NaN. ``mean_r`` is the mean off-diagonal r, a
    one-number summary of global coupling.
    """
    R: np.ndarray
    Z: np.ndarray | None
    labels: np.ndarray
    layout: pd.DataFrame
    mean_r: float
    whitened: bool
    subject: str
    task: str


def fisher_z(R: np.ndarray) -> np.ndarray:
    """Fisher z of a correlation matrix, leaving missing entries missing.

    Only the finite entries are clamped. Clamping with NaN-ignoring min and
    max would turn every excluded channel into atanh(1-eps) ~ 18.7 -- one
    constant for every missing pair. With most channels excluded the matrix
    would collapse to that constant and reliability would score it 1.00, a
    perfect result manufactured from missing data.
    """
    Z = np.full(R.shape, np.nan)
    ok = np.isfinite(R)
    Z[ok] = np.arctanh(np.clip(R[ok], -1 + EPS, 1 - EPS))
    return Z


def connectivity(hb: Hb, *, chromophore: Literal["hbo", "hbr"] = "hbo",
                 kind: Literal["pearson", "spearman"] = "pearson", whiten_ar: bool = True,
                 window: tuple[float, float] | None = None, fisher: bool = True,
                 show_short: bool = False) -> Connectivity:
    """Functional connectivity between channels of one recording.

    Motion and systemic physiology inflate fNIRS connectivity badly. Run this
    on data that has been through short-separation regression and read the
    absolute values with caution: differences between conditions or groups
    are far more trustworthy than the raw numbers.

    Parameters
    ----------
    hb : Hb
        Preprocessed recording, already band-passed (filter it with
        :mod:`wnirs.preproc` first if a different band is wanted).
    chromophore : {'hbo', 'hbr'}
    kind : {'pearson', 'spearman'}
    whiten_ar : bool
        Prewhiten each channel with its own AR model (order by BIC up to
        4 fs, the GLM's rule) before correlating. Two channels sharing nothing
        but a slow drift correlate strongly, because autocorrelated series
        have far fewer independent samples than they appear to. Whitening
        removes the shared predictability and leaves co-fluctuation; it is
        what ``nirs.sFC.ar_corr`` does, and unwhitened matrices are inflated
        throughout.
    window : (t1, t2), optional
        Seconds from the start of the recording to restrict to, e.g. a rest
        period.
    fisher : bool
        Also return Fisher z.
    show_short : bool
        Include short channels. They share scalp signal with everything and
        inflate the whole matrix if left in.
    """
    X = getattr(hb, chromophore.lower())
    keep = np.ones(X.shape[1], bool) if show_short else hb.layout["Type"].to_numpy() == 1
    X = X[:, keep]
    layout = hb.layout.loc[keep].reset_index(drop=True)
    if window is not None:
        t = hb.time - hb.time[0]
        m = (t >= window[0]) & (t <= window[1])
        if m.sum() < 20:
            raise ValueError(f"The window {window[0]:g}-{window[1]:g} s leaves only {m.sum()} samples.")
        X = X[m]
    if whiten_ar:
        X = whiten_columns(X, hb.fs)

    R = corr_matrix(X, kind)
    np.fill_diagonal(R, np.nan)          # so the diagonal does not set the scale
    return Connectivity(R=R, Z=fisher_z(R) if fisher else None, labels=channel_labels(layout), layout=layout,
                        mean_r=float(np.nanmean(R)) if np.isfinite(R).any() else np.nan,
                        whitened=whiten_ar, subject=hb.subject, task=hb.task)


def whiten_columns(X: np.ndarray, fs: float) -> np.ndarray:
    """Remove each channel's own autocorrelation before correlating.

    The filter start-up depends on assumed zeros before the recording, so
    those samples are not measurements and are set to NaN.
    """
    Y = X.copy()
    pmax = max(1, round_half_away(4 * fs))
    idx = np.arange(X.shape[0])
    for c in range(X.shape[1]):
        v = X[:, c]
        ok = np.isfinite(v)
        if ok.sum() < 20:
            continue
        if not ok.all():
            v = interp1d(idx[ok], v[ok], fill_value="extrapolate")(idx)
        a = arfit(v, pmax)[0]
        if a.size < 2:
            continue
        w = whiten(a, v)
        w[: a.size] = np.nan
        Y[:, c] = w
    return Y
