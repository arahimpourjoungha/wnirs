"""Stimulus-locked average response per condition and channel."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..types import Hb
from ._stats import nanmean, round_half_away


@dataclass
class ConditionAverage:
    """Block average of one condition. Arrays are samples x channels (x epochs)."""
    name: str
    time: np.ndarray            # (n_samples,), zero at onset
    duration: float
    hbo: np.ndarray
    hbr: np.ndarray
    sem_hbo: np.ndarray
    sem_hbr: np.ndarray
    n_trials: np.ndarray        # (n_channels,): channels can differ when epochs are NaN
    epochs_hbo: np.ndarray      # (n_samples, n_channels, n_epochs)
    epochs_hbr: np.ndarray


@dataclass
class BlockAverage:
    subject: str
    task: str
    layout: pd.DataFrame
    used_display_copy: bool
    conditions: list[ConditionAverage] = field(default_factory=list)


def block_average(hb: Hb, *, pre: float = 10.0, post: float = 10.0, duration: float | None = None,
                  baseline: tuple[float, float] | None = None, baseline_correct: bool = True,
                  reject: float = np.inf) -> BlockAverage:
    """Average epochs locked to each block onset.

    Block averages are read by eye, so the low-passed display copy is used
    when preprocessing made one; the statistics never touch it. Epochs that
    would run off either end of the recording are dropped, which is why
    ``n_trials`` can be smaller than the number of blocks.

    Parameters
    ----------
    hb : Hb
    pre : float
        Seconds before onset.
    post : float
        Seconds after the block ends.
    duration : float, optional
        Block length in seconds. Default: the median recorded duration of the
        condition's blocks, so every epoch is the same length.
    baseline : (t1, t2), optional
        Window in seconds relative to onset whose mean is subtracted from each
        epoch before averaging. Default ``(-pre, 0)``.
    baseline_correct : bool
        False skips baseline correction.
    reject : float
        Drop channel-epochs whose HbO peak-to-peak exceeds this (micromolar).
    """
    baseline = (-pre, 0.0) if baseline is None else baseline
    used_display = hb.hbo_display is not None and np.size(hb.hbo_display) > 0
    src_o, src_r = (hb.hbo_display, hb.hbr_display) if used_display else (hb.hbo, hb.hbr)
    fs, t0, n_samples_total = hb.fs, hb.time[0], src_o.shape[0]
    out = BlockAverage(hb.subject, hb.task, hb.layout, used_display)
    label = f"{hb.subject} / {hb.task}"

    for name, onsets, durs in zip(hb.conditions.name, hb.conditions.onsets, hb.conditions.durations):
        onsets = np.asarray(onsets, float) - t0
        if onsets.size == 0:
            continue
        dur = float(np.median(durs)) if duration is None else duration
        n_pre = round_half_away(pre * fs)
        n_post = round_half_away((dur + post) * fs)
        lags = np.arange(-n_pre, n_post + 1)
        t_epoch = lags / fs

        start = round_half_away(onsets * fs)
        start = np.atleast_1d(start)
        fits = (start - n_pre >= 0) & (start + n_post <= n_samples_total - 1)
        if not fits.any():
            warnings.warn(f'{label}: no block of "{name}" fits a {pre + dur + post:g} s epoch -- skipped.',
                          stacklevel=2)
            continue
        if not fits.all():
            warnings.warn(f"{label} / {name}: {np.sum(~fits)} of {fits.size} blocks run off the end of the "
                          "recording and were dropped.", stacklevel=2)
        idx = start[fits][None, :] + lags[:, None]                 # (n_samples, n_epochs)
        EO = np.moveaxis(src_o[idx], 2, 1)                          # (n_samples, n_ch, n_epochs)
        ER = np.moveaxis(src_r[idx], 2, 1)

        if baseline_correct:
            win = (t_epoch >= baseline[0]) & (t_epoch <= baseline[1])
            if win.any():
                EO = EO - nanmean(EO[win], axis=0)[None]
                ER = ER - nanmean(ER[win], axis=0)[None]

        if np.isfinite(reject):
            finite = np.isfinite(EO)
            p2p = np.max(np.where(finite, EO, -np.inf), axis=0) - np.min(np.where(finite, EO, np.inf), axis=0)
            drop = p2p > reject                                                  # (n_ch, n_epochs)
            EO = np.where(drop[None], np.nan, EO)
            ER = np.where(drop[None], np.nan, ER)
            if drop.any():
                warnings.warn(f"{label} / {name}: rejected {drop.sum()} channel-epochs over {reject:g} uM "
                              "peak-to-peak.", stacklevel=2)

        n_trials = np.sum(np.all(np.isfinite(EO), axis=0), axis=1)
        denom = np.sqrt(np.maximum(n_trials, 1))[None, :]
        out.conditions.append(ConditionAverage(
            name=str(name), time=t_epoch, duration=dur, hbo=nanmean(EO, axis=2), hbr=nanmean(ER, axis=2),
            sem_hbo=_nanstd(EO) / denom, sem_hbr=_nanstd(ER) / denom,
            n_trials=n_trials, epochs_hbo=EO, epochs_hbr=ER))

    if not out.conditions:
        warnings.warn(f"{label}: nothing could be block averaged.", stacklevel=2)
    return out


def _nanstd(E: np.ndarray) -> np.ndarray:
    """Sample SD over epochs ignoring NaN; 0 for a single epoch."""
    n = np.sum(np.isfinite(E), axis=2)
    dev = np.where(np.isfinite(E), E - nanmean(E, axis=2)[..., None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(np.sum(dev**2, axis=2) / np.maximum(n - 1, 1))
    return np.where(n > 0, sd, np.nan)
