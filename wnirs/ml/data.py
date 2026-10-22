"""Turn preprocessed recordings into fixed-length windows for learning.

The unit of data is a *window*: ``(n_signals, n_samples)`` cut from one
recording, where the signals are HbO and HbR of every long channel stacked
(2 x n_channels). Windows carry the labels the downstream tasks need
(subject, session, task, per-channel quality) so that one cached array set
serves every experiment.

Two rules shape everything here:

* **Splits are by session, never by window.** Neighbouring windows of one
  recording share slow drifts and the same optode placement; putting them on
  both sides of a split measures memorisation, not generalisation.
* **Bad channels are masked, not dropped.** Wearable recordings lose
  different channels on different days. Dropping them would change the input
  shape per session; instead they are zeroed and a mask is carried alongside,
  and the model is trained with the same masking.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

TASKS = ("Flanker", "GoNogo", "NBack", "Rest")


@dataclass
class WindowSet:
    """A cached set of windows and their labels.

    x        (n_windows, n_signals, n_samples) float32, z-scored per recording
    mask     (n_windows, n_signals) bool, True where the channel is usable
    quality  (n_windows, n_signals) float32 in [0, 1], fraction of QC gates passed
             (NaN where no QC was computed)
    meta     one row per window: subject, session, task, recording, start_s
    fs       sampling rate after resampling
    """
    x: np.ndarray
    mask: np.ndarray
    quality: np.ndarray
    meta: pd.DataFrame
    fs: float

    def __len__(self) -> int:
        return len(self.x)

    def subset(self, idx: np.ndarray) -> "WindowSet":
        return WindowSet(self.x[idx], self.mask[idx], self.quality[idx],
                         self.meta.iloc[idx].reset_index(drop=True), self.fs)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez_compressed(path, x=self.x, mask=self.mask, quality=self.quality, fs=self.fs)
        self.meta.to_csv(path.with_suffix(".csv"), index=False)

    @classmethod
    def load(cls, path: str | Path) -> "WindowSet":
        path = Path(path)
        z = np.load(path if path.suffix == ".npz" else path.with_suffix(".npz"))
        meta = pd.read_csv(path.with_suffix(".csv"), dtype={"subject": str, "session": str})
        return cls(z["x"], z["mask"], z["quality"], meta, float(z["fs"]))

    @classmethod
    def concat(cls, sets: Sequence["WindowSet"]) -> "WindowSet":
        sets = [s for s in sets if len(s)]
        if not sets:
            raise ValueError("no windows to concatenate")
        fs = {s.fs for s in sets}
        if len(fs) != 1:
            raise ValueError(f"window sets were resampled to different rates: {sorted(fs)}")
        return cls(np.concatenate([s.x for s in sets]),
                   np.concatenate([s.mask for s in sets]),
                   np.concatenate([s.quality for s in sets]),
                   pd.concat([s.meta for s in sets], ignore_index=True), fs.pop())


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resampling along the last axis to a common rate.

    The tablet logs at a nominal ~7.1 Hz that drifts slightly between
    sessions; models need one rate. A rational approximation of the ratio
    keeps the filter short.
    """
    if np.isclose(fs_in, fs_out):
        return x
    from fractions import Fraction
    ratio = Fraction(fs_out / fs_in).limit_denominator(200)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)


def windows_from_arrays(
    hbo: np.ndarray,
    hbr: np.ndarray,
    usable: np.ndarray,
    fs: float,
    *,
    quality: np.ndarray | None = None,
    meta: dict,
    fs_out: float = 5.0,
    win_s: float = 30.0,
    hop_s: float = 15.0,
) -> WindowSet:
    """Cut one recording into overlapping windows.

    Parameters
    ----------
    hbo, hbr : (n_samples, n_channels) concentration changes, long channels only.
    usable   : (n_channels,) bool channel mask from preprocessing / QC.
    quality  : (n_channels,) optional per-channel QC score in [0, 1].
    meta     : labels copied onto every window (subject, session, task, recording).

    Each signal is z-scored over the whole recording *before* windowing, using
    only usable channels' own statistics. That removes the arbitrary scale
    differences between sessions (optode coupling, DPF assumptions) while
    keeping within-recording dynamics intact.
    """
    sig = np.concatenate([hbo, hbr], axis=1).T.astype(np.float64)       # (2C, T)
    m = np.concatenate([usable, usable]).astype(bool)
    sig = resample(sig, fs, fs_out)
    sig = np.nan_to_num(sig, nan=0.0)
    mu = sig.mean(axis=1, keepdims=True)
    sd = sig.std(axis=1, keepdims=True)
    sig = np.where(sd > 0, (sig - mu) / np.where(sd > 0, sd, 1), 0.0)
    sig[~m] = 0.0

    q = np.full(m.shape, np.nan, np.float32)
    if quality is not None:
        q[:] = np.concatenate([quality, quality])

    n = int(round(win_s * fs_out))
    hop = int(round(hop_s * fs_out))
    starts = np.arange(0, sig.shape[1] - n + 1, hop)
    if len(starts) == 0:
        return WindowSet(np.empty((0, len(m), n), np.float32), np.empty((0, len(m)), bool),
                         np.empty((0, len(m)), np.float32), pd.DataFrame(), fs_out)
    x = np.stack([sig[:, s:s + n] for s in starts]).astype(np.float32)
    rows = pd.DataFrame([{**meta, "start_s": s / fs_out} for s in starts])
    return WindowSet(x, np.tile(m, (len(starts), 1)), np.tile(q, (len(starts), 1)), rows, fs_out)


def session_split(
    meta: pd.DataFrame,
    *,
    test_frac: float = 0.25,
    by: str = "within_subject",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train/test window indices with whole sessions held out.

    by="within_subject" : each subject contributes sessions to both sides
                          (the precision-imaging question: does a model of
                          *this* person transfer to their future sessions?)
    by="subject"        : leave whole subjects out (cross-person transfer).
    Test sessions are the *latest* ones within each subject when
    by="within_subject", because a deployed model only ever sees the future.
    """
    rng = np.random.default_rng(seed)
    key = meta["subject"].astype(str) + "|" + meta["session"].astype(str)
    test = np.zeros(len(meta), bool)
    if by == "within_subject":
        for _, g in meta.groupby("subject"):
            ses = sorted(g["session"].astype(str).unique(), key=_session_order)
            k = max(1, int(round(test_frac * len(ses))))
            held = {f"{g['subject'].iloc[0]}|{s}" for s in ses[-k:]}
            test |= key.isin(held).to_numpy()
    elif by == "subject":
        subs = meta["subject"].unique()
        k = max(1, int(round(test_frac * len(subs))))
        test = meta["subject"].isin(rng.choice(subs, k, replace=False)).to_numpy()
    else:
        raise ValueError(f"unknown split {by!r}")
    return np.flatnonzero(~test), np.flatnonzero(test)


def _session_order(s: str) -> float:
    digits = "".join(ch for ch in s if ch.isdigit())
    return float(digits) if digits else float("inf")


def label_codes(values: Iterable, classes: Sequence | None = None) -> tuple[np.ndarray, list]:
    """Integer-code labels with a fixed, reproducible class order."""
    values = list(values)
    classes = list(classes) if classes is not None else sorted(set(values))
    lut = {c: i for i, c in enumerate(classes)}
    return np.array([lut.get(v, -1) for v in values]), classes
