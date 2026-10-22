"""Per-channel quality metrics.

Each two-wavelength metric is reduced to the worse wavelength, since the
Beer-Lambert inversion needs both.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy import stats

from ..preproc import dark_level
from ..types import Hb, Raw
from ._signal import (Filter, cpsd, design_butter, filt_safe, filtfilt, nan_max, nan_min,
                      robust_sigma, round_half_away)

_EPS = np.finfo(float).eps

TASK_BAND = (0.01, 0.20)
NOISE_CUTOFF = 0.50
CARDIAC_BAND = (0.50, 1.50)
#: Motion amplitude floor in base-10 optical density; about 0.18 in the
#: natural-log units Homer uses.
MOTION_AMPLITUDE = 0.08
MOTION_WINDOW_S = 0.5
#: How far below the largest value the device wrote still counts as clipped.
CEILING_TOLERANCE = 0.02


@dataclass
class RecordingContext:
    """Everything the metrics need that is a property of the whole recording."""
    fs: float
    dark: float
    ceiling: float
    task: Optional[Filter]
    noise: Optional[Filter]
    cardiac: Optional[Filter]
    hb: Optional[Hb] = None

    @classmethod
    def from_raw(cls, raw: Raw, hb: Optional[Hb] = None) -> RecordingContext:
        fs = raw.fs
        return cls(fs=fs, dark=mean_dark_level(raw), ceiling=saturation_ceiling(raw),
                   task=design_butter(fs, TASK_BAND, "bandpass"),
                   noise=design_butter(fs, NOISE_CUTOFF, "highpass"),
                   cardiac=design_butter(fs, (CARDIAC_BAND[0], min(CARDIAC_BAND[1], 0.49 * fs)),
                                         "bandpass"),
                   hb=hb)


def channel_metric(name: str, a: np.ndarray, b: np.ndarray, ch: int,
                   ctx: RecordingContext) -> float:
    """One metric for one channel, from its two wavelength traces ``a`` and ``b``."""
    if name == "SCI":
        return sci_psp(a, b, ctx.fs, ctx.cardiac, "sci")
    if name == "PSP":
        return sci_psp(a, b, ctx.fs, ctx.cardiac, "psp")
    if name == "SNR_dB":
        return nan_min(band_snr(a, ctx), band_snr(b, ctx))
    if name == "SNR_simple":
        return nan_min(temporal_snr(a), temporal_snr(b))
    if name == "CV":
        return nan_max(coefficient_of_variation(a), coefficient_of_variation(b))
    if name == "SignalAboveDark":
        # CV and SNR on the raw trace are misleading here: an uncoupled
        # optode still looks stable because it is a constant offset.
        # Subtracting the dark level separates light from pedestal.
        return nan_min(np.nanmean(a), np.nanmean(b)) - ctx.dark
    if name == "LightOverNoise":
        return light_over_noise(a, b, ctx.dark)
    if name == "Saturation":
        return nan_max(np.mean(a >= ctx.ceiling), np.mean(b >= ctx.ceiling))
    if name == "MAD":
        return nan_max(_mad(a), _mad(b))
    if name == "Kurtosis":
        return nan_max(_moment(a, stats.kurtosis, fisher=False),
                       _moment(b, stats.kurtosis, fisher=False))
    if name == "Skewness":
        sa, sb = _moment(a, stats.skew), _moment(b, stats.skew)
        return sb if abs(sb) > abs(sa) or (np.isnan(sa) and not np.isnan(sb)) else sa
    if name == "MotionRate":
        # On optical density, so the threshold means the same whatever the LED
        # drive and detector gain were.
        return nan_max(motion_rate(to_od(a), ctx.fs), motion_rate(to_od(b), ctx.fs))
    if name == "HbCorr":
        return hb_correlation(ctx.hb, ch, ctx.task) if ctx.hb is not None else np.nan
    if name in ("CNR", "CBR", "BlockConsistency"):
        if ctx.hb is None or not len(ctx.hb.conditions):
            return np.nan
        return task_metric(name, ctx.hb, ch)
    raise ValueError(f"Unknown QC metric {name!r}.")


def sci_psp(a: np.ndarray, b: np.ndarray, fs: float, cardiac: Optional[Filter],
            which: Literal["sci", "psp"]) -> float:
    """Scalp coupling index, or peak cross-spectral power, in the cardiac band.

    Both wavelengths see the same heartbeat through a well coupled optode, so
    their cardiac-band traces correlate; through an uncoupled one they do not.
    """
    if cardiac is None or fs < 2:
        return np.nan
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < max(8, int(np.ceil(4 * fs))):
        return np.nan
    try:
        af, bf = filtfilt(cardiac, a[ok]), filtfilt(cardiac, b[ok])
    except ValueError:
        return np.nan
    sa, sb = np.std(af, ddof=1), np.std(bf, ddof=1)
    if sa == 0 or sb == 0:
        return np.nan
    an, bn = (af - af.mean()) / sa, (bf - bf.mean()) / sb
    if which == "sci":
        return float(np.corrcoef(an, bn)[0, 1])
    nfft = max(256, int(2 ** np.ceil(np.log2(min(an.size, 2048)))))
    pxy, freq = cpsd(an, bn, nfft, fs)
    band = (freq >= CARDIAC_BAND[0]) & (freq <= min(CARDIAC_BAND[1], 0.49 * fs))
    return float(np.max(np.abs(pxy[band]))) if band.any() else np.nan


def band_snr(x: np.ndarray, ctx: RecordingContext) -> float:
    """Task-band RMS over high-frequency RMS, in dB."""
    s, n = filt_safe(x, ctx.task), filt_safe(x, ctx.noise)
    if s is None or n is None:
        return np.nan
    return 20 * np.log10(nan_max(_rms(s), _EPS) / nan_max(_rms(n), _EPS))


def temporal_snr(x: np.ndarray) -> float:
    return 20 * np.log10(nan_max(abs(np.nanmean(x)), _EPS) / nan_max(np.nanstd(x, ddof=1), _EPS))


def coefficient_of_variation(x: np.ndarray) -> float:
    return np.nanstd(x, ddof=1) / nan_max(abs(np.nanmean(x)), _EPS)


def light_over_noise(a: np.ndarray, b: np.ndarray, dark: float) -> float:
    """Signal above the dark floor in units of the channel's own instrument noise.

    Noise is the robust SD of the first difference divided by sqrt(2), the
    white-noise level of the trace. The SD of the trace itself would be
    measuring the cardiac and haemodynamic signal on a well coupled channel,
    making the ratio circular.
    """
    vals = []
    for w in (a, b):
        w = w[np.isfinite(w)]
        if w.size < 20:
            continue
        noise = robust_sigma(np.diff(w)) / np.sqrt(2)
        if np.isfinite(noise) and noise > 0:
            vals.append((w.mean() - dark) / noise)
    return min(vals) if vals else np.nan


def motion_rate(od: np.ndarray, fs: float) -> float:
    """Motion artefacts per minute, in the spirit of Homer's hmrMotionArtifactByChannel.

    A sample is an artefact when the change over 0.5 s exceeds both 5 robust
    SD of those changes and an absolute 0.08 OD. The absolute floor is what
    makes this usable: with the relative criterion alone a quantised or very
    clean channel has a near-zero spread, so quantisation steps and ordinary
    cardiac pulsation clear it and the rate reports the heart rate. Runs are
    counted, not samples, so the rate does not scale with the sampling rate.
    """
    x = od[np.isfinite(od)]
    if x.size < 20:
        return np.nan
    win = max(1, round_half_away(MOTION_WINDOW_S * fs))
    d = x[win:] - x[:-win]
    if d.size < 10:
        return np.nan
    thresh = max(5 * robust_sigma(d), MOTION_AMPLITUDE)
    hit = np.abs(d) > thresh
    events = int(np.sum(np.diff(np.r_[0, hit.astype(int)]) == 1))
    return events / max(x.size / fs / 60, _EPS)


def to_od(x: np.ndarray) -> np.ndarray:
    """Optical density relative to the trace mean (no dark correction)."""
    m = np.nanmean(x)
    if not np.isfinite(m) or m <= 0:
        return np.full(x.shape, np.nan)
    return -np.log10(np.maximum(x, _EPS) / m)


def hb_correlation(hb: Hb, ch: int, task: Optional[Filter]) -> float:
    """Correlation of task-band HbO and HbR over the samples where both are finite."""
    x, y = filt_safe(hb.hbo[:, ch], task), filt_safe(hb.hbr[:, ch], task)
    if x is None or y is None:
        return np.nan
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 1 else np.nan


def task_metric(name: str, hb: Hb, ch: int) -> float:
    """CNR, CBR or BlockConsistency on HbO, pooled over every condition.

    The response window is 6-12 s after onset whatever the stimulus length:
    capping it at the stimulus duration, as block-design code often does,
    would leave event-related trials (2 s) with an empty window although
    their haemodynamic response still peaks around 6 s.
    """
    x = hb.hbo[:, ch]
    if not np.isfinite(x).any():
        return np.nan
    onsets = np.concatenate(hb.conditions.onsets) if hb.conditions.onsets else np.array([])
    durations = (np.concatenate(hb.conditions.durations) if hb.conditions.durations
                 else np.array([]))
    if not onsets.size:
        return np.nan

    fs, n, pre = hb.fs, x.size, 5.0
    vals = np.full(onsets.size, np.nan)
    segments = []
    for k, (onset, dur) in enumerate(zip(onsets, durations)):
        s = onset - hb.time[0]
        base = _window(s - pre, s, fs, n)
        resp = _window(s + 6, s + 12, fs, n)
        if base is None or resp is None:
            continue
        if name == "CNR":
            den = np.sqrt(np.nanvar(x[base], ddof=1) + np.nanvar(x[resp], ddof=1))
            vals[k] = (np.nanmean(x[resp]) - np.nanmean(x[base])) / nan_max(den, _EPS)
        elif name == "CBR":
            during = _window(s, s + dur, fs, n)
            if during is not None:
                vals[k] = np.nanmean(x[during]) - np.nanmean(x[base])
        else:
            full = _window(s - pre, s + max(dur, 12), fs, n)
            if full is not None and full.stop - full.start > 2:
                segments.append(x[full])

    if name != "BlockConsistency":
        return float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
    if len(segments) < 2:
        return np.nan
    length = min(s.size for s in segments)
    block = np.column_stack([s[:length] for s in segments])
    block = block[np.isfinite(block).all(axis=1)]
    r = np.corrcoef(block, rowvar=False)
    upper = r[np.triu_indices(len(segments), k=1)]
    return float(np.nanmean(upper)) if np.isfinite(upper).any() else np.nan


def mean_dark_level(raw: Raw) -> float:
    """The detector's zero-light level, averaged over the two wavelengths.

    Read off the source-detector pairs the montage does not use, which are
    never illuminated (:func:`wnirs.preproc.dark_level`, ``'unused'``, with
    its 1st-percentile fallback); 0 when neither gives a finite level.
    """
    unused = None
    if raw.all_pairs is not None and raw.all_pairs_index is not None:
        unused = raw.all_pairs[:, ~raw.all_pairs_index["in_layout"].to_numpy(dtype=bool)]
    dark, _ = dark_level(raw.intensity, "unused", unused)
    level = np.nanmean(dark) if np.isfinite(dark).any() else 0.0
    return float(level) if np.isfinite(level) else 0.0


def saturation_ceiling(raw: Raw) -> float:
    """The clipping level: the largest value the device wrote, less a small tolerance.

    Taken from the recording rather than assumed. On a recording where
    nothing clipped this is simply the brightest sample, so the brightest
    channel's peaks count as "at the ceiling"; the Saturation gate's 1 %
    tolerance is what absorbs that.
    """
    x = raw.intensity[np.isfinite(raw.intensity)]
    return float(x.max() - CEILING_TOLERANCE) if x.size else np.inf


def _window(t_start: float, t_end: float, fs: float, n: int) -> Optional[slice]:
    i1 = max(0, round_half_away(t_start * fs))
    i2 = min(n - 1, round_half_away(t_end * fs))
    return slice(i1, i2 + 1) if i2 > i1 else None


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(x ** 2)))


def _mad(x: np.ndarray) -> float:
    return float(np.nanmedian(np.abs(x - np.nanmedian(x))))


def _moment(x: np.ndarray, fn: Callable[..., float], **kw: bool) -> float:
    """A bias-corrected moment (``bias=False``), NaN below four samples."""
    x = x[np.isfinite(x)]
    return float(fn(x, bias=False, **kw)) if x.size >= 4 else np.nan

