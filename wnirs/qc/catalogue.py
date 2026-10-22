"""The quality metrics the toolbox can compute."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd


@dataclass(frozen=True)
class Metric:
    """One entry of the catalogue.

    ``needs`` says what it is computed from: ``'raw'`` intensity, ``'hb'``
    haemoglobin, or ``'task'`` (haemoglobin plus stimulus timing; NaN on a
    resting run). ``direction`` says which way is better: ``'high'``,
    ``'low'``, ``'neg'`` (more negative) or ``'band'`` (inside a range).
    """
    name: str
    group: str
    needs: Literal["raw", "hb", "task"]
    direction: Literal["high", "low", "neg", "band"]
    default: bool
    description: str


METRICS: tuple[Metric, ...] = (
    Metric("SCI", "coupling", "raw", "high", True,
           "Scalp coupling index: correlation between the two wavelengths in the cardiac band "
           "(0.5-1.5 Hz). The standard optode-contact check."),
    Metric("PSP", "coupling", "raw", "high", True,
           "Peak spectral power of the cross-spectrum between wavelengths in the cardiac band. "
           "Confirms SCI with an amplitude, not just a correlation."),
    Metric("SNR_dB", "amplitude", "raw", "high", True,
           "Task-band (0.01-0.2 Hz) RMS over high-frequency (>0.5 Hz) RMS in dB, worst "
           "wavelength."),
    Metric("SNR_simple", "amplitude", "raw", "high", False,
           "Temporal SNR, 20*log10(mean/SD) of raw intensity, worst wavelength. 20 dB is a "
           "10 percent coefficient of variation."),
    Metric("CV", "amplitude", "raw", "low", True,
           "Coefficient of variation of raw intensity, worst wavelength."),
    Metric("SignalAboveDark", "amplitude", "raw", "high", True,
           "Mean intensity above the detector dark level, worst wavelength: the light that "
           "actually reached the detector. Near zero means the optode was not coupled, however "
           "clean the raw trace looks."),
    Metric("LightOverNoise", "amplitude", "raw", "high", True,
           "SignalAboveDark divided by the channel's own sample-to-sample noise, worst "
           "wavelength. Self-scaling, so it means the same on a wearable as on a benchtop "
           "system. Noise comes from the first difference, not the SD, because a coupled "
           "trace is dominated by cardiac and haemodynamic signal."),
    Metric("Saturation", "amplitude", "raw", "low", True,
           "Fraction of samples pinned at the detector ceiling. Above zero means the channel "
           "is clipped and its response compressed."),
    Metric("MAD", "noise", "raw", "low", False,
           "Median absolute deviation of raw intensity; spread that ignores motion spikes."),
    Metric("Kurtosis", "noise", "raw", "low", False,
           "Kurtosis of raw intensity. Far above 3 indicates spiky, artefact-heavy data."),
    Metric("Skewness", "noise", "raw", "band", False,
           "Skewness of raw intensity. Strong asymmetry usually means step shifts from motion."),
    Metric("MotionRate", "noise", "raw", "low", True,
           "Motion artefacts per minute: runs of 0.5 s changes in optical density beyond both "
           "5 robust SD and 0.08 OD."),
    Metric("HbCorr", "coupling", "hb", "neg", True,
           "Correlation between HbO and HbR in the task band. Neurovascular coupling is "
           "anti-correlated, so about -0.5 or lower is healthy."),
    Metric("CNR", "task", "task", "high", True,
           "Contrast-to-noise ratio on HbO: mean 6-12 s after onset minus the 5 s baseline, "
           "over the pooled SD, median across trials."),
    Metric("CBR", "task", "task", "high", False,
           "Contrast-to-baseline ratio on HbO: mean during the stimulus minus the 5 s baseline "
           "in micromolar, median across trials."),
    Metric("BlockConsistency", "task", "task", "high", False,
           "Mean pairwise correlation between single-trial HbO responses."),
)


def metric_table() -> pd.DataFrame:
    """The catalogue as a table: Name, Group, Needs, Direction, Default, Description."""
    return pd.DataFrame([vars(m) for m in METRICS]).rename(columns=str.capitalize)


def resolve_metrics(metrics: Optional[str | Iterable[str]] = None) -> list[str]:
    """Metric names to compute: ``None`` for the defaults, ``'all'``, or explicit names."""
    if metrics is None:
        return [m.name for m in METRICS if m.default]
    if isinstance(metrics, str):
        if metrics.lower() == "all":
            return [m.name for m in METRICS]
        metrics = [metrics]
    names = list(metrics)
    known = {m.name for m in METRICS}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"Unknown QC metric(s): {', '.join(unknown)}. "
                         "See wnirs.qc.metric_table() for the catalogue.")
    return names
