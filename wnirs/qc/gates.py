"""Pass/fail gates on the QC metrics.

A metric decides pass/fail only when it has a finite threshold. The defaults
are the gate set the ADHD study ran on:

- ``LightOverNoise >= 20``. Light measured in units of each channel's own
  noise rather than against an absolute number. An absolute floor calibrated
  on a benchtop system is arbitrary on a wearable: on this hardware 0.02 is
  35.7x the median instrument noise. 20 was set by matching amplitude: at
  that level the beta distribution is indistinguishable from the old absolute
  gate's while keeping 14 % more channels.
- ``SignalAboveDark >= 0.005``. A ratio needs an absolute companion. LON is
  scale-invariant, so it cannot tell clean signal from clean nothing: a frozen
  trace with almost no light and almost no sample-to-sample variation scores
  arbitrarily high. The floor sits where it separates those two populations
  and no higher (it costs 1.5 % of channels; the old 0.02 cost 26 %).
- ``Saturation <= 0.01`` and ``MotionRate <= 5`` per minute.
- SCI, SNR_dB, CV and HbCorr are computed and reported but do not gate.

Overrides are merged into the defaults, so a gate left out of the override
keeps its default; switching a gate off takes an explicit ``None`` or NaN.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS: dict[str, Optional[float]] = {
    "LightOverNoise": 20.0,
    "SignalAboveDark": 0.005,
    "Saturation": 0.01,
    "MotionRate": 5.0,
    "SCI": None,
    "SNR_dB": None,
    "CV": None,
    "HbCorr": None,
}

#: Direction of every gate, as text, for reports. An unknown metric is an
#: error rather than a default: a gate documented in the wrong direction is
#: undetectable downstream.
GATE_OPS: dict[str, str] = {
    "SCI": ">=", "PSP": ">=", "SNR_dB": ">=", "SignalAboveDark": ">=", "LightOverNoise": ">=",
    "CNR": ">=", "CV": "<=", "MotionRate": "<=", "HbCorr": "<=", "Saturation": "<=",
}

# metric -> (fails(value, threshold), reason). Keep in step with GATE_OPS.
_RULES: dict[str, tuple[Callable[[np.ndarray, float], np.ndarray], str]] = {
    "SCI": (lambda v, t: v < t, "low SCI"),
    "SNR_dB": (lambda v, t: v < t, "low SNR"),
    "CV": (lambda v, t: v > t, "high CV"),
    "MotionRate": (lambda v, t: v > t, "motion"),
    "HbCorr": (lambda v, t: v > t, "HbO/HbR not anti-correlated"),
    "SignalAboveDark": (lambda v, t: v < t, "no light (optode not coupled)"),
    "LightOverNoise": (lambda v, t: v < t, "light not resolvable above this channel's noise"),
    "Saturation": (lambda v, t: v > t, "clipped at the detector ceiling"),
}


def gate_op(metric: str) -> str:
    """``'>='`` or ``'<='`` for a gated metric; KeyError for one with no recorded direction."""
    try:
        return GATE_OPS[metric]
    except KeyError:
        raise KeyError(f'No gate direction is recorded for "{metric}". Add it to GATE_OPS '
                       "alongside its rule; it is deliberately not defaulted.") from None


def merge_thresholds(overrides: Optional[Mapping[str, Optional[float]]] = None
                     ) -> dict[str, Optional[float]]:
    """The default gates with ``overrides`` applied on top.

    ``None`` and NaN both mean "report this metric, do not gate on it".
    A threshold for a metric that has no gate rule (PSP, CNR, ...) raises
    rather than being silently ignored.
    """
    unknown = set(overrides or {}) - set(_RULES)
    if unknown:
        raise KeyError(f"No gate rule exists for {', '.join(sorted(unknown))}; gated metrics "
                       f"are {', '.join(_RULES)}.")
    merged = {**DEFAULT_THRESHOLDS, **(overrides or {})}
    return {k: (None if v is None or not math.isfinite(v) else float(v))
            for k, v in merged.items()}


def active_gates(thresholds: Mapping[str, Optional[float]], metrics: list[str]
                 ) -> dict[str, float]:
    """The gates that actually decide: a finite threshold on a computed, judgeable metric."""
    return {k: v for k, v in thresholds.items() if v is not None and k in metrics}


def judge(table: pd.DataFrame, gates: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Pass/fail and a ``'; '``-joined reason per channel.

    A NaN metric does not fail its gate: a metric that could not be computed
    is not evidence against the channel. Note that this includes
    LightOverNoise on a perfectly frozen trace (zero noise); the
    SignalAboveDark floor is what catches that case.
    """
    n = len(table)
    passed = np.ones(n, dtype=bool)
    reasons = [[] for _ in range(n)]
    for name, (fails, reason) in _RULES.items():
        if name not in gates:
            continue
        v = table[name].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            bad = fails(v, gates[name]) & np.isfinite(v)
        passed &= ~bad
        for i in np.flatnonzero(bad):
            reasons[i].append(reason)
    return passed, np.array(["; ".join(r) for r in reasons], dtype=object)


def warn_if_miscalibrated(table: pd.DataFrame, gates: Mapping[str, float]) -> None:
    """Warn when a gate rejects 90 % or more of the channels it can judge.

    Default thresholds come from a literature dominated by benchtop systems.
    A wearable can sit well below them and still carry a usable response, so
    a gate that rejects nearly everything is far more likely miscalibrated for
    the hardware than a sign the recording is worthless.
    """
    for name, threshold in gates.items():
        v = table[name].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        frac = np.mean(v < threshold) if gate_op(name) == ">=" else np.mean(v > threshold)
        if frac >= 0.9:
            warnings.warn(
                f"{name} rejects {100 * frac:.0f}% of channels (threshold {threshold:g}, observed "
                f"median {np.median(v):g}, range {v.min():g} to {v.max():g}). If that is every "
                "recording rather than one bad cap fit, the threshold is calibrated for "
                f"different hardware: set {name} from your own distribution, or pass "
                f"{{'{name}': None}} and judge on the others.", stacklevel=3)
