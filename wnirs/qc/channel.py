"""Per-channel data quality for one recording."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional

import numpy as np
import pandas as pd

from ..types import Hb, Raw
from .catalogue import resolve_metrics
from .gates import active_gates, judge, merge_thresholds, warn_if_miscalibrated
from .metrics import RecordingContext, channel_metric


def channel_qc(raw: Raw, hb: Optional[Hb] = None, *,
               metrics: Optional[str | Iterable[str]] = None,
               thresholds: Optional[Mapping[str, Optional[float]]] = None) -> pd.DataFrame:
    """Quality metrics and a pass/fail verdict for every channel of a recording.

    Parameters
    ----------
    raw : Raw
        The recording, as read. Metrics are computed on raw intensity.
    hb : Hb, optional
        Its haemoglobin, for HbCorr and the task metrics (CNR, CBR,
        BlockConsistency); those are NaN without it.
    metrics : str or iterable of str, optional
        Metric names, ``'all'``, or None for the catalogue defaults (SCI, PSP,
        SNR_dB, CV, SignalAboveDark, LightOverNoise, Saturation, MotionRate,
        HbCorr, CNR). See :func:`metric_table`.
    thresholds : mapping, optional
        Gate overrides, merged into :data:`DEFAULT_THRESHOLDS`; ``None`` or
        NaN switches a gate off. A gate only decides when its metric is also
        computed.

    Returns
    -------
    DataFrame
        One row per channel: ``Subject, Task, Session, Channel, Source,
        Detector, Label, IsShort``, one column per metric, ``Pass`` and
        ``Reason`` (the failed gates, ``''`` when it passed). ``attrs``
        holds the gates applied, the metrics and ``fs``.

    Warns
    -----
    UserWarning
        When a gate rejects 90 % or more of the channels it can judge, which
        usually means the threshold is calibrated for different hardware.
    """
    names = resolve_metrics(metrics)
    gates = active_gates(merge_thresholds(thresholds), names)
    ctx = RecordingContext.from_raw(raw, hb)
    layout = raw.layout

    values = np.full((raw.n_channels, len(names)), np.nan)
    for ch in range(raw.n_channels):
        a, b = raw.intensity[:, ch, 0], raw.intensity[:, ch, 1]
        for j, name in enumerate(names):
            values[ch, j] = channel_metric(name, a, b, ch, ctx)

    table = pd.DataFrame({"Subject": raw.subject, "Task": raw.task, "Session": raw.session,
                          "Channel": np.arange(1, raw.n_channels + 1),
                          "Source": layout.S.to_numpy(), "Detector": layout.D.to_numpy(),
                          "Label": layout.Label.to_numpy(), "IsShort": layout.Type.to_numpy() == 2})
    table[names] = values
    table["Pass"], table["Reason"] = judge(table, gates)
    warn_if_miscalibrated(table, gates)
    table.attrs.update(thresholds=gates, metrics=names, fs=raw.fs)
    return table
