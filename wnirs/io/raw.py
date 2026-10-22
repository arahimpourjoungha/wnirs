"""Turning a sensor table into a :class:`~wnirs.types.Raw` recording."""
from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..types import Conditions, Raw
from .layout import load_layout
from .names import parse_name
from .sensor import SensorTable, all_pairs, parse_sensor, read_sensor_table

WAVELENGTHS = (750.0, 860.0)


def read_csv(path: str | Path, layout: str | Path | pd.DataFrame = "full", *,
             wavelengths: tuple[float, float] = WAVELENGTHS,
             stim_map: Optional[Mapping[float, str]] = None,
             subject: Optional[str] = None, task: Optional[str] = None) -> Raw:
    """Read one headband dataLog CSV, whose task timing is its own Stimulus column.

    For a Brain Games tablet export use :func:`read_brain_games` instead: its
    Stimulus column is zero throughout and the timing lives in a separate
    behavioural workbook.

    Parameters
    ----------
    path : str or Path
        The export (``dataLog_*.csv`` or a renamed copy).
    layout : str, Path or DataFrame
        See :func:`load_layout`.
    wavelengths : (float, float)
        True emitter wavelengths in nm. The headers say 740/850, a known
        firmware labelling error; these are the values for the extinction
        coefficients.
    stim_map : mapping, optional
        Stimulus code -> condition name, e.g. ``{10: '0-Back', 20: '2-Back'}``.
    subject, task : str, optional
        Override what the filename says.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    stem = path.stem
    return build_raw(read_sensor_table(path), load_layout(layout), file=path,
                     subject=subject or parse_name(stem, "subject"),
                     task=task or parse_name(stem, "task"),
                     session=parse_name(stem, "session"),
                     wavelengths=wavelengths, stim_map=stim_map)


def build_raw(table: SensorTable, layout: pd.DataFrame, *, file: Path, subject: str, task: str,
              session: str, wavelengths: tuple[float, float] = WAVELENGTHS,
              trials: Optional[pd.DataFrame] = None,
              stim_map: Optional[Mapping[float, str]] = None,
              warn_no_timing: bool = True) -> Raw:
    """Assemble a :class:`Raw` from a sensor table, shared by every headband reader.

    When ``trials`` is given the conditions are built from it rather than from
    the Stimulus column, because the tablet and portal exports leave Stimulus
    at zero and keep the real timing in the behavioural file. A task recording
    left with no conditions is warned about unless ``warn_no_timing`` is off
    (for a caller that has already said why).
    """
    time = table.column("Time")
    if time is None:
        raise ValueError(f'No "Time" column in {file}.')
    keep = np.isfinite(time)
    if not keep.all():
        table, time = table.take_rows(keep), time[keep]
    dt = np.median(np.diff(time)) if time.size > 1 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"The Time column in {file} is not monotonic.")

    stim_col = table.column("Stimulus")
    stim = np.zeros_like(time) if stim_col is None else np.nan_to_num(stim_col, nan=0.0)

    ch = parse_sensor(table, layout)
    if ch.missing:
        warnings.warn(f"{len(ch.missing)} layout channel(s) have no column in {file}: "
                      f"{', '.join(ch.missing)}", stacklevel=2)
    if np.isnan(ch.intensity).all():
        raise ValueError(f"No layout channel matched a column in {file}. The layout and the "
                         "recording montage do not agree.")
    pairs, pair_index = all_pairs(table, layout)

    if trials is not None and len(trials):
        conditions = conditions_from_trials(trials, time, subject, task)
    else:
        conditions = conditions_from_stim(time, stim, stim_map)
        if (warn_no_timing and not len(conditions) and stim_col is not None
                and not _is_rest_like(task)):
            warnings.warn(
                f"{file} has no task timing: its Stimulus column is zero throughout. A Brain "
                "Games tablet export needs read_brain_games, which takes the timing from the "
                "behavioural file; read as a plain CSV it has no conditions, as here.",
                stacklevel=2)

    return Raw(subject=str(subject), task=str(task), session=str(session), file=Path(file),
               time=time, fs=1.0 / dt, intensity=ch.intensity, layout=layout,
               wavelengths=tuple(wavelengths), stim=stim, reference=ch.reference,
               led_power=ch.led_power, reference_led_power=ch.reference_led_power,
               all_pairs=pairs, all_pairs_index=pair_index, conditions=conditions,
               trials=trials if trials is not None else pd.DataFrame(),
               meta={"missing_channels": ch.missing})


def conditions_from_trials(trials: pd.DataFrame, time: np.ndarray, subject: str = "",
                           task: str = "") -> Conditions:
    """One condition per distinct trial label, in order of first appearance.

    Trials the sensor recording does not cover cannot be epoched. The two
    clocks can disagree at the end of a session, so dropping them is reported
    rather than assumed away. Conditions left with no trial are omitted.
    """
    t0, t_end = time[0], time[-1]
    out = Conditions()
    n_dropped = 0
    for name in pd.unique(trials["Condition"]):
        rows = trials[trials["Condition"] == name]
        onset = rows["Onset"].to_numpy(dtype=float)
        covered = (onset >= t0) & (onset <= t_end)
        n_dropped += int((~covered).sum())
        if covered.any():
            out.name.append(str(name))
            out.onsets.append(onset[covered])
            out.durations.append(rows["Duration"].to_numpy(dtype=float)[covered])
    if n_dropped:
        warnings.warn(f"{subject} / {task}: {n_dropped} trial(s) fall outside the "
                      f"{t_end - t0:.0f} s sensor recording and were dropped. The behavioural "
                      "and sensor clocks do not cover the same window.", stacklevel=3)
    return out


def conditions_from_stim(time: np.ndarray, stim: np.ndarray,
                         stim_map: Optional[Mapping[float, str]] = None) -> Conditions:
    """Blocks from a piecewise-constant Stimulus column (0 between blocks, a code during one).

    Without a ``stim_map``, a single code is named ``BreathHold``, the n-back
    codes 10/20/30 become ``0-Back``/``2-Back``/``3-Back``, and anything else
    ``Stim<code>``. A name that would repeat gets ``_<code>`` appended.
    """
    codes = np.unique(stim[stim > 0])
    names = _stim_names(codes, stim_map or {})
    out = Conditions()
    for code, name in zip(codes, names):
        m = (stim == code).astype(int)
        rising = np.flatnonzero(np.diff(np.r_[0, m]) == 1)
        falling = np.flatnonzero(np.diff(np.r_[m, 0]) == -1)
        n = min(rising.size, falling.size)
        out.name.append(name)
        out.onsets.append(time[rising[:n]])
        out.durations.append(time[falling[:n]] - time[rising[:n]])
    return out


def _stim_names(codes: np.ndarray, stim_map: Mapping[float, str]) -> list[str]:
    builtin = {10: "0-Back", 20: "2-Back", 30: "3-Back"}
    names = []
    for c in codes:
        if c in stim_map:
            names.append(str(stim_map[c]))
        elif codes.size == 1:
            names.append("BreathHold")
        elif c in builtin:
            names.append(builtin[c])
        else:
            names.append(f"Stim{c:g}")
    return [f"{n}_{c:g}" if names.count(n) > 1 else n for n, c in zip(names, codes)]


def _is_rest_like(task: str) -> bool:
    """Tasks that legitimately carry no events, so their silence is not worth a warning."""
    return re.search(r"rest|natural|baseline|spont", task, re.IGNORECASE) is not None
