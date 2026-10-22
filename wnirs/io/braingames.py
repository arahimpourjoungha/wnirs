"""Brain Games tablet exports: one SensorData CSV plus one BehavioralData workbook per task.

The tablet writes each task of each session as loose files::

    NBack_SensorData_Session1_90605_20260805_180557.csv
    NBack_BehavioralData_Session1_90605_20260805_181357.xlsx

The Stimulus column of the sensor file is zero throughout; all task timing is
in the behavioural workbook. A recording read without its companion therefore
has no conditions and cannot be block averaged or fitted, although it is still
fine for connectivity.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..types import Raw
from .layout import load_layout
from .names import BrainGamesName, parse_brain_games_name
from .raw import WAVELENGTHS, build_raw
from .sensor import read_sensor_table

#: Typical task length on this tablet, in seconds.
TASK_LENGTH = {"Flanker": 450.0, "GoNogo": 441.0, "NBack": 487.0, "Rest": 444.0}
#: A recording longer than this multiple of its task length is trimmed.
TRIM_FACTOR = 1.25
#: Behaviour is written as the task finishes, so a real companion is minutes
#: away. Six hours is far more slack than the protocol needs and still keeps a
#: session recorded either side of midnight together.
MAX_COMPANION_GAP_MIN = 6 * 60
_WARN_COMPANION_GAP_MIN = 120
#: The read cap assumes at most this sampling rate; the tablet runs at ~7 Hz.
_MAX_FS_FOR_READ_CAP = 20


def read_brain_games(sensor_file: str | Path, layout: str | Path | pd.DataFrame = "full", *,
                     behaviour_file: Optional[str | Path] = None,
                     wavelengths: tuple[float, float] = WAVELENGTHS,
                     subject: Optional[str] = None, session: Optional[str] = None,
                     task: Optional[str] = None, gonogo_split: bool = True,
                     nback_split: bool = True, correct_only: bool = False,
                     task_length: Optional[Mapping[str, float]] = None,
                     trim_factor: float = TRIM_FACTOR) -> Raw:
    """Read a Brain Games recording saved on the tablet.

    Point this at the SensorData CSV; the matching BehavioralData workbook is
    found by task, session and recording time (see :func:`find_companion`).

    Over-long recordings are trimmed, not dropped. The tablet sometimes keeps
    writing after the task ends, for hours (4.3 GB against a 42 MB norm). The
    task itself is at the start of the file, so the first task-length of
    samples is the session and the rest is not. Only the rows needed are read,
    so a runaway costs the same as a normal file. The cut is at the task's own
    length rather than a round 7 minutes: the tasks run 7.3-8.1 min, and a
    7:00 cut would clip the final trials of every n-back. Trials that start
    after the cut are dropped and ``Raw.trimmed_at`` records where it was cut.

    Parameters
    ----------
    sensor_file : str or Path
        The ``*_SensorData_*.csv`` file.
    layout : str, Path or DataFrame
        See :func:`load_layout`.
    behaviour_file : str or Path, optional
        Use this companion instead of searching for one.
    wavelengths : (float, float)
        True emitter wavelengths in nm (the headers say 740/850).
    subject, session, task : str, optional
        Override what the filename says.
    gonogo_split : bool
        Label Go/NoGo trials separately (otherwise one ``SkipTheX`` condition).
    nback_split : bool
        Keep the n-back load levels apart (otherwise one ``NBack`` condition).
    correct_only : bool
        Keep only trials answered correctly.
    task_length : mapping, optional
        Task length in seconds by paradigm, merged over :data:`TASK_LENGTH`.
    trim_factor : float
        Trim when the recording runs past this multiple of its task length.
        ``math.inf`` switches trimming off.

    Returns
    -------
    Raw
        With ``trials`` from the behavioural file and ``meta`` holding
        ``behaviour_file``, ``recorded_at`` and ``unfinished``.
    """
    sensor_file = Path(sensor_file)
    if not sensor_file.is_file():
        raise FileNotFoundError(f"File not found: {sensor_file}")
    stem = sensor_file.stem
    info = parse_brain_games_name(stem)

    beh = Path(behaviour_file) if behaviour_file else find_companion(sensor_file.parent, info)
    trials = pd.DataFrame()
    if beh is None:
        if info.task != "Rest":
            warnings.warn(
                f"No BehavioralData file found for {info.task} session {info.session} ({stem}). "
                "The sensor file's Stimulus column is zero throughout, so this recording will "
                "have no conditions: usable for connectivity, not for block averaging or a GLM.",
                stacklevel=2)
    else:
        trials = read_trials(beh, info.task, gonogo_split=gonogo_split,
                             nback_split=nback_split, correct_only=correct_only)

    length = {**TASK_LENGTH, **(task_length or {})}.get(info.task, math.nan)
    trimming = math.isfinite(length) and math.isfinite(trim_factor)
    max_rows = math.ceil(2 * length * _MAX_FS_FOR_READ_CAP) + 1 if trimming else None
    table = read_sensor_table(sensor_file, max_rows=max_rows)

    trimmed_at = None
    tt = table.column("Time")
    if trimming and tt is not None and np.isfinite(tt).any():
        t0 = tt[np.isfinite(tt)][0]
        # A read stopped by the row cap means the file is bigger than the task.
        capped = table.n_rows >= max_rows
        if np.nanmax(tt) - t0 > trim_factor * length or capped:
            table = table.take_rows((tt - t0 <= length) | ~np.isfinite(tt))
            trimmed_at = length
            n_lost = 0
            if "Onset" in trials:
                n_before = len(trials)
                trials = trials[trials["Onset"] <= t0 + length].reset_index(drop=True)
                n_lost = n_before - len(trials)
            warnings.warn(f"{stem} runs past its task length ({info.task} is {length:.0f} s); "
                          f"trimmed to the first {length:.0f} s. {n_lost} trial(s) after the "
                          "cut dropped.", stacklevel=2)

    raw = build_raw(table, load_layout(layout), file=sensor_file,
                    subject=subject if subject is not None else info.subject,
                    task=task if task is not None else info.task,
                    session=session if session is not None else info.session,
                    wavelengths=wavelengths, trials=trials, warn_no_timing=False)
    raw.trimmed_at = trimmed_at
    raw.meta.update(behaviour_file=beh, recorded_at=info.recorded_at,
                    unfinished=info.unfinished)
    if info.unfinished:
        warnings.warn(f'{stem} is marked "unfinished": the app stopped before closing the '
                      "file. It is read anyway, but expect fewer blocks than the design "
                      "specifies.", stacklevel=2)
    return raw


def find_companion(folder: Path, info: BrainGamesName) -> Optional[Path]:
    """The BehavioralData workbook belonging to a sensor recording, or None.

    Same session number first, then the closest recording time. The number
    alone is not enough: the app sometimes writes several files with the same
    number on different days, and pairing a sensor file with the wrong day's
    behaviour would silently shift every onset.

    There is deliberately no fallback to "any file for this task". When a
    session's behavioural file is genuinely missing, the nearest other one can
    be weeks away; its onsets have nothing to do with the recording, yet a GLM
    fitted to them still returns ordinary-looking betas. A missing companion
    has to mean no conditions, not someone else's conditions.
    """
    app = info.app_task or info.task
    pool = []
    for f in sorted(Path(folder).glob(f"{app}_BehavioralData_*")):
        if f.name.startswith((".", "~$")):
            continue
        cand = parse_brain_games_name(f.stem)
        if cand.session == info.session:
            pool.append((f, cand.recorded_at))
    if not pool:
        return None

    if info.recorded_at is None or all(dt is None for _, dt in pool):
        if len(pool) == 1:
            return pool[0][0]
        warnings.warn(f"{len(pool)} BehavioralData files share session {info.session} and none "
                      "carries a timestamp, so the right one cannot be identified. Reading this "
                      "recording without conditions.", stacklevel=3)
        return None

    gaps = [abs((dt - info.recorded_at).total_seconds()) / 60 if dt else math.inf
            for _, dt in pool]
    best = int(np.argmin(gaps))
    gap = gaps[best]
    if gap > MAX_COMPANION_GAP_MIN:
        warnings.warn(f"The nearest {info.task} BehavioralData file for session {info.session} "
                      f"is {gap / 1440:.1f} days from the recording, so it belongs to a different "
                      "session. Reading this recording without conditions: wrong timing would "
                      "have produced a contrast map that looked perfectly normal.", stacklevel=3)
        return None
    if gap > _WARN_COMPANION_GAP_MIN:
        warnings.warn(f"The {info.task} BehavioralData file for session {info.session} is "
                      f"{gap:.0f} minutes from the recording. Within tolerance, but check they "
                      "belong together.", stacklevel=3)
    return pool[best][0]


def read_trials(behaviour_file: str | Path, task: str, *, gonogo_split: bool = True,
                nback_split: bool = True, correct_only: bool = False) -> pd.DataFrame:
    """Onsets, durations and condition labels from a tablet BehavioralData workbook.

    The tablet names columns in camelCase where the portal export uses prose.
    ``stimOn_ms`` is in seconds despite its name (it matches the portal's
    "Stim on-time(s.ms)" for the same trial) and shares the sensor file's
    Time clock. Without an offset column every trial lasts 2 s; a missing or
    non-positive duration takes the median of the valid ones.

    Returns
    -------
    DataFrame
        ``Onset``, ``Duration``, ``Condition`` and, where the workbook has
        them, ``Accuracy``, ``ResponseTime``, ``CommissionError``,
        ``Omission``. Empty when the workbook cannot be used.
    """
    try:
        book = pd.read_excel(behaviour_file, sheet_name=0)
    except Exception as err:  # openpyxl raises a zoo of types on a damaged workbook
        warnings.warn(f"Could not read {behaviour_file} ({err}).", stacklevel=2)
        return pd.DataFrame()
    if book.empty:
        return pd.DataFrame()
    cols = [str(c) for c in book.columns]

    on_col = _pick(cols, ("stimOn_ms", "stimOnMs", "stimOn"), "stimon")
    off_col = _pick(cols, ("stimOffTime", "stimOff_ms", "stimOff"), "stimoff")
    if on_col is None:
        warnings.warn(f"{behaviour_file} has no stimulus onset column; no conditions built.",
                      stacklevel=2)
        return pd.DataFrame()

    onset = _numeric(book.iloc[:, on_col])
    duration = (np.full(onset.size, 2.0) if off_col is None
                else _numeric(book.iloc[:, off_col]) - onset)
    valid = np.isfinite(duration) & (duration > 0)
    duration[~valid] = np.median(duration[valid]) if valid.any() else 2.0

    label = _condition_labels(book, cols, task, onset.size, gonogo_split, nback_split)
    ok = np.isfinite(onset) & (np.char.str_len(label) > 0)
    trials = pd.DataFrame({"Onset": onset[ok], "Duration": duration[ok], "Condition": label[ok]})

    for names, out in ((("accuracy",), "Accuracy"), (("responseTime",), "ResponseTime"),
                       (("commissionError", "Commison"), "CommissionError"),
                       (("omission",), "Omission")):
        col = _pick(cols, names)
        if col is not None:
            trials[out] = _numeric(book.iloc[:, col])[ok]

    if correct_only and "Accuracy" in trials:
        n0 = len(trials)
        trials = trials[trials["Accuracy"] == 1].reset_index(drop=True)
        if len(trials) < n0:
            warnings.warn(f"{task}: keeping {len(trials)} of {n0} correct trials (correct_only).",
                          stacklevel=2)
    return trials


def _condition_labels(book: pd.DataFrame, cols: list[str], task: str, n: int,
                      gonogo_split: bool, nback_split: bool) -> np.ndarray:
    def labels(col: Optional[int], fallback: str) -> np.ndarray:
        if col is None:
            return np.full(n, fallback, dtype=object)
        return np.array([_cell_text(v) for v in book.iloc[:, col]], dtype=object)

    key = task.upper()
    if key in ("NBACK", "N-BACK"):
        out = labels(_pick(cols, ("block",), "block"), "NBack")
        if not nback_split:
            out = np.full(n, "NBack", dtype=object)
    elif key in ("FISH", "FLANKER"):
        col = _pick(cols, ("stimuliType", "stimulusType"), "stimulitype")
        out = np.array([s.strip().lower() for s in labels(col, "Flanker")], dtype=object)
    elif key in ("SKIPX", "GONOGO"):
        shown = labels(_pick(cols, ("stimuliPresented", "stimulus"), "stimulipresented"), "GoNogo")
        if gonogo_split:
            out = np.where([s.strip().upper() == "X" for s in shown], "NoGo", "Go").astype(object)
        else:
            out = np.full(n, "SkipTheX", dtype=object)
    else:
        out = np.full(n, task, dtype=object)
    return np.array([s.strip() for s in out], dtype=str)


def _pick(cols: Sequence[str], exact: Sequence[str], contains: str = "") -> Optional[int]:
    """Column index by case-insensitive exact name, then by substring."""
    lower = [c.lower() for c in cols]
    for name in exact:
        if name.lower() in lower:
            return lower.index(name.lower())
    if contains:
        return next((j for j, c in enumerate(lower) if contains.lower() in c), None)
    return None


def _numeric(col: pd.Series) -> np.ndarray:
    """Spreadsheet column as floats; a cell stored as text is parsed, junk becomes NaN."""
    return pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)


def _cell_text(v: object) -> str:
    if not isinstance(v, str) and pd.isna(v):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)
