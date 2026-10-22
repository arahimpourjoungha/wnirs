"""Discovering the recordings under a study folder."""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .braingames import MAX_COMPANION_GAP_MIN
from .names import BrainGamesName, parse_brain_games_name, parse_name

#: A recording below this fraction of its task's median size stopped within
#: seconds and cannot carry a session.
STUB_FRACTION = 1 / 20

SESSION_COLUMNS = ["Subject", "Session", "Task", "Format", "File", "Bytes",
                   "Unfinished", "RecordedAt", "HasBehaviour"]

# Folder names that say what a level is rather than whose it is.
_GENERIC_FOLDERS = {"wnirs", "raw", "data", "csv", "nirs", "pilot", "report", "reports",
                    "brain game", "braingame", "brain games", "nirduino", "nirduino app",
                    "sensor", "sensordata"}
# Folders that hold another modality's exports.
_OTHER_MODALITY = {"ema", "wearable", "wearables", "biosensor", "biosensors", "sensors",
                   "actigraphy", "questionnaires", "derivatives", "output", "results"}
_TEST_WORDS = ("test", "demo", "dummy", "sample", "trial", "practice", "xxx")
_NIRDUINO = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_dataLog$", re.IGNORECASE)


def find_sessions(data_root: str | Path) -> pd.DataFrame:
    """Every headband recording under ``data_root``, one row per subject, session and task.

    Searches recursively for sensor CSVs (Brain Games tablet exports, NIRduino
    logs and plain dataLog exports) and works out who and what each one is.

    The subject is the deepest folder under the root that looks like an id (a
    run of 3+ digits), so a batch folder above the participant and a modality
    folder below it are both skipped. For a tablet export the folder overrides
    the id written into the filename, which is whatever was typed at setup;
    a test login (``testuser_0327``) is a rehearsal, not a mistyped id, and is
    dropped rather than reassigned.

    Several recordings for one subject, session and task are reduced to one:

    1. candidates below 1/20 of the task's median size are set aside, unless
       that would leave none (there is no upper bound: an over-long file is a
       complete session plus an idle tail, which the reader trims);
    2. a recording whose behavioural file is on disk and within 6 h is
       preferred, because it is the attempt that ran to completion;
    3. the bigger file wins among what is left.

    A recording below 1/20 of its task's median is then dropped even when it
    is the only one for its session.

    Not recordings, and skipped: behavioural workbooks, header-only files
    (a task abandoned at the first screen), headerless fragments of an
    interrupted stream, study metadata and questionnaire exports
    (``participants.csv``, ``90589_EMA_raw.csv``, ``*_CAARS_*``), and anything
    in a folder named after another modality (``ema``, ``wearable``, ...).

    Returns
    -------
    DataFrame
        Columns ``Subject, Session, Task, Format, File, Bytes, Unfinished,
        RecordedAt, HasBehaviour``, sorted by subject, session and task (as
        text). Format is ``'braingames'``, ``'nirduino'`` or ``'wnirscsv'``.
    """
    # Canonical form first: on a symlinked root (/tmp on macOS) the relative
    # paths would otherwise keep their prefix and a mount point would be read
    # as the subject.
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {data_root}")

    rows = []
    for f in sorted(root.rglob("*.csv")):
        if not f.is_file() or not _is_candidate(f, root) or not _has_samples(f):
            continue
        row = _describe(f, root)
        if row is not None:
            rows.append(row)

    sessions = pd.DataFrame(rows, columns=SESSION_COLUMNS)
    if sessions.empty:
        return sessions
    sessions = _drop_stubs(_dedupe(sessions))
    return sessions.sort_values(["Subject", "Session", "Task"], kind="stable",
                                ignore_index=True)


def _is_candidate(f: Path, root: Path) -> bool:
    name = f.name
    if name.startswith((".", "~$")) or _is_metadata(name):
        return False
    if "_behavioraldata_" in name.lower():
        return False
    folders = f.parent.relative_to(root).parts
    return not any(p.lower() in _OTHER_MODALITY for p in folders)


def _describe(f: Path, root: Path) -> dict | None:
    stem = f.stem
    folders = [p for p in f.parent.relative_to(root).parts if p.lower() not in _GENERIC_FOLDERS]
    subject = parse_name(stem, "subject")
    if subject == stem and folders:
        subject = _subject_from_path(folders)
    session = parse_name(stem, "session")
    task = parse_name(stem, "task")
    fmt, recorded_at, has_behaviour = "wnirscsv", None, True

    if re.search(r"_SensorData_Session", stem, re.IGNORECASE):
        info = parse_brain_games_name(stem)
        if _is_test_id(info.subject):
            return None
        fmt, task, recorded_at = "braingames", info.task, info.recorded_at
        if folders:
            subject = _subject_from_path(folders)
        if info.session:
            session = info.session
        # Rest has no behavioural file by design; flagging every resting run
        # would bury the task recordings that are genuinely missing theirs.
        has_behaviour = info.task.lower() == "rest" or has_companion(f.parent, info)
    elif _NIRDUINO.match(stem):
        # NIRduino names a file by when it started and nothing else, so the
        # start time is the session identity.
        fmt, task = "nirduino", "Naturalistic"
        if folders:
            subject = _subject_from_path(folders)
        session = re.sub(r"_dataLog$", "", stem)

    return {"Subject": subject, "Session": session, "Task": task, "Format": fmt,
            "File": str(f), "Bytes": f.stat().st_size,
            "Unfinished": re.search(r"_unfinished$", stem, re.IGNORECASE) is not None,
            "RecordedAt": recorded_at, "HasBehaviour": has_behaviour}


def has_companion(folder: Path, info: BrainGamesName) -> bool:
    """Is this recording's BehavioralData workbook on disk, within 6 h of it?

    The pattern is anchored on both sides of the session number, so session 1
    does not borrow session 15's file. The time match is the same rule
    :func:`find_companion` applies when it pairs them: the app reuses session
    numbers across days, so a file can share the number and still belong to
    another attempt.
    """
    app = info.app_task or info.task
    found = [f for f in Path(folder).glob(f"{app}_BehavioralData_Session{info.session}_*")
             if not f.name.startswith((".", "~$"))]
    if not found or info.recorded_at is None:
        return bool(found)
    for f in found:
        dt = parse_brain_games_name(f.stem).recorded_at
        if dt is None or abs((dt - info.recorded_at).total_seconds()) / 60 <= MAX_COMPANION_GAP_MIN:
            return True
    return False


def _subject_from_path(folders: list[str]) -> str:
    ids = [p for p in folders if re.fullmatch(r"\d{3,}", p)]
    return ids[-1] if ids else folders[0]


def _is_test_id(subject: str) -> bool:
    """A genuine id here is all digits; a word is a test login or a typo worth a look by hand."""
    s = subject.strip().lower()
    return bool(s) and (any(w in s for w in _TEST_WORDS) or re.match(r"^[a-z]+", s) is not None)


def _is_metadata(name: str) -> bool:
    """Files that describe the study, or another modality, rather than being a recording."""
    n = name.lower()
    return (n in {"participants.csv", "participants.tsv", "sessions.csv", "sessions.tsv",
                  "qc.csv", "qc_all_recordings.csv", "behaviour.csv", "behavior.csv"}
            or n.startswith(("wnirs_", "qc_", "contrast_", "group_", "corr_",
                             "ema_", "ema-", "survey_", "redcap", "questionnaire"))
            or "tracksheet" in n or n.endswith("_ema.csv")
            or any(s in n for s in ("_ema_", "_caars", "caars_", "_survey_", "_redcap")))


def _has_samples(f: Path) -> bool:
    """False for a header with no data, or data with no header.

    Only the first four lines are read, so the check costs the same on a
    70 MB recording as on an empty one.
    """
    with open(f, encoding="utf-8-sig", errors="replace") as fh:
        first = fh.readline()
        n_data = sum(1 for _ in range(3) if (line := fh.readline()) and line.strip())
    if n_data == 0:
        warnings.warn(f"{f.name} has a header but no samples: the recording never started. "
                      "Skipped.", stacklevel=3)
        return False
    if _is_headerless(first):
        warnings.warn(f"{f.name} begins with data rather than a column header, so it is a "
                      "fragment of an interrupted recording and its channels cannot be "
                      "identified. Skipped.", stacklevel=3)
        return False
    return True


def _is_headerless(line: str) -> bool:
    fields = line.strip().split(",")[:6]
    return len(fields) >= 2 and all(_is_finite_number(x) for x in fields)


def _is_finite_number(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def _median_bytes_by_task(sessions: pd.DataFrame) -> dict[str, float]:
    positive = sessions[sessions.Bytes > 0]
    return positive.groupby("Task").Bytes.median().to_dict()


def _dedupe(sessions: pd.DataFrame) -> pd.DataFrame:
    key = sessions[["Subject", "Session", "Task", "Format"]].astype(str).agg("|".join, axis=1)
    if key.is_unique:
        return sessions
    median = _median_bytes_by_task(sessions)
    keep = []
    for _, group in sessions.groupby(key, sort=False):
        idx = group.index.to_numpy()
        if idx.size > 1:
            idx = _prefer_plausible(sessions, idx, median)
        if idx.size > 1:
            with_behaviour = idx[sessions.HasBehaviour[idx].to_numpy(dtype=bool)]
            if with_behaviour.size:
                idx = with_behaviour
        if idx.size > 1:
            best = idx[np.argmax(sessions.Bytes[idx].to_numpy())]
            r = sessions.loc[best]
            warnings.warn(f"{r.Subject} session {r.Session} {r.Task} has {idx.size} recordings; "
                          f"keeping the longest ({r.Bytes / 1e6:.0f} MB) and ignoring "
                          f"{idx.size - 1} other(s). The app reuses a session number when a "
                          "task is restarted.", stacklevel=3)
            idx = np.array([best])
        keep.extend(idx)
    return sessions.loc[sorted(keep)].reset_index(drop=True)


def _prefer_plausible(sessions: pd.DataFrame, idx: np.ndarray,
                      median: dict[str, float]) -> np.ndarray:
    ref = median.get(sessions.Task[idx[0]], math.nan)
    if not math.isfinite(ref) or ref <= 0:
        return idx
    ok = sessions.Bytes[idx].to_numpy() >= ref * STUB_FRACTION
    if not ok.any():
        return idx
    for i in idx[~ok]:
        r = sessions.loc[i]
        warnings.warn(f"{r.Subject} session {r.Session} {r.Task}: ignoring a "
                      f"{r.Bytes / 1e6:.0f} MB recording against a {ref / 1e6:.0f} MB median "
                      "for this task; a file that far out is a stub, not a longer session.",
                      stacklevel=4)
    return idx[ok]


def _drop_stubs(sessions: pd.DataFrame) -> pd.DataFrame:
    if len(sessions) < 2:
        return sessions
    median = sessions.Task.map(_median_bytes_by_task(sessions))
    drop = (sessions.Bytes > 0) & (sessions.Bytes < median * STUB_FRACTION)
    for i, r in sessions[drop].iterrows():
        warnings.warn(f"{r.Subject} session {r.Session} {r.Task}: dropping a "
                      f"{r.Bytes / 1e6:.2f} MB recording against a "
                      f"{median[i] / 1e6:.0f} MB median for this task; it stopped within "
                      "seconds and cannot carry a session.", stacklevel=3)
    return sessions[~drop].reset_index(drop=True)
