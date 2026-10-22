"""The manifest: every recording a run will touch, with what the study knows about it."""
from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .sessions import find_sessions
from .tracksheet import Tracksheet, read_tracksheet

_SUBJECT_ALIASES = ("subject", "subject_id", "subjectid", "participant", "participant_id", "id")
_GROUP_ALIASES = ("group", "cohort", "arm", "condition", "diagnosis")
#: Assumed session length when the recording does not say: the platform's protocol.
SESSION_MINUTES = 45


def read_participants(path_or_root: str | Path) -> pd.DataFrame:
    """The participants table: one row per subject, at the top of the study folder.

    ::

        Subject,Group,Age,Sex,Medication
        90606,Patient,31,F,Stimulant

    ``Subject`` (or ``participant_id``, ``id``, ...) is the only required
    column. A missing Group becomes ``'unassigned'``, so a half-filled file
    still runs everything that does not need the arms. Every other column is
    carried through as a subject-level covariate.

    Group membership is a fact about the study, not a setting for one run, so
    it lives in a file beside the data where it survives a re-scan and can be
    version-controlled.

    Parameters
    ----------
    path_or_root : str or Path
        The CSV/TSV itself, or the study folder holding ``participants.csv``
        (or ``participants.tsv``, as BIDS has it).

    Returns
    -------
    DataFrame
        Empty (with Subject and Group columns) when there is no file.

    Raises
    ------
    ValueError
        No subject column, or a subject listed twice.
    """
    path = Path(path_or_root)
    if path.is_dir():
        path = next((p for p in (path / "participants.csv", path / "participants.tsv")
                     if p.is_file()), path / "participants.csv")
    if not path.is_file():
        return pd.DataFrame(columns=["Subject", "Group"])

    table = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    table = _rename_alias(table, _SUBJECT_ALIASES, "Subject")
    table = _rename_alias(table, _GROUP_ALIASES, "Group")
    if "Subject" not in table:
        raise ValueError(f"{path} has no Subject column (nor participant_id, or id). That column "
                         "is what joins this file to the recordings.")
    table["Subject"] = _as_text(table.Subject)
    table = table[table.Subject != ""].reset_index(drop=True)
    table["Group"] = (_as_text(table.Group).replace("", "unassigned") if "Group" in table
                      else "unassigned")
    dup = table.Subject[table.Subject.duplicated()].unique()
    if dup.size:
        raise ValueError(f"{path} lists {', '.join(dup)} more than once. One row per subject; "
                         "per-session measures belong in the EMA or biosensor files.")
    return table


def build_manifest(data_root: str | Path, *, subjects: Optional[Iterable[str]] = None,
                   sessions: Optional[Iterable[str]] = None,
                   tasks: Optional[Iterable[str]] = None,
                   groups: Optional[Iterable[str]] = None,
                   tracksheet: Optional[str | Path] = None,
                   exclude: Iterable[str | Path] = (),
                   tracksheet_decides: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Everything a run will touch, as one table keyed on subject, session and task.

    Scans ``data_root`` (:func:`find_sessions`), joins the participants table
    and the tracksheet onto it, and applies the filters. The session is a
    first-class part of the key: ten sessions of one person's N-Back are ten
    rows, never one.

    The tracksheet annotates; it does not decide. It is kept by hand, so its
    session numbers and dates can be wrong, and a recording is not less real
    because a row about it was never filled in. Each recording is matched to
    the tracksheet row of the same day (a day either side allowed), not by
    session number, because the filename's number comes from a counter that
    repeats. When no row is within a day, the row with the same session
    number is used.

    Parameters
    ----------
    data_root : str or Path
        The study folder.
    subjects, sessions, tasks, groups : iterable of str, optional
        Keep only these. ``None`` keeps everything.
    tracksheet : str or Path, optional
        The tracking workbook. Default: the first ``*racksheet*.xlsx`` in
        ``data_root``, if any.
    exclude : iterable of paths
        Folders whose files are not recordings however they are named (the
        output folder when it sits inside the data folder, EMA exports, ...).
    tracksheet_decides : bool
        Keep only recordings on a numbered tracksheet row not marked
        incomplete.

    Returns
    -------
    manifest : DataFrame
        ``Subject, Session, Task, Group, Format, File, Bytes, Unfinished,
        RecordedAt, HasBehaviour``, every participants covariate, and the
        tracksheet columns ``SessionDate, OnMed, MedTiming, TaskOrder,
        TracksheetSession, Note, InTracksheet, Incomplete, Unnumbered,
        OnMedNum, Condition, ConditionDetail``. Sorted by subject, then
        session numerically when every session id is a number.
    participants : DataFrame
        The participants table that was joined.
    """
    root = Path(data_root)
    found = find_sessions(root)
    for folder in exclude:
        prefix = str(Path(folder).resolve())
        found = found[~found.File.str.startswith(prefix)]
    if found.empty:
        return (pd.DataFrame(columns=["Subject", "Session", "Task", "Group", "Format", "File",
                                      "Bytes"]), pd.DataFrame())
    found = found.reset_index(drop=True)
    found["Session"] = found.Session.replace("", "1")

    sheet = _load_tracksheet(root, tracksheet)
    participants = read_participants(root)
    if participants.empty and sheet is not None and not sheet.participants.empty:
        participants = sheet.participants.copy()
    unknown = sorted(set(found.Subject) - set(participants.Subject))
    if unknown:
        participants = pd.concat([participants, pd.DataFrame({"Subject": unknown,
                                                              "Group": "unassigned"})],
                                 ignore_index=True)

    by_subject = participants.set_index("Subject")
    manifest = found.copy()
    manifest.insert(3, "Group", manifest.Subject.map(by_subject.Group).fillna("unassigned"))
    for col in by_subject.columns.drop("Group"):
        manifest[col] = manifest.Subject.map(by_subject[col])

    manifest = join_tracksheet(manifest, sheet, tracksheet_decides=tracksheet_decides)
    for col, want in (("Subject", subjects), ("Task", tasks), ("Session", sessions),
                      ("Group", groups)):
        if want is not None:
            manifest = manifest[manifest[col].isin([str(w) for w in want])]
    return _sort_manifest(manifest), participants


def join_tracksheet(manifest: pd.DataFrame, sheet: Optional[Tracksheet], *,
                    tracksheet_decides: bool = False) -> pd.DataFrame:
    """Attach medication state, task order and the log's own notes to each recording.

    Condition is not always ON or OFF: this study's log also says CBT for one
    participant's later sessions, a treatment arm rather than a blank. The
    label is kept verbatim as ``Condition``; ``OnMedNum`` is 1/0 for ON/OFF and
    NaN otherwise. ``ConditionDetail`` splits ON by whether the session ran
    before or after that day's dose, which is pharmacologically closer to
    OFF when it was before.
    """
    m = manifest.copy().reset_index(drop=True)
    n = len(m)
    blank = {"SessionDate": [None] * n, "OnMed": "", "MedTiming": "", "TaskOrder": "",
             "TracksheetSession": "", "Note": "", "InTracksheet": False, "Incomplete": False,
             "Unnumbered": False, "OnMedNum": np.nan, "Condition": "", "ConditionDetail": ""}
    for col, value in blank.items():
        m[col] = value
    if sheet is None or sheet.sessions.empty or n == 0:
        return m

    log = sheet.sessions
    log_day = pd.to_datetime(log.Date).dt.normalize()
    mismatch = 0
    for i in range(n):
        cand = np.flatnonzero(log.Subject.to_numpy() == m.Subject[i])
        if not cand.size:
            continue
        j = None
        recorded = m.RecordedAt[i]
        if recorded is not None and not pd.isna(recorded):
            gap = (log_day.iloc[cand] - pd.Timestamp(recorded).normalize()).abs().dt.days
            if gap.notna().any() and gap.min() <= 1:
                j = cand[int(np.nanargmin(gap.to_numpy(dtype=float)))]
        if j is None:
            same = cand[log.Session.to_numpy()[cand] == m.Session[i]]
            j = same[0] if same.size else None
        if j is None:
            continue
        row = log.iloc[j]
        m.loc[i, ["SessionDate", "OnMed", "MedTiming", "TaskOrder", "TracksheetSession",
                  "Note"]] = [row.Date, row.OnMed, row.MedTiming, row.TaskOrder, row.Session,
                              row.Note]
        m.loc[i, "InTracksheet"] = True
        m.loc[i, "Unnumbered"] = row.Session == ""
        m.loc[i, "Incomplete"] = any(w in row.Note.lower() for w in ("incomplete", "partial"))
        if row.Session and row.Session != m.Session[i]:
            mismatch += 1

    m["Condition"] = m.OnMed
    m["OnMedNum"] = m.OnMed.map({"ON": 1.0, "OFF": 0.0})
    timing = m.MedTiming.str.strip().str.lower()
    on = m.Condition == "ON"
    m["ConditionDetail"] = m.Condition.where(~(on & timing.str.startswith("after")), "ON-AfterMed")
    m["ConditionDetail"] = m.ConditionDetail.where(~(on & timing.str.startswith("before")),
                                                   "ON-BeforeMed")
    _report_join(m, mismatch)
    if tracksheet_decides:
        m = m[m.InTracksheet & ~m.Incomplete & ~m.Unnumbered].reset_index(drop=True)
    return m


def data_quality(manifest: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """What is questionable about the recordings in a manifest.

    The pipeline keeps every recording it finds and annotates the doubtful
    ones. That is only honest if the flags are somewhere they will be seen
    afterwards: warnings scroll past in a long batch, this table goes into
    the run report.

    Flags:

    ``Unfinished``       the app did not close the file cleanly (on this study
                         the same length as clean files, but worth knowing)
    ``Incomplete``       the log marked the attempt incomplete
    ``Unnumbered``       the log has a row for the date but no session number
    ``NoTracksheetRow``  no row matched, so no medication state
    ``NoMedState``       medication state unknown for any reason
    ``ShortForTask``     under 70 % of the task's median size (needs 3+ files)
    ``NoBehaviour``      no behavioural companion: fine for connectivity, but
                         no block average or GLM

    Returns
    -------
    flagged : DataFrame
        One row per flagged recording: Subject, Session, Task, whichever of
        RecordedAt / TracksheetSession / OnMed / Note exist, and the flags.
    summary : str
        A printable summary of the counts.
    """
    if manifest.empty:
        return pd.DataFrame(), ""
    n = len(manifest)

    def flag(name: str, default: bool) -> np.ndarray:
        return (manifest[name].fillna(False).to_numpy(dtype=bool) if name in manifest
                else np.full(n, default))

    flags = {"Unfinished": flag("Unfinished", False),
             "Incomplete": flag("Incomplete", False),
             "Unnumbered": flag("Unnumbered", False),
             "NoTracksheetRow": ~flag("InTracksheet", True),
             "NoMedState": (manifest.OnMedNum.isna().to_numpy() if "OnMedNum" in manifest
                            else np.zeros(n, bool)),
             "ShortForTask": _short_for_task(manifest),
             "NoBehaviour": ~flag("HasBehaviour", True)}
    any_flag = np.logical_or.reduce(list(flags.values()))

    cols = ["Subject", "Session", "Task"] + [c for c in ("RecordedAt", "TracksheetSession",
                                                          "OnMed", "Note") if c in manifest]
    flagged = manifest.loc[any_flag, cols].reset_index(drop=True)
    for name, mask in flags.items():
        flagged[name] = mask[any_flag]

    labels = {"Unfinished": "unfinished file", "Incomplete": "log marked incomplete",
              "Unnumbered": "log row unnumbered", "NoTracksheetRow": "no tracksheet row",
              "NoMedState": "medication state unknown", "ShortForTask": "short for its task",
              "NoBehaviour": "no behavioural companion"}
    n_sessions = (manifest.Subject + "|" + manifest.Session).nunique()
    lines = [f"Recordings: {n} across {manifest.Subject.nunique()} subject(s) and "
             f"{n_sessions} session(s).",
             f"Flagged   : {int(any_flag.sum())} (all are still analysed)."]
    lines += [f"  {labels[k]:<26} {int(v.sum())}" for k, v in flags.items() if v.any()]
    if "OnMedNum" in manifest:
        med = manifest.OnMedNum
        lines.append(f"Medication: {int((med == 1).sum())} ON, {int((med == 0).sum())} OFF, "
                     f"{int(med.isna().sum())} unknown.")
    return flagged, "\n".join(lines)


def session_times(manifest: pd.DataFrame,
                  data_root: Optional[str | Path] = None) -> tuple[pd.DataFrame, str]:
    """When each session was recorded: ``StartTime`` and ``EndTime`` added to the manifest.

    Sources, best first; the second return value says which one was used:

    ``'filename timestamp'``  the time the tablet wrote into the filename
                              (``RecordedAt``). Machine-written, exact.
    ``'sessions.csv'``        ``Subject,Session,StartTime[,EndTime]`` at the
                              study root. Exact.
    ``'filename'``            a date in the file or folder name. Day accuracy:
                              fine for day-scale alignment rules, not hour-scale.
    ``'filedate'``            the modification time, which may be when the
                              file was exported or synced, days later. A weak
                              fallback, and a warning says so.

    Without an end time a session is assumed to last 45 minutes.
    """
    out = manifest.copy().reset_index(drop=True)
    out["StartTime"] = pd.Series([pd.NaT] * len(out), dtype="datetime64[ns]")
    out["EndTime"] = pd.Series([pd.NaT] * len(out), dtype="datetime64[ns]")
    if out.empty:
        return out, "none"

    source = "none"
    if "RecordedAt" in out:
        recorded = pd.to_datetime(out.RecordedAt)
        if recorded.notna().any():
            out["StartTime"] = recorded
            source = "filename timestamp"

    if data_root is not None and (Path(data_root) / "sessions.csv").is_file():
        source = _times_from_sessions_csv(out, Path(data_root) / "sessions.csv", source)

    todo = out.StartTime.isna()
    if todo.any():
        from_name = out.File[todo].map(_date_from_name)
        out.loc[todo, "StartTime"] = pd.to_datetime(from_name)
        if from_name.notna().any() and source == "none":
            source = "filename"

    todo = out.StartTime.isna()
    if todo.any():
        out.loc[todo, "StartTime"] = out.File[todo].map(
            lambda f: datetime.fromtimestamp(Path(f).stat().st_mtime) if Path(f).exists()
            else pd.NaT)
        if source == "none":
            source = "filedate"
            warnings.warn(
                "Session times came from file modification dates: no sessions.csv and no date "
                "in the filenames. That is when the file was last written, not when the "
                "recording happened. Day-scale alignment rules are usually still fine; "
                "hour-scale ones are not. Write a sessions.csv at the study root to fix this.",
                stacklevel=2)

    need = out.EndTime.isna() & out.StartTime.notna()
    out.loc[need, "EndTime"] = out.StartTime[need] + timedelta(minutes=SESSION_MINUTES)
    return out, source


def _times_from_sessions_csv(out: pd.DataFrame, path: Path, source: str) -> str:
    table = pd.read_csv(path, dtype=str)
    table = _rename_alias(table, ("subject", "participant", "id"), "Subject")
    table = _rename_alias(table, ("session", "session_id", "visit"), "Session")
    table = _rename_alias(table, ("starttime", "start", "datetime", "timestamp", "date"),
                          "StartTime")
    table = _rename_alias(table, ("endtime", "stop", "finish"), "EndTime")
    if not {"Subject", "Session", "StartTime"} <= set(table.columns):
        warnings.warn("sessions.csv needs Subject, Session and StartTime columns; ignoring it.",
                      stacklevel=3)
        return source
    table.index = table.Subject.str.strip() + "|" + table.Session.str.strip()
    key = out.Subject.astype(str) + "|" + out.Session.astype(str)
    listed = key.isin(table.index)
    for col in ("StartTime", "EndTime"):
        if col in table:
            parsed = pd.to_datetime(table[col], errors="coerce", format="mixed")
            out.loc[listed, col] = key[listed].map(parsed).to_numpy()
    if not listed.all():
        warnings.warn(f"{int((~listed).sum())} of {len(listed)} recordings are not listed in "
                      "sessions.csv and fall back to the file date.", stacklevel=3)
    return "sessions.csv" if listed.any() else source


def _date_from_name(file: str) -> Optional[datetime]:
    path = Path(file)
    for text in (path.stem, str(path.parent)):
        m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", text)
        if m:
            try:
                return datetime(*map(int, m.groups()))
            except ValueError:
                continue
    return None


def _short_for_task(manifest: pd.DataFrame) -> np.ndarray:
    short = np.zeros(len(manifest), dtype=bool)
    if not {"Bytes", "Task"} <= set(manifest.columns):
        return short
    for _, idx in manifest.groupby("Task").groups.items():
        sizes = manifest.Bytes[idx]
        positive = sizes[sizes > 0]
        if positive.size >= 3:
            short[manifest.index.get_indexer(idx)] = (sizes < 0.7 * positive.median()).to_numpy()
    return short


def _load_tracksheet(root: Path, tracksheet: Optional[str | Path]) -> Optional[Tracksheet]:
    if tracksheet is None:
        found = sorted(p for p in root.glob("*racksheet*.xlsx")
                       if not p.name.startswith((".", "~$")))
        if not found:
            return None
        tracksheet = found[0]
    try:
        return read_tracksheet(tracksheet)
    except (OSError, ValueError) as err:
        warnings.warn(f"Tracksheet {tracksheet} could not be read ({err}); continuing without "
                      "it.", stacklevel=3)
        return None


def _report_join(m: pd.DataFrame, mismatch: int) -> None:
    odd = m.Condition[~m.Condition.isin(["ON", "OFF", ""])]
    if not odd.empty:
        counts = ", ".join(f"{k} ({v})" for k, v in sorted(odd.value_counts().items()))
        warnings.warn(f"The tracksheet records condition(s) that are neither ON nor OFF: "
                      f"{counts}. They are kept in Condition but OnMedNum is NaN for them, so "
                      "an on-versus-off analysis will leave them out.", stacklevel=4)
    if mismatch:
        warnings.warn(f"{mismatch} recording(s) carry a session number that differs from the "
                      "one the tracksheet gives that date. The filename number is kept; "
                      "TracksheetSession holds the other.", stacklevel=4)
    if (~m.InTracksheet).any():
        warnings.warn(f"{int((~m.InTracksheet).sum())} recording(s) have no tracksheet row for "
                      "their date, so they carry no medication state. They are analysed anyway: "
                      "a missing row in a hand-kept log is not evidence the recording is bad.",
                      stacklevel=4)
    if m.Incomplete.any():
        warnings.warn(f"{int(m.Incomplete.sum())} recording(s) sit on a session the log marked "
                      "incomplete; kept and flagged.", stacklevel=4)
    if m.Unnumbered.any():
        warnings.warn(f"{int(m.Unnumbered.sum())} recording(s) sit on a tracksheet row that was "
                      "never given a session number; kept and flagged.", stacklevel=4)


def _sort_manifest(m: pd.DataFrame) -> pd.DataFrame:
    number = pd.to_numeric(m.Session, errors="coerce")
    keys = ["Subject", "_n", "Task"] if number.notna().all() else ["Subject", "Session", "Task"]
    return (m.assign(_n=number).sort_values(keys, kind="stable").drop(columns="_n")
            .reset_index(drop=True))


def _rename_alias(table: pd.DataFrame, aliases: Iterable[str], target: str) -> pd.DataFrame:
    if target in table:
        return table
    aliases = set(aliases)
    hit = next((c for c in table.columns if str(c).lower() in aliases), None)
    return table if hit is None else table.rename(columns={hit: target})


def _as_text(col: pd.Series) -> pd.Series:
    """Ids as text: ``90590`` read as a number (or as ``90590.0`` beside a blank) -> ``'90590'``."""
    def one(v: object) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v).strip()
    return col.map(one).astype(object)
