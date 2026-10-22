"""Tracksheet, participants, manifest and data-quality flags, against stored reference outputs."""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic import write_discovery_study, write_sensor_stub
from wnirs.io import (build_manifest, data_quality, read_participants, read_tracksheet,
                      session_times)

FIXTURES = Path(__file__).parent / "fixtures" / "reference"
TEXT = {"Subject": str, "Session": str, "TracksheetSession": str}


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    # The discovery test checks this tree's fingerprint against the fixtures.
    return write_discovery_study(tmp_path_factory.mktemp("manifest"))


@pytest.fixture(scope="module")
def manifest(study):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, participants = build_manifest(study)
    m["File"] = [Path(f).relative_to(study.resolve()).as_posix() for f in m.File]
    return m, participants


def _same_text(py: pd.Series, ml: pd.Series) -> None:
    assert py.fillna("").astype(str).tolist() == ml.fillna("").astype(str).tolist(), py.name


def _same_flag(py: pd.Series, ml: pd.Series) -> None:
    assert py.astype(bool).tolist() == ml.astype(bool).tolist(), py.name


def _same_day(py: pd.Series, ml: pd.Series) -> None:
    assert pd.to_datetime(py).dt.normalize().equals(pd.to_datetime(ml).dt.normalize()), py.name


def test_tracksheet_matches_reference(study):
    sheet = read_tracksheet(study / "synthetic_tracksheet.xlsx")
    ml = pd.read_csv(FIXTURES / "io_tracksheet_sessions.csv", dtype=TEXT, keep_default_na=False)
    py = sheet.sessions
    for col in ("Subject", "Session", "Weekday", "OnMed", "MedTiming", "TaskOrder", "Note"):
        _same_text(py[col], ml[col])
    _same_flag(py.Valid, ml.Valid)
    _same_day(py.Date, ml.Date)
    np.testing.assert_array_equal(py.TasksDone, ml.TasksDone.astype(float))
    np.testing.assert_array_equal(py.RestDone, ml.RestDone.astype(float))

    mp = pd.read_csv(FIXTURES / "io_tracksheet_participants.csv", dtype=str)
    for col in ("Subject", "Group", "Box"):
        _same_text(sheet.participants[col], mp[col])
    _same_day(sheet.participants.StartDate, mp.StartDate)


def test_manifest_matches_reference(manifest):
    py = manifest[0]
    ml = pd.read_csv(FIXTURES / "io_manifest.csv", dtype=TEXT, keep_default_na=False)
    assert list(py.columns) == list(ml.columns)
    for col in ("Subject", "Session", "Task", "Group", "Format", "File", "OnMed", "MedTiming",
                "TaskOrder", "TracksheetSession", "Note", "Condition", "ConditionDetail"):
        _same_text(py[col], ml[col])
    for col in ("Unfinished", "HasBehaviour", "InTracksheet", "Incomplete", "Unnumbered"):
        _same_flag(py[col], ml[col].astype(int))
    for col in ("Bytes", "Age", "OnMedNum"):
        np.testing.assert_array_equal(py[col].astype(float),
                                      pd.to_numeric(ml[col], errors="coerce"))
    has_date = ml.SessionDate != "NaT"
    _same_day(py.SessionDate[has_date], ml.SessionDate[has_date])
    assert py.SessionDate[~has_date].isna().all()


def test_data_quality_matches_reference(manifest):
    flagged, summary = data_quality(manifest[0])
    ml = pd.read_csv(FIXTURES / "io_data_quality.csv", dtype=TEXT, keep_default_na=False)
    assert list(flagged.columns) == list(ml.columns)
    for col in ("Subject", "Session", "Task", "TracksheetSession", "OnMed", "Note"):
        _same_text(flagged[col], ml[col])
    for col in ("Unfinished", "Incomplete", "Unnumbered", "NoTracksheetRow", "NoMedState",
                "ShortForTask", "NoBehaviour"):
        _same_flag(flagged[col], ml[col])
    assert summary == (FIXTURES / "io_data_quality.txt").read_text().rstrip("\n")


def test_tracksheet_annotates_by_date(manifest):
    m = manifest[0].set_index(["Subject", "Session", "Task"])
    # Session 9 was recorded on the date the log calls session 10: matched, both kept.
    assert m.loc[("90001", "9", "Flanker"), "TracksheetSession"] == "10"
    assert m.loc[("90001", "2", "NBack"), "ConditionDetail"] == "ON-AfterMed"
    assert m.loc[("90001", "5", "NBack"), "ConditionDetail"] == "ON-BeforeMed"
    assert np.isnan(m.loc[("90001", "3", "Flanker"), "OnMedNum"])       # CBT is kept, not ON/OFF
    assert m.loc[("90001", "4", "GoNogo"), ["Unnumbered", "Incomplete"]].tolist() == [True, True]
    assert not m.loc[("90002", "1", "NBack"), "InTracksheet"]


def test_participants_and_filters(study, manifest):
    _, participants = manifest
    assert participants.set_index("Subject").Age.to_dict() == {"90001": 31, "90002": 29}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, _ = build_manifest(study, tasks=["NBack"], groups=["Patient"])
        decided, _ = build_manifest(study, tracksheet_decides=True)
        without, _ = build_manifest(study, exclude=[study / "90002"])
    assert set(m.Task) == {"NBack"} and set(m.Subject) == {"90001"}
    assert decided.InTracksheet.all() and not decided.Unnumbered.any()
    assert set(without.Subject) == {"90001"}


def test_sessions_sort_numerically(tmp_path):
    tab = tmp_path / "90590"
    for s in (1, 2, 10):
        write_sensor_stub(tab / f"Rest_SensorData_Session{s}_90590_202607{s:02d}_090000.csv", 300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, participants = build_manifest(tmp_path)
    assert m.Session.tolist() == ["1", "2", "10"]
    assert (m.Group == "unassigned").all() and participants.Subject.tolist() == ["90590"]


def test_participants_file(tmp_path):
    assert read_participants(tmp_path).empty
    (tmp_path / "participants.tsv").write_text("participant_id\tarm\tAge\n90606\tPatient\t31\n"
                                               "90610\t\t29\n")
    p = read_participants(tmp_path)
    assert p.Subject.tolist() == ["90606", "90610"]
    assert p.Group.tolist() == ["Patient", "unassigned"]
    (tmp_path / "participants.csv").write_text("Subject\n1\n1\n")
    with pytest.raises(ValueError, match="more than once"):
        read_participants(tmp_path)
    (tmp_path / "participants.csv").write_text("Name\nx\n")
    with pytest.raises(ValueError, match="no Subject column"):
        read_participants(tmp_path / "participants.csv")


def test_session_times_prefer_the_tablet_timestamp(manifest):
    times, source = session_times(manifest[0])
    assert source == "filename timestamp"
    assert (times.StartTime == pd.to_datetime(manifest[0].RecordedAt)).all()
    assert ((times.EndTime - times.StartTime) == timedelta(minutes=45)).all()


def test_session_times_fallbacks(tmp_path):
    m = pd.DataFrame({"Subject": ["S01", "S01", "S02"], "Session": ["1", "2", "1"],
                      "File": [str(tmp_path / "S01_2025-03-04_session_1.csv"),
                               str(tmp_path / "S01_session_2.csv"),
                               str(tmp_path / "S02_session_1.csv")]})
    for f in m.File:
        Path(f).write_text("x")
    (tmp_path / "sessions.csv").write_text("subject,session,start,finish\n"
                                           "S02,1,2025-03-05T09:12,2025-03-05T09:40\n")
    with pytest.warns(UserWarning, match="not listed in sessions.csv"):
        times, source = session_times(m, tmp_path)
    assert source == "sessions.csv"
    assert times.StartTime[2] == datetime(2025, 3, 5, 9, 12)
    assert times.EndTime[2] == datetime(2025, 3, 5, 9, 40)
    assert times.StartTime[0] == datetime(2025, 3, 4)               # from the filename
    assert times.StartTime[1].date() == datetime.now().date()        # file date, last resort

    with pytest.warns(UserWarning, match="file modification dates"):
        _, source = session_times(m.iloc[1:2])
    assert source == "filedate"
