"""Layouts and filename parsing, against stored reference outputs where they exist."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from wnirs.io import (builtin_layouts, canonical_task, load_layout, parse_brain_games_name,
                      parse_name)

FIXTURES = Path(__file__).parent / "fixtures" / "reference"


def test_full_layout_matches_reference():
    with pytest.warns(UserWarning, match="S8-D9 moved to slot 22"):
        layout = load_layout("full")
    ml = pd.read_csv(FIXTURES / "io_layout.csv")
    for col in ("S", "D", "Index", "Label", "GridSlot"):
        assert layout[col].tolist() == ml[col].tolist(), col
    assert (layout.Type == 2).sum() == 6


def test_loaded_layout_is_passed_through_without_new_warnings():
    with pytest.warns(UserWarning):
        layout = load_layout("full")
    assert load_layout(layout) is layout


def test_builtin_names_and_paths_resolve(tmp_path):
    assert {"full", "nirxsubset"} <= set(builtin_layouts())
    assert len(load_layout("nirxsubset")) == 26
    custom = tmp_path / "mine.csv"
    custom.write_text("S,D,C,Row,Column,Type\n1,1,1,1,0,1\n1,2,2,1,1,2\n")
    assert load_layout(custom).Label.tolist() == ["S1-D1", "S1-D2 (sc)"]


def test_layout_errors():
    with pytest.raises(FileNotFoundError, match="Built-in layouts: full"):
        load_layout("nope")
    with pytest.raises(ValueError, match="Type"):
        load_layout(pd.DataFrame({"S": [1], "D": [1], "C": [1], "Row": [1], "Column": [1]}))


def test_duplicate_display_slots_are_separated():
    # Two channels claiming slot 5 must not overwrite each other.
    table = pd.DataFrame({"S": [1, 2, 3], "D": [1, 2, 3], "C": [5, 5, 7], "Row": [1, 1, 1],
                          "Column": [5, 5, 7], "Type": [1, 1, 1]})
    with pytest.warns(UserWarning, match="no short-separation"):
        with pytest.warns(UserWarning, match="moved to slot"):
            layout = load_layout(table)
    assert layout.GridSlot.nunique() == 3
    assert layout.Index.tolist() == [1, 2, 3]


def test_brain_games_names_match_reference():
    ml = pd.read_csv(FIXTURES / "io_parse_names.csv", keep_default_na=False)
    for row in ml.itertuples():
        info = parse_brain_games_name(row.Stem)
        stamp = (info.recorded_at.strftime("%Y-%m-%d %H:%M:%S") if info.recorded_at else "")
        assert (info.task, info.app_task, info.kind, info.session, info.subject, stamp,
                info.unfinished) == (row.Task, row.AppTask, row.Kind, str(row.Session),
                                     str(row.Subject), row.RecordedAt, bool(row.Unfinished))


def test_tablet_and_portal_agree_on_task_names():
    for word, task in (("NBack", "NBack"), ("Fish", "Flanker"), ("SkipX", "GoNogo"),
                       ("Rest", "Rest"), ("Middle-fish Sensor data", "Flanker"),
                       ("N-back Sensor data", "NBack"), ("Skip The X", "GoNogo"), ("MIST", "")):
        assert canonical_task(word) == task
    info = parse_brain_games_name("Fish_SensorData_Session1_90590_20260629_101535")
    assert (info.task, info.app_task) == ("Flanker", "Fish")
    assert info.recorded_at == datetime(2026, 6, 29, 10, 15, 35)


def test_subject_with_underscore_is_not_shifted():
    info = parse_brain_games_name("NBack_SensorData_Session13_testuser_0327_20260629_074444")
    assert info.subject == "testuser_0327"
    assert info.session == "13"


def test_generic_names():
    assert parse_name("90606_session_1_Report", "subject") == "90606"
    assert parse_name("90606_session_12_Report", "session") == "12"
    assert parse_name("wNIRS_90599_BH1", "task") == "BH1"
    assert parse_name("dataLog_NBack", "task") == "NBack"
    assert parse_name("sub_S3_run", "session") == "3"
    assert parse_name("oddname", "subject") == "oddname"
    assert parse_name("oddname", "session") == "1"
    assert parse_name("oddname", "task") == "task"
    with pytest.raises(ValueError):
        parse_name("x", "age")


def test_unparseable_stem_gives_empty_fields():
    info = parse_brain_games_name("justone")
    assert (info.task, info.session, info.subject, info.recorded_at) == ("", "", "", None)
