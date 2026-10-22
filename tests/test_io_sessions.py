"""Session discovery against stored reference outputs, and its rules one at a time."""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic import (make_trials, study_fingerprint, tablet_name, write_behaviour_workbook,
                       write_discovery_study, write_sensor_stub)
from wnirs.io import find_sessions

FIXTURES = Path(__file__).parent / "fixtures" / "reference"


@pytest.fixture(scope="module")
def discovery(tmp_path_factory):
    root = write_discovery_study(tmp_path_factory.mktemp("discovery"))
    expected = json.loads((FIXTURES / "io_qc_inputs.json").read_text())["discovery"]
    assert study_fingerprint(root) == expected, "synthetic generator changed; reference fixtures are out of date"
    return root


def found(root: Path) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sessions = find_sessions(root)
    sessions["File"] = [Path(f).relative_to(root.resolve()).as_posix() for f in sessions.File]
    return sessions


def test_matches_reference(discovery):
    py = found(discovery)
    ml = pd.read_csv(FIXTURES / "io_find_sessions.csv", dtype={"Subject": str, "Session": str})
    assert py.File.tolist() == ml.File.tolist()
    for col in ("Subject", "Session", "Task", "Format", "Bytes"):
        assert py[col].tolist() == ml[col].tolist(), col
    for col in ("Unfinished", "HasBehaviour"):
        assert py[col].tolist() == ml[col].astype(bool).tolist(), col
    assert [t.strftime("%Y-%m-%d %H:%M:%S") for t in py.RecordedAt] == ml.RecordedAt.tolist()


def test_each_rule(discovery):
    s = found(discovery).set_index(["Subject", "Session", "Task"])
    # The smaller attempt wins when only it has its behaviour on disk.
    assert "20260108_090000" in s.loc[("90001", "2", "NBack"), "File"]
    # Neither has behaviour: the bigger wins.
    assert "20260109_110000" in s.loc[("90001", "3", "Flanker"), "File"]
    # Session 5's behaviour is not session 15's, and 8 h away is not a companion.
    assert not s.loc[("90001", "5", "NBack"), "HasBehaviour"]
    assert not s.loc[("90001", "8", "Flanker"), "HasBehaviour"]
    # Rest needs no companion; unfinished files are kept, flagged.
    assert s.loc[("90001", "1", "Rest"), ["HasBehaviour", "Unfinished"]].tolist() == [True, True]
    # The folder overrides a mistyped id.
    assert "_90003_" in s.loc[("90002", "1", "NBack"), "File"]
    # Stub, header-only, headerless fragment, rehearsal: never sessions.
    files = " ".join(s.File)
    assert "Session6" not in files and "Session7" not in files
    assert "20260110_080000" not in files and "testuser" not in files
    # Questionnaires, metadata and another modality's exports: never recordings.
    assert all("SensorData" in f for f in s.File)


def test_test_login_never_lands_in_a_subject(tmp_path):
    # The rehearsal is the bigger file.
    tab = tmp_path / "90590" / "Tablet_backup"
    write_sensor_stub(tab / "NBack_SensorData_Session13_90590_20260727_090453.csv", 300)
    write_sensor_stub(tab / "NBack_SensorData_Session13_testuser_0327_20260629_074444.csv", 320)
    s = found(tmp_path)
    assert len(s) == 1 and "_90590_" in s.File[0]


def test_bigger_wins_and_stub_is_dropped(tmp_path):
    tab = tmp_path / "90590" / "Tablet_backup"
    for k in range(1, 4):
        write_sensor_stub(tab / f"NBack_SensorData_Session{k}_90590_2026070{k}_100401.csv", 300)
    write_sensor_stub(tab / "NBack_SensorData_Session4_90590_20260707_100401.csv", 300)
    write_sensor_stub(tab / "NBack_SensorData_Session4_90590_20260706_083125.csv", 30000)
    write_sensor_stub(tab / "NBack_SensorData_Session5_90590_20260708_100401.csv", 5)
    s = found(tmp_path)
    assert "20260706_083125" in s.File[s.Session == "4"].item()
    assert "5" not in set(s.Session)


def test_questionnaires_are_not_subjects(tmp_path):
    sub = tmp_path / "90589"
    write_sensor_stub(sub / "Tablet_backup" / "NBack_SensorData_Session1_90589_20260701_090000.csv",
                      300)
    for name in ("EMA_90589.csv", "90589_EMA_raw.csv", "90589_EMA_labels.csv",
                 "90589_CAARS_raw.csv", "90589_CAARS_labels.csv"):
        (sub / name).write_text("experiment_date,experiment_time,ema_focus\n2026-07-01,09:00,3\n")
    s = found(tmp_path)
    assert len(s) == 1 and s.Subject[0] == "90589"


def test_a_folder_holding_one_recording(tmp_path):
    # With the NIRduino naming.
    sub = tmp_path / "sub01"
    sub.mkdir()
    (sub / "2026_01_01_10_00_dataLog.csv").write_text(
        "Time,Stimulus,S1_D1_740nm_RP,S1_D1_850nm_RP\n0.0,0,1.6,1.6\n0.2,0,1.6,1.6\n")
    s = found(tmp_path)
    assert s[["Subject", "Session", "Task", "Format"]].values.tolist() == [
        ["sub01", "2026_01_01_10_00", "Naturalistic", "nirduino"]]
    (sub / "2026_01_02_10_00_dataLog.csv").write_text((sub / "2026_01_01_10_00_dataLog.csv")
                                                      .read_text())
    assert len(found(tmp_path)) == 2


def test_session_number_is_anchored(tmp_path):
    tab = tmp_path / "90593"
    write_sensor_stub(tab / "NBack_SensorData_Session5_90593_20260701_090000.csv", 300)
    when = datetime(2026, 7, 1, 9, 8)
    trials = make_trials("NBack", 60, np.random.default_rng(0))
    name = tablet_name("NBack", "BehavioralData", 15, "90593", when)
    write_behaviour_workbook(tab / f"{name}.xlsx", "NBack", trials, when)
    assert not found(tmp_path).HasBehaviour.item()


def test_not_a_folder(tmp_path):
    with pytest.raises(NotADirectoryError):
        find_sessions(tmp_path / "missing")
    assert find_sessions(tmp_path).empty
