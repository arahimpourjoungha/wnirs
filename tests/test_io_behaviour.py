"""Per-session behaviour summaries against stored reference outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic import study_fingerprint, write_behaviour_study
from wnirs.io import behaviour_table, summarise_trials
from wnirs.io.behaviour import valid_name

FIXTURES = Path(__file__).parent / "fixtures" / "reference"


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = write_behaviour_study(tmp_path_factory.mktemp("behaviour"))
    expected = json.loads((FIXTURES / "io_qc_inputs.json").read_text())["behaviour"]
    assert study_fingerprint(root) == expected, "synthetic generator changed; reference fixtures are out of date"
    return root


@pytest.fixture(scope="module")
def table(root):
    return behaviour_table(root)


def test_matches_reference(table):
    ml = pd.read_csv(FIXTURES / "io_behaviour.csv", dtype={"Subject": str, "Session": str})
    ml = ml.drop(columns=["OnMedNum", "Condition", "SessionNo"])
    assert sorted(table.columns) == sorted(ml.columns)
    keys = ["Subject", "Session"]
    assert table[keys].values.tolist() == ml[keys].values.tolist()
    measures = [c for c in ml.columns if c not in ("Subject", "Session")]
    np.testing.assert_allclose(table[measures].to_numpy(float), ml[measures].to_numpy(float),
                               rtol=1e-12)


def test_rehearsal_skipped_and_later_duplicate_wins(root, table):
    assert set(table.Subject) == {"90001"}
    assert table.Session.tolist() == ["1", "2", "3", "4", "5", "6"]
    later = sorted(root.rglob("Fish_BehavioralData_Session5_*.xlsx"))[-1]
    expected = summarise_trials(pd.read_excel(later, sheet_name="TEST"), "Flanker")
    row = table.set_index("Session").loc["5"]
    assert all(row[f"Flanker_{k}"] == pytest.approx(v) for k, v in expected.items())


def test_zero_response_time_is_no_response():
    trials = pd.DataFrame({"accuracy": [1, 1, 0, 1], "responseTime": [0.5, 0, 0.7, "0.9"],
                           "omission": [0, 0, 1, 0], "stimuliPresented": ["A", "X", "X", "B"],
                           "commissionError": [0, 0, 1, 0]})
    m = summarise_trials(trials, "GoNogo")
    assert m["Accuracy"] == 75.0
    assert m["RT"] == pytest.approx(700.0)          # mean of 0.5, 0.7, 0.9 s; the 0 is left out
    assert m["FalseAlarmRate"] == 50.0 and m["Acc_NoGo"] == 50.0
    assert m["Acc_Go"] == 100.0 and m["RT_Go"] == pytest.approx(700.0)


def test_flanker_and_nback_contrasts():
    flanker = pd.DataFrame({"accuracy": [1, 1, 1, 0], "responseTime": [0.8, 1.0, 1.1, 1.3],
                            "stimuliType": ["congruent", "Congruent ", "incongruent",
                                            "incongruent"]})
    m = summarise_trials(flanker, "Flanker")
    assert m["RTcost"] == pytest.approx(1200.0 - 900.0)
    assert m["Acc_incongruent"] == 50.0

    nback = pd.DataFrame({"accuracy": [1, 1, 0, 1], "block": ["0-Back", "0-Back", "2-Back",
                                                             "2-Back"]})
    m = summarise_trials(nback, "NBack")
    assert m["LoadCost"] == 50.0 and "RT" not in m


def test_no_accuracy_column_means_no_summary():
    assert summarise_trials(pd.DataFrame({"responseTime": [0.5]}), "Flanker") == {}


def test_valid_names():
    assert valid_name("0-Back") == "x0_Back"
    assert valid_name("2-Back") == "x2_Back"
    assert valid_name("go trial") == "goTrial"


def test_empty_folder_warns(tmp_path):
    with pytest.warns(UserWarning, match="No behavioural sheets"):
        assert behaviour_table(tmp_path).empty
