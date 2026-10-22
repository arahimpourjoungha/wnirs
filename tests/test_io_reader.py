"""The Brain Games tablet reader against stored reference outputs, plus its own rules.

Inputs are the synthetic tablet files of tests/synthetic.py, regenerated here
and checked against the fingerprint of the files the reference outputs were
computed from.
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic import (FIXTURE_TASK_LENGTH, study_fingerprint, tablet_name,
                       write_behaviour_workbook, write_reader_study, write_tablet_recording)
from wnirs.io import find_companion, parse_brain_games_name, read_brain_games, read_csv
from wnirs.io.layout import load_layout

FIXTURES = Path(__file__).parent / "fixtures" / "reference"
RECORDINGS = ("flanker", "nback", "rest", "gonogo")


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("reader")
    recordings = write_reader_study(root)
    expected = json.loads((FIXTURES / "io_qc_inputs.json").read_text())["reader"]
    assert study_fingerprint(root) == expected, (
        "synthetic generator changed; reference fixtures are out of date")
    return recordings


@pytest.fixture(scope="module")
def raws(study):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {k: read_brain_games(r.sensor_file, task_length=FIXTURE_TASK_LENGTH)
                for k, r in study.items()}


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURES / "io_reader.npz") as f:
        data = dict(f)
    return {k: {key.split("__", 1)[1]: v for key, v in data.items() if key.startswith(k + "__")}
            for k in RECORDINGS}


@pytest.mark.parametrize("name", RECORDINGS)
def test_samples_and_rate_match_reference(raws, reference, name):
    raw, ml = raws[name], reference[name]
    np.testing.assert_array_equal(raw.time, ml["time"])
    assert raw.fs == pytest.approx(float(ml["fs"]), rel=1e-14)


@pytest.mark.parametrize("name", RECORDINGS)
def test_channels_land_where_reference_puts_them(raws, reference, name):
    raw, ml = raws[name], reference[name]
    for field, arr in (("intensity", raw.intensity), ("reference", raw.reference)):
        np.testing.assert_allclose(arr.mean(axis=0), ml[f"{field}_mean"], rtol=1e-13)
        np.testing.assert_allclose(arr.std(axis=0, ddof=1), ml[f"{field}_std"], rtol=1e-10)
        np.testing.assert_array_equal(arr[0], ml[f"{field}_first"])
        np.testing.assert_array_equal(arr[-1], ml[f"{field}_last"])
        # The checksum weights sample k of the column-major array by k.
        weights = np.arange(1, arr.size + 1).reshape(arr.shape, order="F")
        assert np.sum(arr * weights) == pytest.approx(float(ml[f"{field}_checksum"]), rel=1e-12)
    np.testing.assert_array_equal(raw.led_power.mean(axis=0), ml["led_rp"])
    np.testing.assert_array_equal(raw.reference_led_power.mean(axis=0), ml["led_lp"])


@pytest.mark.parametrize("name", RECORDINGS)
def test_all_pairs_match_reference(raws, reference, name):
    raw, ml = raws[name], reference[name]
    np.testing.assert_array_equal(raw.all_pairs_index[["S", "D"]].to_numpy(), ml["pairs_sd"])
    np.testing.assert_array_equal(raw.all_pairs_index.in_layout.to_numpy(),
                                  ml["pairs_in_layout"].astype(bool))
    np.testing.assert_allclose(raw.all_pairs.mean(axis=0), ml["pairs_mean"], rtol=1e-13)


@pytest.mark.parametrize("name", RECORDINGS)
def test_conditions_match_reference(raws, name):
    raw = raws[name]
    ml = pd.read_csv(FIXTURES / f"io_conditions_{name}.csv")
    py = pd.DataFrame([(n, o, d) for n, on, du in zip(raw.conditions.name, raw.conditions.onsets,
                                                      raw.conditions.durations)
                       for o, d in zip(on, du)], columns=["Name", "Onset", "Duration"])
    assert py.Name.tolist() == ml.Name.tolist()
    np.testing.assert_allclose(py.Onset.astype(float), ml.Onset.astype(float), rtol=1e-14)
    np.testing.assert_allclose(py.Duration.astype(float), ml.Duration.astype(float), rtol=1e-12)


@pytest.mark.parametrize("name", ("flanker", "nback"))
def test_trials_match_reference(raws, name):
    ml = pd.read_csv(FIXTURES / f"io_trials_{name}.csv")
    py = raws[name].trials
    assert list(py.columns) == list(ml.columns)
    assert py.Condition.tolist() == ml.Condition.tolist()
    numeric = [c for c in py.columns if c != "Condition"]
    np.testing.assert_allclose(py[numeric].to_numpy(float), ml[numeric].to_numpy(float),
                               rtol=1e-12)


def test_trimming_matches_reference(raws, reference):
    for name in RECORDINGS:
        ml_cut = float(reference[name]["trimmed_at"])
        py_cut = raws[name].trimmed_at
        assert (py_cut is None and math.isnan(ml_cut)) or py_cut == ml_cut, name
        assert len(raws[name].trials) == int(reference[name]["n_trials"]), name
        assert raws[name].meta["unfinished"] == bool(reference[name]["unfinished"]), name


def test_overlong_recording_is_trimmed_to_task_length(study, raws):
    raw, full = raws["nback"], study["nback"]
    assert full.time[-1] > 1.25 * FIXTURE_TASK_LENGTH["NBack"]
    assert raw.trimmed_at == FIXTURE_TASK_LENGTH["NBack"]
    assert raw.time[-1] - raw.time[0] <= FIXTURE_TASK_LENGTH["NBack"]
    assert raw.time.size == np.sum(full.time - full.time[0] <= 100)
    kept = full.trials.Onset <= full.time[0] + 100
    assert len(raw.trials) == kept.sum() < len(full.trials)


def test_read_cap_counts_as_overlong(study, raws):
    # GoNogo at 10 s: the read stops at 2 * 10 * 20 + 1 rows, well short of the file,
    # and every trial (the first starts at 12 s) falls after the cut.
    raw = raws["gonogo"]
    assert raw.trimmed_at == 10
    assert raw.time.size < 401 < study["gonogo"].time.size
    assert raw.trials.empty and not len(raw.conditions)


def test_trimming_can_be_switched_off(study):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = read_brain_games(study["nback"].sensor_file, task_length=FIXTURE_TASK_LENGTH,
                               trim_factor=math.inf)
    assert raw.trimmed_at is None
    assert raw.time.size == study["nback"].time.size


def test_default_lengths_leave_a_normal_recording_alone(study):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = read_brain_games(study["nback"].sensor_file)
    assert raw.trimmed_at is None and raw.time.size == study["nback"].time.size


def test_recording_fields(raws, study):
    raw = raws["flanker"]
    assert (raw.subject, raw.session, raw.task) == ("90001", "1", "Flanker")
    assert raw.meta["behaviour_file"] == study["flanker"].behaviour_file
    assert raw.wavelengths == (750.0, 860.0)
    assert raw.intensity.shape == (study["flanker"].time.size, 37, 2)
    assert not np.any(raw.stim)
    assert raw.meta["missing_channels"] == []
    assert raws["rest"].meta["behaviour_file"] is None and not len(raws["rest"].conditions)


def test_missing_companion_means_no_conditions(tmp_path):
    rec = write_tablet_recording(tmp_path, "NBack", duration=40, behaviour=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw = read_brain_games(rec.sensor_file)
    messages = [str(w.message) for w in caught]
    assert any("No BehavioralData file" in m for m in messages)
    assert not any("plain CSV" in m for m in messages)
    assert not len(raw.conditions) and raw.trials.empty


def test_companion_is_chosen_by_session_then_time(tmp_path):
    rec = write_tablet_recording(tmp_path, "Flanker", session=4, duration=40)
    start = parse_brain_games_name(rec.sensor_file.stem).recorded_at
    # Same session number, three weeks later; and a near-in-time file of another session.
    for session, when in ((4, start + timedelta(days=21)), (5, start + timedelta(minutes=2))):
        write_behaviour_workbook(
            tmp_path / (tablet_name("Flanker", "BehavioralData", session, "90001", when)
                        + ".xlsx"), "Flanker", rec.trials, when)
    info = parse_brain_games_name(rec.sensor_file.stem)
    assert find_companion(tmp_path, info) == rec.behaviour_file

    rec.behaviour_file.unlink()
    with pytest.warns(UserWarning, match="days from the recording"):
        assert find_companion(tmp_path, info) is None


def test_explicit_behaviour_file_and_overrides(study):
    rec = study["flanker"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = read_brain_games(rec.sensor_file, behaviour_file=rec.behaviour_file,
                               subject="S01", session="7", correct_only=True)
    assert (raw.subject, raw.session) == ("S01", "7")
    assert (raw.trials.Accuracy == 1).all()
    assert len(raw.trials) == rec.trials.Accuracy.sum()


def test_label_options(tmp_path):
    rec = write_tablet_recording(tmp_path, "GoNogo", duration=40, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        split = read_brain_games(rec.sensor_file)
        pooled = read_brain_games(rec.sensor_file, gonogo_split=False)
    assert set(split.conditions.name) == {"Go", "NoGo"}
    assert pooled.conditions.name == ["SkipTheX"]
    expected = (np.array(rec.trials.Shown) == "X")
    assert split.trials.Condition.eq("NoGo").to_numpy().tolist() == expected.tolist()


def test_plain_dataLog_csv_takes_timing_from_stimulus(tmp_path):
    with pytest.warns(UserWarning):
        layout = load_layout("full")
    fs, n = 5.0, 200
    stim = np.zeros(n)
    stim[20:40], stim[100:120], stim[60:70] = 10, 10, 20
    header = ["Time", "Stimulus"]
    rows = [np.arange(n) / fs, stim]
    for s, d in zip(layout.S, layout.D):
        for tag in ("740nm_RP", "850nm_RP", "740nm_LP", "850nm_LP"):
            header.append(f"S{s}_D{d}_{tag}")
            rows.append(1.6 + 0.01 * np.sin(np.arange(n) / 7))
    path = tmp_path / "dataLog_NBack_S2.csv"
    pd.DataFrame(np.column_stack(rows), columns=header).to_csv(path, index=False)

    raw = read_csv(path, layout)
    assert (raw.task, raw.session) == ("NBack", "2")
    assert raw.conditions.name == ["0-Back", "2-Back"]
    np.testing.assert_allclose(raw.conditions.onsets[0], [4.0, 20.0])
    np.testing.assert_allclose(raw.conditions.durations[0], [3.8, 3.8])
    assert raw.fs == pytest.approx(fs)


def test_unreadable_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_brain_games(tmp_path / "missing.csv")
    bad = tmp_path / "NBack_SensorData_Session1_90001_20260105_100000.csv"
    bad.write_text("Clock,Stimulus, S1-D1-740nm-RP\n0.1,0,1.5\n0.2,0,1.5\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match='No "Time" column'):
            read_brain_games(bad)
