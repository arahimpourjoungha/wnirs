"""Channel QC and the session check against reference outputs, on the synthetic tablet files.

The reference tables hold 15 significant digits, which bounds the
agreement for most metrics. The ones built on a 0.01-0.2 Hz Butterworth
band-pass (SNR_dB, HbCorr) carry rounding differences of ~1e-10 in absolute
terms (dB, or correlation) from the filter's poles close to z = 1, as in the
preprocessing parity tests; on an SNR near 0 dB that is a large relative
error, so those two are held to an absolute tolerance.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic import FIXTURE_TASK_LENGTH, study_fingerprint, write_reader_study
from wnirs.io import read_brain_games
from wnirs.qc import channel_qc, metric_table, session_check
from wnirs.types import Hb

FIXTURES = Path(__file__).parent / "fixtures" / "reference"
RECORDINGS = ("flanker", "nback", "rest", "gonogo")
BAND_PASSED = {"SNR_dB", "HbCorr"}
GATES_K = {"LightOverNoise": 20, "Saturation": 0.01, "MotionRate": 5, "SignalAboveDark": 0.005}


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("qc")
    recordings = write_reader_study(root)
    expected = json.loads((FIXTURES / "io_qc_inputs.json").read_text())["reader"]
    assert study_fingerprint(root) == expected, "synthetic generator changed; reference fixtures are out of date"
    return root, recordings


@pytest.fixture(scope="module")
def qc(study):
    _, recordings = study
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, rec in recordings.items():
            raw = read_brain_games(rec.sensor_file, task_length=FIXTURE_TASK_LENGTH)
            hb = None
            if name == "flanker":
                hb = Hb(subject=raw.subject, task=raw.task, session=raw.session, time=raw.time,
                        fs=raw.fs, hbo=rec.hbo, hbr=rec.hbr, layout=raw.layout,
                        conditions=raw.conditions, channel_status=np.full(raw.n_channels, "ok"))
            out[name] = (raw, channel_qc(raw, hb, metrics="all"))
    return out


def compare(py: pd.DataFrame, ml: pd.DataFrame) -> None:
    for col in ("Channel", "Source", "Detector", "Label"):
        assert py[col].tolist() == ml[col].tolist(), col
    assert py.IsShort.tolist() == ml.IsShort.astype(bool).tolist()
    for name in metric_table().Name:
        if name not in ml:
            continue
        rtol, atol = (1e-9, 1e-8) if name in BAND_PASSED else (1e-12, 1e-12)
        np.testing.assert_allclose(py[name].to_numpy(float), ml[name].to_numpy(float),
                                   rtol=rtol, atol=atol, err_msg=name)
    assert py.Pass.tolist() == ml.Pass.astype(bool).tolist()
    assert py.Reason.tolist() == ml.Reason.fillna("").tolist()


@pytest.mark.parametrize("name", RECORDINGS)
def test_metrics_and_verdicts_match_reference(qc, name):
    compare(qc[name][1], pd.read_csv(FIXTURES / f"qc_{name}.csv"))


def test_user_override_is_merged_onto_the_gates(qc):
    raw = qc["gonogo"][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = channel_qc(raw, thresholds={"SCI": 0.5})
    ml = pd.read_csv(FIXTURES / "qc_gonogo_sci_gate.csv")
    compare(table, ml)
    assert table.attrs["thresholds"] == {**GATES_K, "SCI": 0.5}


def test_faulty_channels_fail_for_the_right_reason(study, qc):
    roles = study[1]["flanker"].roles
    table = qc["flanker"][1]
    reason = table.Reason.to_numpy()
    assert reason[roles["saturated"]] == "clipped at the detector ceiling"
    assert reason[roles["motion"]] == "motion"
    assert reason[roles["weak"]] == "light not resolvable above this channel's noise"
    assert reason[roles["dark"]].startswith("no light (optode not coupled)")
    healthy = np.setdiff1d(np.arange(len(table)), list(roles.values()))
    assert table.Pass.to_numpy()[healthy].all()


def test_task_metrics_need_hb(qc):
    with_hb, without = qc["flanker"][1], qc["rest"][1]
    assert np.isfinite(with_hb[["HbCorr", "CNR", "CBR", "BlockConsistency"]]).all().all()
    assert without[["HbCorr", "CNR", "CBR", "BlockConsistency"]].isna().all().all()
    # The synthetic HbO and HbR are coupled with opposite sign.
    assert (with_hb.HbCorr < -0.5).all()


def test_session_check_matches_reference(study):
    root, _ = study
    with pytest.warns(UserWarning, match="display slot"):
        ok, table = session_check(root)
    ml = pd.read_csv(FIXTURES / "qc_session_check.csv", keep_default_na=False)
    assert ok
    for col in ("File", "Task", "Verdict", "Advice"):
        assert table[col].tolist() == ml[col].tolist(), col
    for col in ("nLong", "nLongDark", "nShort", "nShortSaturated"):
        assert table[col].tolist() == ml[col].tolist(), col
    for col in ("MedianLong", "MedianShort"):
        np.testing.assert_allclose(table[col], ml[col], rtol=1e-13)
