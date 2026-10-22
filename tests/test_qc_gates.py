"""The gate system and the metrics' behaviour on constructed traces."""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import interpolate, signal

from synthetic import write_sensor_stub
from wnirs.io import load_layout
from wnirs.qc import (DEFAULT_THRESHOLDS, GATE_OPS, METRICS, gate_op, judge, merge_thresholds,
                      metric_table, resolve_metrics, session_check, session_check_report)
from wnirs.qc._signal import cpsd, interp_extrap
from wnirs.qc.gates import active_gates, warn_if_miscalibrated
from wnirs.qc.metrics import light_over_noise, motion_rate, to_od


def test_defaults_are_the_study_gate_set():
    gates = merge_thresholds()
    assert active_gates(gates, resolve_metrics()) == {
        "LightOverNoise": 20.0, "SignalAboveDark": 0.005, "Saturation": 0.01, "MotionRate": 5.0}
    for off in ("SCI", "SNR_dB", "CV", "HbCorr"):
        assert off in gates and gates[off] is None
    assert DEFAULT_THRESHOLDS["LightOverNoise"] == 20.0


def test_overrides_merge_rather_than_replace():
    gates = merge_thresholds({"SCI": 0.5, "MotionRate": math.nan, "Saturation": None})
    assert gates["SCI"] == 0.5 and gates["LightOverNoise"] == 20.0
    assert gates["MotionRate"] is None and gates["Saturation"] is None
    with pytest.raises(KeyError, match="No gate rule exists for PSP"):
        merge_thresholds({"PSP": 0.1})


def test_gate_directions_are_never_defaulted():
    assert gate_op("LightOverNoise") == ">=" and gate_op("MotionRate") == "<="
    assert set(GATE_OPS) >= set(DEFAULT_THRESHOLDS)
    with pytest.raises(KeyError, match="deliberately not defaulted"):
        gate_op("Kurtosis")


def test_judge_names_every_failed_gate_and_lets_nan_through():
    table = pd.DataFrame({"LightOverNoise": [50, 5, np.nan, 5],
                          "SignalAboveDark": [0.1, 0.1, 0.1, 0.001],
                          "Saturation": [0, 0, 0, 0.5]})
    passed, reason = judge(table, {"LightOverNoise": 20, "SignalAboveDark": 0.005,
                                   "Saturation": 0.01})
    assert passed.tolist() == [True, False, True, False]
    assert reason[1] == "light not resolvable above this channel's noise"
    assert reason[3] == ("no light (optode not coupled); light not resolvable above this "
                         "channel's noise; clipped at the detector ceiling")


def test_a_gate_rejecting_nearly_everything_warns():
    table = pd.DataFrame({"LightOverNoise": np.linspace(1, 10, 20)})
    with pytest.warns(UserWarning, match="LightOverNoise rejects 100%"):
        warn_if_miscalibrated(table, {"LightOverNoise": 20})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_miscalibrated(table, {"LightOverNoise": 5})


def test_catalogue():
    names = [m.name for m in METRICS]
    assert resolve_metrics("all") == names
    assert resolve_metrics("SCI") == ["SCI"]
    assert "SNR_simple" not in resolve_metrics()
    assert list(metric_table().columns) == ["Name", "Group", "Needs", "Direction", "Default",
                                           "Description"]
    with pytest.raises(ValueError, match="Unknown QC metric"):
        resolve_metrics(["SCI", "Brightness"])


def test_light_over_noise_is_in_units_of_the_channels_own_noise():
    rng = np.random.default_rng(0)
    noise = 0.001 * rng.standard_normal((2, 5000))
    slow = 0.2 * np.sin(np.linspace(0, 20, 5000))        # signal the SD would count as noise
    lon = light_over_noise(1.6 + slow + noise[0], 1.6 + slow + noise[1], dark=1.5)
    assert lon == pytest.approx(0.1 / 0.001, rel=0.05)
    frozen = np.full(100, 1.6)
    assert np.isnan(light_over_noise(frozen, frozen, dark=1.5))


def test_motion_rate_counts_events_not_samples():
    fs, n = 10.0, 6000                                    # ten minutes
    x = 2.0 + 0.001 * np.random.default_rng(1).standard_normal(n)
    for start in range(500, n - 100, 1200):               # five 2 s steps up and back
        x[start:start + 20] *= 0.6
    assert motion_rate(to_od(x), fs) == pytest.approx(10 / 10)
    assert motion_rate(to_od(np.full(n, 2.0)), fs) == 0
    assert np.isnan(motion_rate(np.full(10, 0.0), fs))


def test_session_check_tells_dark_from_saturated(tmp_path):
    with pytest.warns(UserWarning, match="display slot"):
        layout = load_layout("full")
    is_long = layout.Type.to_numpy() == 1

    def write_levels(path: Path, long_level: float, short_level: float) -> Path:
        names, vals = [], []
        for s, d, lng in zip(layout.S, layout.D, is_long):
            for wl in (740, 850):
                for panel in ("RP", "LP"):
                    names.append(f" S{s}-D{d}-{wl}nm-{panel}")
                    vals.append(long_level if lng else short_level)
        rows = [f"{t / 7.05:.3f}, 0," + ",".join(f"{v:.4f}" for v in vals) for t in range(1, 61)]
        path.write_text("Time,Stimulus," + ",".join(names) + "\n" + "\n".join(rows) + "\n")
        return path

    bad = write_levels(tmp_path / "NBack_SensorData_Session1_99999_20260101_090000.csv",
                       1.515, 4.99)
    ok, table = session_check(bad, layout)
    assert not ok and table.Verdict[0] == "FAIL"
    assert table.nLongDark[0] >= 0.9 * table.nLong[0]
    assert table.nShortSaturated[0] == table.nShort[0]
    assert "RAISE" in table.Advice[0]
    assert "FAIL" in session_check_report(table)

    good = write_levels(tmp_path / "NBack_SensorData_Session2_99999_20260102_090000.csv",
                        2.2, 2.6)
    ok, table = session_check(good, layout)
    assert ok and table.Verdict[0] == "PASS"
    assert "PASS" in session_check_report(table)


def test_session_check_needs_recordings(tmp_path):
    with pytest.raises(FileNotFoundError):
        session_check(tmp_path)
    write_sensor_stub(tmp_path / "NBack_SensorData_Session1_9_20260101_090000.csv", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="could be read"):
            session_check(tmp_path)


def test_gaps_are_bridged_by_linear_extrapolation():
    xp, fp = np.array([2.0, 3.0, 5.0, 6.0]), np.array([1.0, 2.0, 2.0, 4.0])
    x = np.arange(0.0, 9.0)
    expected = interpolate.interp1d(xp, fp, fill_value="extrapolate")(x)
    np.testing.assert_allclose(interp_extrap(xp, fp, x), expected)


def test_cpsd_follows_welch_segmenting():
    rng = np.random.default_rng(3)
    x, y = rng.standard_normal((2, 1000))
    pxy, freq = cpsd(x, y, nfft=256, fs=7.0)
    seg = int(1000 // 4.5)
    f_ref, p_ref = signal.csd(x, y, fs=7.0, window=signal.windows.hamming(seg, sym=True),
                              nperseg=seg, noverlap=seg // 2, nfft=256, detrend=False)
    np.testing.assert_allclose(freq, f_ref)
    np.testing.assert_allclose(np.abs(pxy), np.abs(p_ref), rtol=1e-12)
    # Segments longer than nfft are wrapped rather than rejected.
    assert np.isfinite(cpsd(*rng.standard_normal((2, 5000)), nfft=256, fs=7.0)[0]).all()
