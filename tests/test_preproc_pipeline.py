"""The full preprocess() chain on a synthetic recording with known content."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import signal

import wnirs.preproc as P
from synthetic import make_recording

pytestmark = pytest.mark.filterwarnings("ignore::wnirs.preproc.PreprocWarning")


@pytest.fixture(scope="module")
def rec():
    return make_recording()


@pytest.fixture(scope="module")
def hb(rec):
    return P.preprocess(rec.raw)


def good_longs(rec, hb):
    return np.flatnonzero(hb.is_long & (hb.channel_status == "ok"))


def mean_abs_corr(x, ref, cols):
    return np.mean([abs(np.corrcoef(x[:, c], ref)[0, 1]) for c in cols])


def test_channel_flags(rec, hb):
    np.testing.assert_array_equal(hb.channel_status, rec.expected_status)
    np.testing.assert_array_equal(hb.usable, ~np.isin(rec.expected_status, P.UNUSABLE))
    assert np.isnan(hb.hbo[:, ~hb.usable]).all()
    assert np.isfinite(hb.hbo[:, hb.usable]).all()


def test_flag_warnings_name_the_channels(rec):
    with pytest.warns(P.PreprocWarning, match="S2-D16"):
        P.preprocess(rec.raw)


def test_documented_defaults(hb):
    o = P.PreprocessOptions()
    assert (o.dark, o.motion, o.ssr, o.lp_regress) == ("unused", "tddr", "samesource", "none")
    assert (o.band, o.sat_level, o.sat_tol, o.sat_max_frac) == ((0.01, 0.09), 5.0, 0.02, 0.20)
    assert (o.sd_distance, o.age, o.median_win) == (30.0, 28.0, 0)
    assert o.filter and o.lowpass and o.detrend
    assert not (o.baseline_shift or o.global_signal)
    assert hb.info["dark"]["method"] == "unused"
    assert [s.split(" ")[0] for s in hb.steps] == [
        "flagged", "dark", "intensity", "motion", "modified", "high-pass", "additional",
        "linear", "short-separation"]


def test_recovers_the_task_response(rec, hb):
    longs = good_longs(rec, hb)
    assert mean_abs_corr(hb.hbo, rec.task, longs) > 0.7
    assert min(mean_abs_corr(hb.hbo, rec.task, [c]) for c in longs) > 0.5


def test_ssr_removes_the_systemic_component(rec, hb):
    without = P.preprocess(rec.raw, ssr="none")
    longs = good_longs(rec, hb)
    systemic = P.bandpass(P.detrend(rec.systemic), rec.raw.fs, (0.01, np.nan))
    before = mean_abs_corr(without.hbo, systemic, longs)
    after = mean_abs_corr(hb.hbo, systemic, longs)
    assert after < 0.5 * before
    assert mean_abs_corr(hb.hbo, rec.task, longs) > mean_abs_corr(without.hbo, rec.task, longs)
    applied = hb.info["ssr"]["hbo"]["applied"]
    np.testing.assert_array_equal(applied, hb.is_long & hb.usable)


def test_display_copy_is_low_passed_analysis_copy_is_not(rec, hb):
    def high_power(x):
        f, p = signal.welch(x[:, good_longs(rec, hb)], hb.fs, axis=0, nperseg=256)
        return p[f > 0.3].sum()
    assert high_power(hb.hbo_display) < 0.01 * high_power(hb.hbo)
    no_lp = P.preprocess(rec.raw, lowpass=False)
    assert no_lp.hbo_display is None
    np.testing.assert_array_equal(no_lp.hbo, hb.hbo)


def test_motion_correction_repairs_the_spike(rec, hb):
    raw_od = P.preprocess(rec.raw, motion="none", filter=False, detrend=False, ssr="none").od
    c = rec.spike_channel
    near = np.abs(rec.raw.time - rec.spike_time) < 3
    assert np.ptp(hb.od[near, c, 0]) < 0.3 * np.ptp(raw_od[near, c, 0])


def test_optional_steps_are_recorded(rec):
    hb = P.preprocess(rec.raw, lp_regress="perchannel", baseline_shift=True,
                      motion=("tddr", "wavelet"), global_signal=True)
    assert {"lp", "baseline", "motion", "global", "ssr"} <= hb.info.keys()
    assert "motion correction on OD: tddr then wavelet" in hb.steps
    assert len(hb.info["motion"]) == 2          # one entry per wavelength


def test_foreign_recordings_degrade_gracefully(rec):
    layout = rec.raw.layout.assign(Type=1)
    raw = dataclasses.replace(rec.raw, reference=None, all_pairs=None, all_pairs_index=None,
                              layout=layout, meta={"source": "snirf"})
    with pytest.warns(P.PreprocWarning) as caught:
        hb = P.preprocess(raw, lp_regress="perchannel")
    text = " ".join(str(w.message) for w in caught)
    assert "low-power reference" in text and "1st percentile" in text and "SSR" in text
    assert hb.info["opts"]["lp_regress"] == "none" and hb.info["opts"]["ssr"] == "none"
    assert hb.info["dark"]["method"] == "quantile"


def test_numeric_dark_level(rec):
    hb = P.preprocess(rec.raw, dark=np.array([1.511, 1.507]))
    assert hb.info["dark"]["method"] == "user-supplied"
    np.testing.assert_array_equal(hb.info["dark"]["level"], [1.511, 1.507])


def test_unknown_options_are_rejected(rec):
    with pytest.raises(TypeError):
        P.preprocess(rec.raw, motoin="spline")
