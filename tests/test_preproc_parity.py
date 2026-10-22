"""Each preprocessing step against reference outputs, on shared synthetic inputs.

The fixtures in tests/fixtures/reference/ hold the inputs and reference
outputs computed independently on them. Differences are reported relative
to the largest output magnitude. Everything agrees to floating-point rounding; the only steps that need more than ~1e-13 are the
ones that run a 0.01 Hz high-pass, whose third-order transfer function has
poles within 0.4 % of z = 1 and so magnifies rounding differences between
SciPy's ``butter`` and the reference design (the coefficients agree to 4e-16)
to ~1e-11 of the signal. The same filter design and ``filtfilt`` padding are
used, so the edges match as well as the interior.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import signal

import wnirs.preproc as P
from wnirs.preproc.pipeline import _unused_pairs
from wnirs.types import Raw

FIXTURES = Path(__file__).parent / "fixtures" / "reference"
EXACT = 1e-12            # relative tolerance for steps without a near-DC IIR filter
HIGHPASS = 1e-9          # relative tolerance once a 0.01 Hz high-pass is involved


@pytest.fixture(scope="module")
def fx():
    def load(name: str) -> dict[str, np.ndarray]:
        with np.load(FIXTURES / f"preproc_{name}.npz") as f:
            return dict(f)
    return load


def assert_matches(name: str, py, ml, rtol: float = EXACT, scale: float | None = None) -> None:
    py, ml = np.asarray(py, dtype=float), np.asarray(ml, dtype=float)
    assert py.shape == ml.shape, f"{name}: shape {py.shape} vs reference {ml.shape}"
    np.testing.assert_array_equal(np.isfinite(py), np.isfinite(ml), err_msg=f"{name}: NaN layout")
    ok = np.isfinite(ml)
    diff = np.abs(py - ml)[ok].max(initial=0.0)
    if scale is None:
        scale = max(np.abs(ml[ok]).max(initial=0.0), 1.0e-300)
    print(f"{name:24s} max |py - ref| = {diff:.2e}  (relative {diff / scale:.1e})")
    assert diff <= rtol * scale, f"{name}: max abs difference {diff:.3e} on scale {scale:.3e}"


# ---- Beer-Lambert ----------------------------------------------------------

def test_dpf(fx):
    m = fx("mbll")
    assert_matches("dpf", P.dpf(m["dpf_age"][:, None], m["dpf_lambda"][None, :]), m["dpf"])


def test_mbll_two_wavelengths(fx):
    m = fx("mbll")
    hbo, hbr = P.mbll(m["od"])
    assert_matches("mbll hbo", hbo, m["mbll_hbo"])
    assert_matches("mbll hbr", hbr, m["mbll_hbr"])


def test_mbll_three_wavelengths(fx):
    m = fx("mbll")
    hbo, hbr = P.mbll(m["od3"], (780, 805, 830), sd_distance=25, age=40)
    assert_matches("mbll3 hbo", hbo, m["mbll3_hbo"])
    assert_matches("mbll3 hbr", hbr, m["mbll3_hbr"])


# ---- filtering --------------------------------------------------------------

def test_butter_coefficients(fx):
    f = fx("filter")
    b, a = signal.butter(3, 0.01 / (float(f["fs"]) / 2), "highpass")
    assert_matches("butter b", b, f["butter_b"], 1e-15)
    assert_matches("butter a", a, f["butter_a"], 1e-15)


@pytest.mark.parametrize("key, band, median_win", [
    ("filt_band", (0.01, 0.09), 0),
    ("filt_high", (0.01, np.nan), 0),
    ("filt_low", (np.nan, 0.09), 0),
    ("filt_med5", (0.01, 0.09), 5),
    ("filt_med4", (0.01, 0.09), 4),
])
def test_bandpass(fx, key, band, median_win):
    f = fx("filter")
    rtol = EXACT if np.isnan(band[0]) else HIGHPASS
    assert_matches(key, P.bandpass(f["hb"], float(f["fs"]), band, median_win), f[key], rtol)


def test_bandpass_bridges_gaps(fx):
    f = fx("filter")
    assert_matches("filt_gap", P.bandpass(f["hb_gap"], float(f["fs"])), f["filt_gap"], HIGHPASS)


# ---- motion -----------------------------------------------------------------

@pytest.mark.parametrize("key, fs_key, variant", [
    ("tddr_pub", "motion_fs", "published"),
    ("tddr_full", "motion_fs", "fullband"),
    ("tddr_fs1", None, "published"),
])
def test_tddr(fx, key, fs_key, variant):
    m = fx("motion")
    fs = float(m[fs_key]) if fs_key else 1.0
    assert_matches(key, P.tddr(m["motion_x"], fs, variant), m[key])


def test_smoothing_spline(fx):
    m = fx("motion")
    assert_matches("smoothing_spline", P.smoothing_spline(m["csaps_y"], 0.99), m["csaps"])


def test_spline(fx):
    m = fx("motion")
    y, info = P.spline_correct(m["motion_x"], float(m["motion_fs"]))
    assert_matches("spline", y, m["spline"])
    np.testing.assert_array_equal(info["n_segments"], m["spline_nseg"])
    assert_matches("spline fraction", info["fraction"], m["spline_frac"])


@pytest.mark.parametrize("key, opts", [
    ("wavelet", {}),
    ("wavelet_iqr", {"iqr_thresh": 1.5}),
    ("wavelet_scaling", {"remove_scaling": True}),
])
def test_wavelet(fx, key, opts):
    m = fx("motion")
    y, info = P.wavelet_correct(m["motion_x"], **opts)
    assert_matches(key, y, m[key])
    if f"{key}_removed" in m:
        assert_matches(f"{key} removed", info["removed_fraction"], m[f"{key}_removed"])


@pytest.mark.parametrize("key, method", [
    ("sgolay", "sgolay"),
    ("splinesg", "splinesg"),
    ("pca", "pca"),
    ("chain", ["spline", "wavelet"]),
])
def test_motion_dispatch(fx, key, method):
    m = fx("motion")
    y, _ = P.motion_correct(m["motion_x"], float(m["motion_fs"]), method)
    assert_matches(key, y, m[key])


# ---- baseline shift ---------------------------------------------------------

def test_baseline_shift_ari(fx):
    b = fx("baseline")
    y, info = P.baseline_shift(b["motion_x"], float(b["motion_fs"]))
    assert_matches("baseline ari", y, b["bs_ari"])
    assert_matches("baseline ari weight", info["mean_weight"], b["bs_ari_weight"])
    np.testing.assert_array_equal(info["ar_order"], b["bs_ari_order"])


def test_baseline_shift_step(fx):
    b = fx("baseline")
    y, info = P.baseline_shift(b["motion_x"], float(b["motion_fs"]), method="step")
    assert_matches("baseline step", y, b["bs_step"])
    np.testing.assert_array_equal(info["n_shifts"], b["bs_step_n"])


def test_arfit(fx):
    b = fx("baseline")
    assert_matches("arfit", P.arfit(b["ar_x"], 20), b["arfit"])


# ---- regressions ------------------------------------------------------------

@pytest.mark.parametrize("mode, key", [("perchannel", "lp_pc"), ("mean", "lp_mean")])
def test_lp_regress(fx, mode, key):
    r = fx("regression")
    y, info = P.lp_regress(r["od"], r["od_ref"], mode)
    assert_matches(key, y, r[key])
    assert_matches(f"{key} beta", info["beta"], r[f"{key}_beta"])


@pytest.mark.parametrize("mode, key", [("samesource", "ssr_same"), ("mean", "ssr_mean")])
def test_ssr(fx, mode, key):
    r = fx("regression")
    y, info = P.ssr(r["hb"], r["layout_S"], r["layout_Type"] == 2, mode,
                    r["usable"].astype(bool), r["clip_mask"].astype(bool))
    assert_matches(key, y, r[key])
    assert_matches(f"{key} beta", info["beta"], r[f"{key}_beta"])
    assert_matches(f"{key} source", info["source"], r[f"{key}_source"])


# ---- dark level and the full pipeline ---------------------------------------

@pytest.fixture(scope="module")
def raw(fx) -> Raw:
    p = fx("pipeline")
    layout = pd.DataFrame({k: p[f"layout_{k}"].astype(int)
                           for k in ("S", "D", "C", "Row", "Column", "Type")})
    pair_index = pd.DataFrame(p["all_pairs_sd"].astype(int), columns=["S", "D"])
    pair_index["in_layout"] = pd.MultiIndex.from_frame(pair_index).isin(
        pd.MultiIndex.from_frame(layout[["S", "D"]]))
    T = p["intensity"].shape[0]
    fs = float(p["fs"])
    return Raw(subject="synthetic", task="blocks", session="1", file=Path("synthetic.csv"),
               time=np.arange(T) / fs, fs=fs, intensity=p["intensity"], layout=layout,
               reference=p["reference"], all_pairs=p["all_pairs"],
               all_pairs_index=pair_index)


def test_dark_level(fx, raw):
    p = fx("pipeline")
    dark, info = P.dark_level(raw.intensity, "unused", _unused_pairs(raw))
    assert_matches("dark unused", dark, p["dark_unused"])
    # The spread is a MAD of readings near 1.5, so its rounding is set by them.
    assert_matches("dark unused spread", info["spread"], p["dark_unused_spread"],
                   scale=float(np.max(dark)))
    assert_matches("dark quantile", P.dark_level(raw.intensity, "quantile")[0], p["dark_quantile"])


PIPELINE_CASES = {
    # flagging, dark level, OD and Beer-Lambert only
    "p0": dict(motion="none", filter=False, detrend=False, ssr="none"),
    # the defaults: TDDR, high-pass + display low-pass, detrend, same-source SSR
    "p1": {},
    "p2": dict(lp_regress="perchannel", baseline_shift=True, motion="spline", global_signal=True),
    "p3": dict(motion="wavelet", ssr="mean", dark="quantile", lowpass=False),
}


@pytest.mark.parametrize("case", PIPELINE_CASES)
def test_preprocess(fx, raw, case):
    p = fx("pipeline")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", P.PreprocWarning)
        hb = P.preprocess(raw, **PIPELINE_CASES[case])
    np.testing.assert_array_equal(hb.channel_status.astype(str), p[f"{case}_status"])
    rtol = EXACT if case == "p0" else HIGHPASS
    assert_matches(f"{case} od", hb.od, p[f"{case}_od"])
    assert_matches(f"{case} hbo", hb.hbo, p[f"{case}_hbo"], rtol)
    assert_matches(f"{case} hbr", hb.hbr, p[f"{case}_hbr"], rtol)
    if f"{case}_hbo_display" in p:
        assert_matches(f"{case} hbo display", hb.hbo_display, p[f"{case}_hbo_display"], rtol)
        assert_matches(f"{case} hbr display", hb.hbr_display, p[f"{case}_hbr_display"], rtol)
    else:
        assert hb.hbo_display is None and hb.hbr_display is None
