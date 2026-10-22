"""Behaviour of the preprocessing steps on signals with a known answer.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal
from scipy.interpolate import make_smoothing_spline

import wnirs.preproc as P
from synthetic import DARK, forward_od, motion_signals

pytestmark = pytest.mark.filterwarnings("ignore::wnirs.preproc.PreprocWarning")


def corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


# ---- Beer-Lambert ----------------------------------------------------------

@pytest.mark.parametrize("hbo, hbr", [(1.0, -0.4), (-2.5, 0.9), (0.0, 1.0)])
def test_mbll_inverts_forward_model(hbo, hbr):
    od = forward_od(np.array([[hbo]]), np.array([[hbr]]))
    o, r = P.mbll(od)
    assert o.item() == pytest.approx(hbo, abs=1e-9)
    assert r.item() == pytest.approx(hbr, abs=1e-9)


def test_mbll_has_no_crosstalk():
    # A pure HbO change must give exactly zero HbR, and the other way round.
    for truth in ([1.0, 0.0], [0.0, 1.0]):
        o, r = P.mbll(forward_od(np.array([[truth[0]]]), np.array([[truth[1]]]), age=35), age=35)
        np.testing.assert_allclose([o.item(), r.item()], truth, atol=1e-9)


def test_mbll_two_wavelength_reference_value():
    o, r = P.mbll(np.array([0.01, 0.02]).reshape(1, 1, 2))
    assert (o.item(), r.item()) == pytest.approx((1.315476, -0.105470), abs=1e-6)


def test_mbll_three_wavelengths_beat_two():
    lam = (780.0, 805.0, 830.0)
    truth = (6.0, -2.0)
    od = forward_od(np.array([[truth[0]]]), np.array([[truth[1]]]), lam)
    np.testing.assert_allclose([h.item() for h in P.mbll(od, lam)], truth, atol=1e-6)

    rng = np.random.default_rng(2)
    noisy = od + 5e-4 * rng.standard_normal((200, 1, 3))
    err2 = np.hypot(*(h[:, 0] - t for h, t in zip(P.mbll(noisy[..., :2], lam[:2]), truth)))
    err3 = np.hypot(*(h[:, 0] - t for h, t in zip(P.mbll(noisy, lam), truth)))
    assert np.median(err3) < np.median(err2)


def test_mbll_rejects_mismatched_wavelengths():
    with pytest.raises(ValueError, match="wavelength"):
        P.mbll(np.zeros((1, 1, 2)), (780, 805, 830))
    with pytest.raises(ValueError, match="outside"):
        P.extinction([1200.0])


def test_extinction_matches_prahl_table():
    np.testing.assert_allclose(P.extinction([750.0, 751.0]),
                               [[518.0, 1405.24], [525.6, 1460.28]])


# ---- filtering --------------------------------------------------------------

def test_bandpass_keeps_passband_and_cuts_the_rest():
    fs = 10.0
    t = np.arange(0, 200 + 1 / fs, 1 / fs)
    passband = np.sin(2 * np.pi * 0.05 * t)
    cardiac = np.sin(2 * np.pi * 1.0 * t)
    y = P.bandpass(passband + cardiac, fs)
    assert 0.8 < np.std(y) / np.std(passband) < 1.2
    assert np.sqrt(np.mean((y - passband) ** 2)) < 0.25


def test_bandpass_leaves_dead_columns_and_open_nyquist():
    x = np.column_stack([np.random.default_rng(0).standard_normal(500), np.full(500, np.nan)])
    with pytest.warns(P.PreprocWarning, match="Nyquist"):
        y = P.bandpass(x, 1.0, (0.01, 0.6))
    assert np.isnan(y[:, 1]).all()
    np.testing.assert_allclose(y[:, 0], P.bandpass(x[:, 0], 1.0, (0.01, np.nan)))


def test_detrend_removes_lines_and_keeps_gaps():
    t = np.arange(100.0)
    x = np.column_stack([3 + 0.5 * t, np.sin(t)])
    x[10:20, 1] = np.nan
    y = P.detrend(x)
    np.testing.assert_allclose(y[:, 0], 0.0, atol=1e-10)
    assert np.isnan(y[10:20, 1]).all()
    ok = np.isfinite(x[:, 1])
    np.testing.assert_allclose(y[ok, 1], signal.detrend(x[ok, 1]))


# ---- TDDR -------------------------------------------------------------------

def _step_signal():
    fs = 10.0
    t = np.arange(0, 100 + 1 / fs, 1 / fs)
    return fs, t, np.sin(2 * np.pi * 0.03 * t), 5.0 * (t >= 50)


def test_tddr_flattens_a_step_without_eating_the_slow_signal():
    fs, t, slow, step = _step_signal()
    y = P.tddr(slow + step, fs)

    def jump(v):
        return v[(t > 50) & (t < 60)].mean() - v[(t > 40) & (t < 50)].mean()
    # The published variant adds the high band back uncorrected, and the edge
    # of an instantaneous step lives there, so it does not go to zero.
    assert abs(jump(y) - jump(slow)) < 0.7
    assert corr(y, slow) > 0.8


def test_tddr_removes_a_spike():
    fs, t, slow, _ = _step_signal()
    spike = 4.0 * np.exp(-0.5 * ((t - 60) / 0.6) ** 2)
    y = P.tddr(slow + spike, fs)
    near = np.abs(t - 60) < 5
    assert np.abs(y - slow)[near].max() < 0.25 * spike.max()


def test_tddr_published_beats_fullband_on_shifts_and_uses_fs():
    fs, t, _, _ = _step_signal()
    rng = np.random.default_rng(11)
    clean = np.sin(2 * np.pi * 0.02 * t) + 0.3 * np.sin(2 * np.pi * 1.1 * t)
    shifted = clean + 0.05 * rng.standard_normal(t.size)
    for s in (300, 700):
        shifted[s:] += 3
    r_pub = corr(P.tddr(shifted, fs), clean)
    assert r_pub > corr(shifted, clean)
    assert r_pub > corr(P.tddr(shifted, fs, "fullband"), clean) - 0.02
    # fs drives the 0.5 Hz split; ignoring it would silently disable it.
    assert not np.array_equal(P.tddr(shifted, fs), P.tddr(shifted, 1.0))


def test_tddr_leaves_short_and_constant_channels():
    x = np.column_stack([np.full(50, 2.0), np.r_[np.arange(5.0), np.full(45, np.nan)]])
    np.testing.assert_array_equal(P.tddr(x, 10.0), x)


# ---- spline, wavelet, other motion methods ----------------------------------

def test_smoothing_spline_limits_and_scipy_agreement():
    y = np.random.default_rng(3).standard_normal(25)
    x = np.arange(25.0)
    np.testing.assert_allclose(P.smoothing_spline(y, 1.0), y, atol=1e-12)
    np.testing.assert_allclose(P.smoothing_spline(y, 0.0), np.polyval(np.polyfit(x, y, 1), x),
                               atol=1e-10)
    np.testing.assert_allclose(P.smoothing_spline(y, 0.9),
                               make_smoothing_spline(x, y, lam=0.1 / 0.9)(x), atol=1e-12)


def test_spline_corrects_artefacts_and_leaves_clean_data():
    x, clean = motion_signals()
    y, info = P.spline_correct(x, 10.0)
    np.testing.assert_array_equal(y[:, 0], x[:, 0])
    assert info["n_segments"][0] == 0 and (info["n_segments"][1:] > 0).all()

    def worst_jump(v, c):
        return np.abs(np.diff(v[:, c] - clean[:, c])).max()
    for c in (1, 2, 3):
        assert worst_jump(y, c) < 0.25 * worst_jump(x, c)
    assert corr(y[:, 2], clean[:, 2]) > 0.95 > corr(x[:, 2], clean[:, 2])


def test_wavelet_removes_spikes_and_spares_clean_traces():
    rng = np.random.default_rng(7)
    fs = 10.0
    t = np.arange(2000) / fs
    clean = np.sin(2 * np.pi * 0.02 * t) + 0.05 * rng.standard_normal(t.size)
    spiky = clean.copy()
    for s in (300, 700, 1200, 1600):
        spiky[s:s + 9] += 3

    def rmse(v):
        return np.sqrt(np.mean((v - clean) ** 2))
    y, info = P.wavelet_correct(spiky)
    assert rmse(y) < 0.5 * rmse(spiky)

    # Per-level thresholds: a step must not hide the fine-scale spikes.
    stepped = spiky.copy()
    stepped[1000:] += 3

    def rough(v):
        return np.sqrt(np.mean(np.diff(v) ** 2))
    assert rough(P.wavelet_correct(stepped)[0]) < 0.75 * rough(stepped)

    _, info_iqr = P.wavelet_correct(spiky, iqr_thresh=1.5)
    assert info_iqr["removed_fraction"][0] >= info["removed_fraction"][0]
    assert corr(P.wavelet_correct(clean)[0], clean) > 0.99


@pytest.mark.parametrize("method", [m for m in P.MOTION_METHODS if m != "pca"])
def test_every_motion_method_preserves_the_signal(method):
    fs = 10.0
    t = np.arange(0, 120 + 1 / fs, 1 / fs)
    clean = np.sin(2 * np.pi * 0.03 * t)
    x = clean.copy()
    x[499:502] += 6
    x[799:] += 3
    y, info = P.motion_correct(x, fs, method)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert info["method"] == [method]
    # Each method targets a different artefact, so the common requirement is
    # only that none makes things worse; the spike methods must shrink it.
    assert corr(y, clean) >= corr(x, clean) - 0.05
    if method in ("tddr", "spline", "splinesg", "sgolay"):
        window = slice(494, 510)
        assert np.abs(y - clean)[window].max() <= np.abs(x - clean)[window].max()


def test_pca_removes_a_shared_component():
    rng = np.random.default_rng(5)
    shared = np.cumsum(rng.standard_normal(500))
    local = rng.standard_normal((500, 6))
    y, info = P.pca_correct(local + shared[:, None] * rng.uniform(1, 2, 6))
    assert info["explained"][0] > 90
    assert max(abs(corr(y[:, c], shared)) for c in range(6)) < 0.3


def test_motion_rejects_unknown_methods():
    with pytest.raises(ValueError, match="homer_spline"):
        P.motion_correct(np.zeros(100), 10.0, "homer_spline")


# ---- baseline shift ---------------------------------------------------------

@pytest.mark.parametrize("method", ["ari", "step"])
def test_baseline_shift_removes_a_step(method):
    fs = 10.0
    t = np.arange(0, 120 + 1 / fs, 1 / fs)
    clean = np.sin(2 * np.pi * 0.03 * t)
    y, info = P.baseline_shift(clean + 4.0 * (t >= 60), fs, method=method)

    def jump(v):
        return v[(t > 60) & (t < 70)].mean() - v[(t > 50) & (t < 60)].mean()
    assert abs(jump(y) - jump(clean)) < 0.5
    if method == "step":
        assert info["n_shifts"][0] >= 1


def test_arfit_recovers_an_ar1():
    rng = np.random.default_rng(4)
    x = signal.lfilter([1.0], [1.0, -0.7], rng.standard_normal(4000))
    a = P.arfit(x, 10)
    assert a.size == 2 and -a[1] == pytest.approx(0.7, abs=0.05)
    white = signal.lfilter(a, 1.0, x)
    assert abs(corr(white[1:], white[:-1])) < 0.05


# ---- regressions ------------------------------------------------------------

def test_ssr_removes_shared_systemic_oscillation():
    fs = 5.0
    t = np.arange(0, 200 + 1 / fs, 1 / fs)
    scalp = np.sin(2 * np.pi * 0.04 * t)
    brain = np.sin(2 * np.pi * 0.02 * t + 1)
    x = np.column_stack([brain + 2 * scalp, scalp])
    y, info = P.ssr(x, np.array([1, 1]), np.array([False, True]))
    assert corr(y[:, 0], brain) > 0.95
    assert abs(corr(y[:, 0], scalp)) < 0.15
    np.testing.assert_array_equal(y[:, 1], x[:, 1])
    assert info["source"][0] == 1 and info["beta"][0] == pytest.approx(2.0, abs=0.1)


def test_ssr_falls_back_to_mean_short_and_skips_clipped_samples():
    rng = np.random.default_rng(8)
    T = 1000
    sys1, sys2 = rng.standard_normal((2, T))
    shorts = np.column_stack([sys1, sys2])
    long_ = 1.5 * (sys1 + sys2) / 2 + 0.1 * rng.standard_normal(T)
    x = np.column_stack([long_, shorts])
    clip = np.zeros((T, 3), dtype=bool)
    clip[:200, 1] = True
    x_bad = x.copy()
    x_bad[:200, 1] = 0.0                  # what the interpolation invented
    y, info = P.ssr(x_bad, np.array([3, 1, 2]), np.array([False, True, True]),
                    clip_mask=clip)
    assert info["source"][0] == 0         # source 3 has no short of its own
    assert info["beta"][0] == pytest.approx(1.5, abs=0.05)


def test_ssr_ignores_unusable_shorts():
    x = np.random.default_rng(1).standard_normal((300, 2))
    with pytest.warns(P.PreprocWarning, match="no usable short"):
        y, info = P.ssr(x, np.array([1, 1]), np.array([False, True]),
                        usable=np.array([True, False]))
    np.testing.assert_array_equal(y, x)
    assert not info["applied"].any()


def test_lp_regress_removes_ambient_drift():
    rng = np.random.default_rng(6)
    T, C = 800, 3
    drift = np.cumsum(rng.standard_normal(T)) * 0.01
    signal_ = rng.standard_normal((T, C, 2)) * 0.01
    od = signal_ + drift[:, None, None]
    ref = drift[:, None, None] + 0.001 * rng.standard_normal((T, C, 2))
    y, info = P.lp_regress(od, ref)
    assert info["applied"].all()
    assert abs(corr(y[:, 0, 0], drift)) < 0.1


def test_global_regress_needs_three_channels():
    rng = np.random.default_rng(9)
    g = np.sin(np.linspace(0, 20, 600))
    x = g[:, None] + 0.1 * rng.standard_normal((600, 4))
    y, info = P.global_regress(x, np.ones(4, dtype=bool))
    assert info["applied"].all() and np.abs(y).max() < 0.5
    with pytest.warns(P.PreprocWarning, match="fewer than 3"):
        y2, _ = P.global_regress(x, np.array([True, True, False, False]))
    np.testing.assert_array_equal(y2, x)


# ---- intensity, dark level, channel flags -----------------------------------

def test_dark_level_methods():
    rng = np.random.default_rng(0)
    intensity = 1.6 + 0.1 * rng.random((300, 4, 2))
    unused = DARK + 0.001 * rng.standard_normal((300, 6, 2))
    unused[:, 0] += 0.4                   # stray light on one pair
    dark, info = P.dark_level(intensity, "unused", unused)
    np.testing.assert_allclose(dark, DARK, atol=1e-3)
    assert info["method"] == "unused" and info["n_unused"] == 6

    with pytest.warns(P.PreprocWarning, match="falling back"):
        dark_q, info_q = P.dark_level(intensity, "unused", unused[:, :3])
    assert info_q["method"] == "quantile"
    np.testing.assert_allclose(dark_q, [np.percentile(intensity[..., w], 1, method="hazen")
                                        for w in range(2)])
    np.testing.assert_array_equal(P.dark_level(intensity, 1.5)[0], [1.5, 1.5])
    np.testing.assert_array_equal(P.dark_level(intensity, "none")[0], [0.0, 0.0])
    with pytest.raises(ValueError):
        P.dark_level(intensity, "median")


def test_optical_density_is_log_ratio_and_dark_channels_are_nan():
    I = np.ones((10, 2, 2))
    I[:, 0, 0] = np.linspace(1, 2, 10)
    I[:, 1, 1] = -0.01
    od = P.optical_density(I)
    np.testing.assert_allclose(od[:, 0, 0], -np.log10(I[:, 0, 0] / I[:, 0, 0].mean()))
    assert np.isnan(od[:, 1, 1]).all()
    np.testing.assert_array_equal(P.dark_channels(I), [False, True])


def test_flag_channels_tells_clipping_cases_apart():
    rng = np.random.default_rng(1)
    T = 500
    I = 2.0 + 0.1 * rng.standard_normal((T, 5, 2))
    I[:, 1] = 5.0                          # pinned at the ceiling
    I[::20, 2, 0] = 5.0                    # clips on 5 % of samples
    I[:, 3] = 4.7                          # close to the ceiling, never clips
    I[:, 4, 1] = np.nan                    # a wavelength never recorded
    status, repaired, clip = P.flag_channels(I)
    assert list(status) == ["ok", "saturated", "intermittentClip", "nearCeiling", "missing"]
    assert np.isnan(repaired[:, [1, 4]]).all()
    assert clip[::20, 2].all() and repaired[::20, 2, 0].max() < 3.0
    np.testing.assert_array_equal(repaired[:, 0], I[:, 0])
