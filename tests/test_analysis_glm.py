"""GLM machinery: HRF, design, AR fit, robust fit, AR-IRLS, contrasts, block average.

Behavioural checks on planted truths; numerical agreement with the stored
reference outputs is in test_analysis_reference.py.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from synthetic import analysis_layout, boxcar, glm_recording, null_recording
from wnirs.analysis import (arfit, ar_irls, block_average, contrast, default_contrasts, design_matrix, fdr_bh,
                            glm, hrf, parse_contrast, robust_fit, whiten)
from wnirs.types import Conditions, Hb


def test_hrf_peaks_at_six_seconds_with_unit_height():
    h, t = hrf(10.0)
    assert t[np.argmax(h)] == pytest.approx(6.0, abs=0.1)
    assert h.max() == pytest.approx(1.0)
    assert h.min() < 0, "the canonical shape has an undershoot"
    assert hrf(10.0, "gamma")[0].min() >= 0
    with pytest.raises(ValueError):
        hrf(10.0, "spm")


def test_design_regressors_are_unit_peak_and_named():
    hb = glm_recording()
    d = design_matrix(hb.time, hb.conditions, hb.fs, derivative=True, drift=2)
    assert d.names == ["0-Back", "0-Back:deriv", "2-Back", "2-Back:deriv", "intercept", "drift1", "drift2"]
    assert d.n_condition_regressors == 4
    assert np.max(np.abs(d.X[:, [0, 2]]), axis=0) == pytest.approx([1, 1])
    with pytest.raises(ValueError, match="nothing to model"):
        design_matrix(hb.time, Conditions(), hb.fs)


def test_design_warns_about_indistinguishable_conditions():
    t = np.arange(0, 300, 0.2)
    on = np.arange(20, 260, 30.0)
    cond = Conditions(["a", "b"], [on, on + 0.2], [np.full(on.size, 10.0)] * 2)
    with pytest.warns(UserWarning, match="correlate"):
        design_matrix(t, cond, 5.0)


def test_arfit_recovers_ar1_and_whitens():
    rng = np.random.default_rng(11)
    x = lfilter([1.0], [1.0, -0.8], rng.standard_normal(4000))
    a, order = arfit(x, 20)
    assert order >= 1
    assert -a[1] == pytest.approx(0.8, abs=0.05)
    w = whiten(a, x)[order + 1:]
    assert abs(np.corrcoef(w[:-1], w[1:])[0, 1]) < 0.1


def test_arfit_refuses_short_or_constant_series():
    assert arfit(np.ones(50), 5)[1] == 0
    assert arfit(np.arange(8.0), 5)[1] == 0


def test_robust_fit_ignores_outliers():
    rng = np.random.default_rng(3)
    X = np.column_stack((np.ones(300), rng.standard_normal(300)))
    y = X @ [1.0, 2.0] + 0.3 * rng.standard_normal(300)
    y[:15] += 20
    fit = robust_fit(X, y)
    assert fit.beta == pytest.approx([1.0, 2.0], abs=0.08)
    assert np.all(fit.w[:15] == 0), "gross outliers get zero bisquare weight"
    assert fit.dfe == 298


def test_glm_recovers_planted_beta():
    """testGLM: a clean block response is recovered with a large t."""
    fs, T = 5.0, 3000
    time = np.arange(T) / fs
    on = np.arange(30.0, 541.0, 60.0)
    reg = np.convolve(boxcar(time, on, 20.0), hrf(fs)[0])[:T]
    reg /= reg.max()
    rng = np.random.default_rng(7)
    hb = Hb("t", "t", "1", time, fs, 2.5 * reg[:, None] + 0.05 * rng.standard_normal((T, 1)),
            -1.25 * reg[:, None] + 0.05 * rng.standard_normal((T, 1)), analysis_layout(1),
            Conditions(["block"], [on], [np.full(on.size, 20.0)]), np.array(["ok"]))
    G = glm(hb, ar=0)
    assert G.beta[0, 0, 0] == pytest.approx(2.5, abs=0.05)
    assert G.beta[0, 0, 1] == pytest.approx(-1.25, abs=0.05)
    assert G.tstat[0, 0, 0] > 10


def test_glm_skips_excluded_channels_and_downweights_motion():
    G = glm(glm_recording())
    assert np.all(np.isnan(G.beta[:, 4])), "the QC-excluded channel stays NaN"
    assert np.all(np.isfinite(G.beta[:, :4]))
    spike = slice(int(150 * 5), int(153 * 5))
    assert np.nanmin(G.weights[spike, 2, 0]) == 0, "the artefact onset gets zero bisquare weight"
    assert np.nanmedian(G.weights[:, 2, 0]) > 0.8
    assert G.beta[1, 0, 0] == pytest.approx(1.5, abs=0.3)


def test_arirls_controls_false_positives_where_ols_does_not():
    """testNullRate: the check that catches a sign error in the whitening filter."""
    p_ar, p_ols = [], []
    for s in range(1, 41):
        hb = null_recording(s)
        p_ar.append(contrast(glm(hb, method="arirls"), [1, -1, 0, 0])["Pval"].iloc[0])
        p_ols.append(contrast(glm(hb, method="ols"), [1, -1, 0, 0])["Pval"].iloc[0])
    fp_ar, fp_ols = np.mean(np.array(p_ar) < 0.05), np.mean(np.array(p_ols) < 0.05)
    assert fp_ar <= 0.2
    assert fp_ols > fp_ar


def test_arirls_still_detects_a_real_response():
    """testPower: controlling false positives is easy if nothing is ever detected."""
    hb = null_recording(1)
    X = glm(hb, method="ols").X
    hb.hbo = hb.hbo + 3 * X[:, :1]
    R = contrast(glm(hb), [1, -1, 0, 0])
    assert R["Pval"].iloc[0] < 0.01
    assert R["Beta"].iloc[0] == pytest.approx(3, abs=1.0)


def test_ar_irls_refuses_constant_or_tiny_input():
    X = np.column_stack((np.ones(100), np.linspace(-1, 1, 100)))
    assert np.isnan(ar_irls(X, np.full(100, 2.0), fs=5).dfe)
    assert np.isnan(ar_irls(X[:5], np.arange(5.0), fs=5).dfe)


def test_contrast_formulas():
    """testContrast: hyphenated names, coefficients, unknown names rejected."""
    names = ["0-Back", "2-Back", "intercept"]
    assert parse_contrast("2-Back - 0-Back", names) == pytest.approx([-1, 1, 0])
    assert parse_contrast("0.5*2-Back + 0.5*0-Back", names) == pytest.approx([0.5, 0.5, 0])
    assert parse_contrast("2-Back", names) == pytest.approx([0, 1, 0])
    with pytest.raises(ValueError, match="not a regressor"):
        parse_contrast("NotACondition", names)


def test_contrast_table_and_fdr_per_chromophore():
    G = glm(glm_recording())
    R = contrast(G, "2-Back - 0-Back")
    assert list(R.columns) == ["Channel", "Source", "Detector", "Label", "Chromophore", "Beta", "SE", "Tstat",
                               "Pval", "Qval"]
    assert len(R) == 10 and set(R["Chromophore"]) == {"HbO", "HbR"}
    hbo = R[R["Chromophore"] == "HbO"]
    assert hbo["Qval"].to_numpy() == pytest.approx(fdr_bh(hbo["Pval"].to_numpy()), nan_ok=True)
    assert R["Beta"].iloc[1] == pytest.approx(G.beta[1, 1, 0] - G.beta[0, 1, 0])
    with pytest.raises(ValueError, match="Weight vector"):
        contrast(G, [1, -1])


def test_fdr_matches_bh_and_keeps_nan():
    """testFDR."""
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    expected = np.minimum(1, np.minimum.accumulate((p * p.size / np.arange(1, 11))[::-1])[::-1])
    q = fdr_bh(p)
    assert q == pytest.approx(expected, abs=1e-12)
    assert np.all(np.diff(q) >= -1e-12)
    assert np.isnan(fdr_bh([0.01, np.nan, 0.02])[1])
    assert np.all(np.isnan(fdr_bh([np.nan, np.nan])))


def test_default_contrasts():
    """testDefaultContrasts."""
    assert "2-Back - 0-Back" in default_contrasts(["0-Back", "1-Back", "2-Back"])
    assert "1-Back - 0-Back" in default_contrasts(["0-Back", "1-Back", "2-Back"])
    assert default_contrasts(["congruent", "incongruent"]) == ["incongruent - congruent"]
    assert default_contrasts(["Go", "NoGo"]) == ["NoGo - Go"]
    assert default_contrasts(["Alpha", "Beta"]) == ["Alpha", "Beta"]


def test_block_average_recovers_shape():
    """testBlockAvg: peak amplitude and latency of a planted response."""
    fs, T = 5.0, 2010
    time = np.arange(T) / fs
    on = np.arange(20.0, 371.0, 50.0)
    resp = np.zeros(T)
    for o in on:
        m = (time >= o) & (time < o + 40)
        resp[m] = 3 * np.exp(-((time[m] - o - 8) ** 2) / 20)
    hb = Hb("t", "t", "1", time, fs, resp[:, None], -0.4 * resp[:, None], analysis_layout(1),
            Conditions(["block"], [on], [np.full(on.size, 20.0)]), np.array(["ok"]))
    c = block_average(hb, pre=10, post=10).conditions[0]
    assert c.hbo[:, 0].max() == pytest.approx(3, abs=0.15)
    assert c.time[np.argmax(c.hbo[:, 0])] == pytest.approx(8, abs=0.5)
    assert c.n_trials[0] == on.size


def test_block_average_drops_epochs_off_the_end_and_rejects_by_amplitude():
    hb = glm_recording()
    with pytest.warns(UserWarning):
        ba = block_average(hb, pre=30, post=60)
    assert all(c.n_trials.max() < len(o) for c, o in zip(ba.conditions, hb.conditions.onsets))
    with pytest.warns(UserWarning, match="rejected"):
        c = block_average(hb, reject=4).conditions[0]
    assert c.n_trials[2] < c.n_trials[0], "the motion spike's epoch is rejected on channel 2 only"
