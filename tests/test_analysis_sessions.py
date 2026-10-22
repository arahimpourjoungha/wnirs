"""Dense-sampling analyses across sessions: behaviour on planted truths, plus
EZ-diffusion, rmcorr and the mixed-model engine against independent references."""
from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from synthetic import behaviour_trials, growth_sessions, gtheory_sessions, series_sessions
from wnirs.analysis import (SessionMeasure, change_point, dense_sampling, ez_parameters, ezddm, fit_mixed,
                            growth_curve, gtheory, iv_metrics, rmcorr, state_trait)
from wnirs.analysis._stats import spearman
from wnirs.analysis.mixed import random_intercept, random_slope


def _table(beta, se=0.2):
    n = beta.size
    return pd.DataFrame({"Channel": np.arange(1, n + 1), "Source": 1, "Detector": np.arange(1, n + 1),
                         "Label": [f"ch{i}" for i in range(1, n + 1)], "Chromophore": "HbO", "Beta": beta,
                         "SE": np.full(n, se)})


def test_dense_sampling_models_find_planted_channels():
    rng = np.random.default_rng(5)
    n_s, n_ch, real = 12, 20, 5
    truth = np.r_[np.full(real, 0.6), np.zeros(n_ch - real)]
    R = dense_sampling([dict(session=s, table=_table(truth + 0.25 * rng.standard_normal(n_ch)))
                        for s in range(1, n_s + 1)])
    assert np.all(R["Qval"][:real] < 0.05) and np.sum(R["Qval"][real:] < 0.05) <= 3
    assert np.isfinite(R.attrs["reliability"]["pattern"])

    trend = []
    for s in range(1, n_s + 1):
        b = 0.25 * rng.standard_normal(n_ch)
        b[:real] += 0.12 * s
        trend.append(dict(session=s, table=_table(b)))
    R = dense_sampling(trend, model="trend")
    assert np.all(R["Qval"][:real] < 0.05) and np.sum(R["Qval"][real:] < 0.05) <= 2

    dose = 10 + 3 * rng.standard_normal(n_s)
    pred = []
    for s in range(n_s):
        b = 0.25 * rng.standard_normal(n_ch)
        b[:real] += 0.15 * dose[s]
        pred.append(dict(session=s + 1, dose=dose[s], table=_table(b)))
    R = dense_sampling(pred, model="predictor", predictor="dose")
    assert np.all(R["Qval"][:real] < 0.05) and np.sum(R["Qval"][real:] < 0.05) <= 2


def test_dense_sampling_null_rate():
    rng = np.random.default_rng(6)
    p = np.concatenate([dense_sampling([dict(session=s, table=_table(0.25 * rng.standard_normal(20)))
                                        for s in range(1, 13)])["Pval"] for _ in range(20)])
    assert np.mean(p < 0.05) < 0.15


def test_one_excluded_channel_does_not_void_pattern_reliability():
    """testPatternReliability."""
    rng = np.random.default_rng(7)
    base = rng.standard_normal(12)
    stats = []
    for k in range(1, 6):
        b = base + 0.15 * rng.standard_normal(12)
        b[2] = np.nan
        stats.append(dict(session=str(k), table=_table(b)))
    assert dense_sampling(stats).attrs["reliability"]["pattern"] > 0.5


def test_dense_sampling_guards():
    with pytest.raises(ValueError, match="at least 3"):
        dense_sampling([dict(session=1, table=_table(np.zeros(3)))] * 2)
    with pytest.raises(ValueError, match="predictor"):
        dense_sampling([dict(session=i, table=_table(np.zeros(3))) for i in range(4)], model="predictor",
                       predictor="dose")
    with pytest.warns(UserWarning, match="not a key"):
        dense_sampling([dict(session=i, table=_table(np.arange(3.0) + i)) for i in range(4)], covariates=["sleep"])


def test_change_point_separates_noise_trend_and_step():
    """testChangePoint."""
    rng = np.random.default_rng(7)
    n, n_el = 24, 20
    t = np.arange(1, n + 1)
    run = lambda M: change_point(series_sessions(M), n_perm=400, rng=1)  # noqa: E731
    flat = run(rng.standard_normal((n_el, n)))
    assert not np.any(flat["qTrend"] < 0.05) and not np.any(flat["qStep"] < 0.05)
    assert np.mean(flat["Model"] == "constant") > 0.6
    trend = run(0.15 * t + rng.standard_normal((n_el, n)))
    assert np.mean(trend["Model"] == "trend") > 0.5
    assert np.sum(trend["qTrend"] < 0.05) > n_el / 2
    assert trend["Slope"].median() > 0.05
    step = run(3.0 * (t > 12) + rng.standard_normal((n_el, n)))
    assert np.mean(step["Model"] == "step") > 0.5
    assert abs(step["StepAt"].median() - 12) <= 1


def test_change_point_refuses_unordered_sessions():
    M = np.random.default_rng(0).standard_normal((3, 10))
    bad = [SessionMeasure("s", f"sess_{chr(97 + i)}", M[:, i], "x") for i in range(10)]
    with pytest.raises(ValueError, match="time order"):
        change_point(bad, n_perm=10)
    ordered = [SessionMeasure("s", f"sess_{chr(97 + i)}", M[:, i], "x", order=10 - i) for i in range(10)]
    assert len(change_point(ordered, n_perm=10, rng=0)) == 3


def test_iv_metrics_white_noise_drift_and_order():
    """testIVMetrics: sqrt(2) for white noise; drift vs oscillation; order dependence."""
    one = lambda x, sub="s": iv_metrics(series_sessions(x[None, :], subject=sub)).iloc[0]  # noqa: E731
    rng = np.random.default_rng(3)
    W = one(rng.standard_normal(400))
    assert abs(W["RMSSD_iSD_ratio"] - np.sqrt(2)) < 0.12 and abs(W["AR1"]) < 0.15
    t = np.arange(1, 201)
    D = one(np.linspace(-2, 2, 200) + 0.05 * rng.standard_normal(200))
    O = one(2 * (-1.0) ** t + 0.05 * rng.standard_normal(200))
    assert D["RMSSD"] < 0.1 * O["RMSSD"]
    assert D["AR1"] > 0.8 and O["AR1"] < -0.8
    v = np.cumsum(rng.standard_normal(120))
    A, B = one(v), one(rng.permutation(v))
    assert A["iSD"] == pytest.approx(B["iSD"])
    assert A["AR1"] - B["AR1"] > 0.3


def test_gtheory_recovers_planted_components():
    """testGTheory."""
    truth = dict(person=1.0, personSession=0.5, channel=0.8, personChannel=0.3, residual=0.4)
    g = gtheory(gtheory_sessions(6, 12, 40, *truth.values(), seed=1))[0]
    for k, v in truth.items():
        assert abs(g.var[k] - v) < 0.25 * v + 0.1, k
    assert g.projection["Rc"].is_monotonic_increasing and g.projection["R1F"].is_monotonic_increasing
    g1 = gtheory(gtheory_sessions(1, 20, 40, 0, 0.5, 0.8, 0, 0.4, seed=4))[0]
    assert np.isfinite(g1.Rc_single) and np.isnan(g1.R1F_single)


def test_gtheory_dissociates_identity_from_change():
    """testGTheoryDissociation."""
    stable = gtheory(gtheory_sessions(6, 12, 40, 2.0, 0.02, 0.8, 0.3, 0.4, seed=2))[0]
    moving = gtheory(gtheory_sessions(6, 12, 40, 0.02, 2.0, 0.8, 0.02, 0.4, seed=3))[0]
    assert stable.R1F_single > 0.6 and stable.Rc_single < 0.3
    assert moving.Rc_single > 0.6 and moving.R1F_single < 0.4


def test_growth_curve_estimates_and_guards():
    """testGrowthCurve."""
    G = growth_curve(growth_sessions(20, 15, 10, 0.20, 0.15, seed=3))
    assert G.adequate and abs(G.fixed["mean_slope"] - 0.2) < 0.1
    assert abs(G.table["SlopeVar"].median() - 0.0225) < 0.02
    G0 = growth_curve(growth_sessions(20, 15, 10, 0.0, 0.0, seed=7))
    assert abs(G0.fixed["mean_slope"]) < 0.05 and G0.fixed["n_significant"] <= 2
    G2 = growth_curve(growth_sessions(2, 18, 10, 0.2, 0.15, seed=4))
    assert not G2.adequate and len(G2.per_subject) > 0
    G5 = growth_curve(growth_sessions(5, 15, 10, 0.2, 0.15, seed=5))
    assert G5.adequate and not G5.fixed["random_slope_fitted"]
    with pytest.warns(UserWarning, match="random intercept only"):
        growth_curve(growth_sessions(5, 15, 2, 0.2, 0.15, seed=5), random_slope=True)


def test_state_trait_separates_drift_from_state():
    rng = np.random.default_rng(0)
    rows = []
    for s, subj in enumerate(("a", "b", "c")):
        sess = np.r_[1:11, 13:23]                 # a gap: AR1 must skip the non-adjacent pair
        for name, val in (("drifting", s + 0.3 * sess + 0.2 * rng.standard_normal(20)),
                          ("noisy", s + 0.3 * rng.standard_normal(20))):
            rows += [dict(Subject=subj, SessionNo=k, Measure=name, Value=v) for k, v in zip(sess, val)]
    V, M = state_trait(pd.DataFrame(rows))
    by = V.set_index(["Subject", "Measure"])
    assert (by.xs("drifting", level="Measure")["PctTrend"] > 80).all()
    assert (by.xs("noisy", level="Measure")["PctState"] > 70).all()
    assert list(M["Measure"]) == ["noisy", "drifting"], "sorted by PctState, descending"
    assert M.set_index("Measure").loc["noisy", "ICC_between"] > 0.5
    with pytest.raises(ValueError, match="sessions"):
        state_trait(pd.DataFrame(rows), min_obs=50)


def test_ezddm_edges_and_closed_form():
    rt, correct = behaviour_trials(1, 120, 0.9)
    P = ezddm(rt, correct)
    assert P.n == np.sum((rt >= 0.15) & (rt <= 5)) and not P.edge_corrected
    assert P.v > 0 and P.a > 0 and 0 < P.ter < P.mrt
    assert ezddm(rt, np.ones(rt.size, bool)).edge_corrected
    assert np.isnan(ezddm(rt, np.zeros(rt.size, bool)).v)
    assert np.isnan(ezddm(rt[:5], correct[:5]).v)
    # Wagenmakers et al. (2007), worked example: Pc .802, VRT .112, MRT .723 -> v .1, a .14, Ter .3
    assert ez_parameters(0.802, 0.112, 0.723) == pytest.approx((0.0999, 0.1400, 0.3002), abs=2e-3)


def test_rmcorr_equals_the_ancova_slope_test():
    """rmcorr's t is the t of x in y ~ x + subject (Bakdash & Marusich 2017)."""
    rng = np.random.default_rng(2)
    g = np.repeat(["a", "b", "c", "d"], 12)
    offset = np.repeat(rng.normal(0, 3, 4), 12)
    x = offset + rng.standard_normal(48)
    y = 2 * offset + 0.5 * x + rng.standard_normal(48)
    rm = rmcorr(x, y, g)
    fit = smf.ols("y ~ x + C(g)", pd.DataFrame(dict(x=x, y=y, g=g))).fit()
    assert rm.df == fit.df_resid == 48 - 4 - 1
    assert rm.p == pytest.approx(fit.pvalues["x"], rel=1e-10)
    assert np.sign(rm.r) == np.sign(fit.params["x"])
    assert np.corrcoef(x, y)[0, 1] > 0.9 > rm.r, "between-person offsets inflate the pooled r"


def test_spearman_exact_p_is_the_permutation_distribution():
    rng = np.random.default_rng(4)
    x, y = rng.standard_normal(7), rng.standard_normal(7)
    rho, p = spearman(x, y)
    null = [np.corrcoef(np.argsort(np.argsort(x)), perm)[0, 1] for perm in permutations(range(7))]
    assert p == pytest.approx(min(1, 2 * min(np.mean(np.array(null) >= rho - 1e-12),
                                             np.mean(np.array(null) <= rho + 1e-12))))


def test_mixed_model_matches_statsmodels():
    rng = np.random.default_rng(8)
    g = np.repeat(np.arange(12), 10)
    t = np.tile(np.arange(10.0), 12) - 4.5
    y = 1 + rng.normal(0, 1, 12)[g] + (0.3 + rng.normal(0, 0.2, 12)[g]) * t + rng.standard_normal(120)
    X = np.column_stack((np.ones(120), t))
    ours = fit_mixed(y, X, [random_slope("g", g, t)], method="reml")
    ref = smf.mixedlm("y ~ t", pd.DataFrame(dict(y=y, t=t, g=g)), groups="g", re_formula="~t").fit(reml=True)
    np.testing.assert_allclose(ours.beta, ref.fe_params, rtol=1e-4)
    np.testing.assert_allclose(ours.psi["g"], ref.cov_re, rtol=1e-3)
    assert ours.sigma2 == pytest.approx(ref.scale, rel=1e-4)
    ml = fit_mixed(y, X[:, :1], [random_intercept("g", g)], method="ml")
    ref = smf.mixedlm("y ~ 1", pd.DataFrame(dict(y=y, g=g)), groups="g").fit(reml=False)
    assert ml.loglik >= ref.llf - 1e-6
