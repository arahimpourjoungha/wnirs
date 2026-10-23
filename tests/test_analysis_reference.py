"""wnirs.analysis against reference outputs computed independently on the same
synthetic inputs.

Deterministic outputs are compared to rounding error. Mixed models are
compared to optimiser tolerance, since the reference and this implementation
reach the same REML optimum by different routes. Outputs that depend on the
reference's random stream (split-half draws, permutation nulls) are compared
statistically: the Python value must sit within a few Monte Carlo standard
errors of the reference value.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic import series_sessions
from wnirs.analysis import (SessionMeasure, arfit, block_average, change_point, channel_behaviour, connectivity,
                            contrast, dense_sampling, ezddm, fdr_bh, glm, growth_curve, gtheory, hrf, icc21,
                            iv_metrics, reliability, robust_fit, state_trait)
from wnirs.analysis._stats import spearman, t_pvalue
from wnirs.types import Conditions, Hb

FIXTURES = Path(__file__).parent / "fixtures" / "reference"
EXACT = dict(rtol=1e-9, atol=1e-12, equal_nan=True)


def load(group: str) -> dict:
    with np.load(FIXTURES / f"analysis_{group}.npz") as f:
        return {k: f[k] for k in f.files}


def rows(padded: np.ndarray) -> list[np.ndarray]:
    """Undo the NaN padding of a ragged list."""
    return [r[np.isfinite(r)] for r in padded]


def sessions(d: dict, prefix: str) -> list[SessionMeasure]:
    M = d[f"{prefix}_measure"]
    return [SessionMeasure(str(d[f"{prefix}_subject"][i]), str(d[f"{prefix}_session"][i]), M[..., i],
                           str(d[f"{prefix}_task"][i])) for i in range(M.shape[-1])]


def recording(d: dict, prefix: str = "") -> Hb:
    layout = pd.DataFrame({c: d[f"{prefix}layout_{c}"].astype(int) for c in ("S", "D", "Type")})
    names = d.get(f"{prefix}cond_names", np.array([], str))
    cond = (Conditions(list(map(str, names)), rows(d["cond_onsets"]), rows(d["cond_durations"]))
            if names.size else Conditions())
    return Hb("sim", "sim", "1", d[f"{prefix}time"], float(d[f"{prefix}fs"]), d[f"{prefix}hbo"],
              d[f"{prefix}hbr"], layout, cond, np.full(len(layout), "ok"))


@pytest.fixture(scope="module")
def stats_fx():
    return load("stats")


@pytest.fixture(scope="module")
def glm_fx():
    return load("glm")


@pytest.fixture(scope="module")
def rel_fx():
    return load("reliability")


@pytest.fixture(scope="module")
def ses_fx():
    return load("sessions")


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["canonical", "gamma", "derivative"])
def test_hrf(stats_fx, kind):
    for i, fs in enumerate(stats_fx["hrf_fs"], start=1):
        np.testing.assert_allclose(hrf(fs, kind)[0], stats_fx[f"stats__hrf_{kind}_{i}"], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("x,crit,a_key,order_key", [("ar_x", "bic", "ar_a", "ar_order"),
                                                     ("ar_x", "aic", "ar_a_aic", "ar_order_aic"),
                                                     ("ar_short", "bic", "ar_short_a", "ar_short_order")])
def test_arfit(stats_fx, x, crit, a_key, order_key):
    a, order = arfit(stats_fx[x], 20, crit)
    np.testing.assert_allclose(a, np.atleast_1d(stats_fx[f"stats__{a_key}"]), **EXACT)
    assert order == stats_fx[f"stats__{order_key}"]


def test_robust_fit(stats_fx):
    fit = robust_fit(stats_fx["rob_X"], stats_fx["rob_y"])
    np.testing.assert_allclose(fit.beta, stats_fx["stats__rob_beta"], **EXACT)
    np.testing.assert_allclose(fit.covb, stats_fx["stats__rob_covb"], **EXACT)
    np.testing.assert_allclose(fit.w, stats_fx["stats__rob_w"], rtol=1e-9, atol=1e-12)
    assert fit.dfe == stats_fx["stats__rob_dfe"]
    assert fit.sigma2 == pytest.approx(float(stats_fx["stats__rob_sigma2"]), rel=1e-12)
    assert fit.r2 == pytest.approx(float(stats_fx["stats__rob_r2"]), rel=1e-12)


def test_fdr(stats_fx):
    np.testing.assert_allclose(fdr_bh(stats_fx["fdr_p"]), stats_fx["stats__fdr_q"], **EXACT)


def test_spearman_p_values(stats_fx):
    """Exact (n < 10), AS 89 (untied) and Student (tied) branches."""
    for i, (x, y) in enumerate(zip(rows(stats_fx["sp_x"]), rows(stats_fx["sp_y"]))):
        rho, p = spearman(x, y)
        assert rho == pytest.approx(stats_fx["stats__sp_rho"][i], abs=1e-12)
        assert p == pytest.approx(stats_fx["stats__sp_p"][i], rel=1e-12)


def test_ezddm(stats_fx):
    for i, (rt, c) in enumerate(zip(rows(stats_fx["ez_rt"]), rows(stats_fx["ez_correct"]))):
        P = ezddm(rt, c.astype(bool))
        got = [P.v, P.a, P.ter, P.pc, P.mrt, P.vrt, P.n, P.n_correct, P.edge_corrected]
        want = [stats_fx[f"stats__ez_{k}"][i] for k in ("v", "a", "Ter", "Pc", "MRT", "VRT", "n", "nCorrect",
                                                          "edgeCorrected")]
        np.testing.assert_allclose(np.array(got, float), np.array(want, float), **EXACT)


# ---------------------------------------------------------------------------
# GLM, contrasts, block average
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["arirls", "ar", "ols"])
def test_glm(glm_fx, method):
    G = glm(recording(glm_fx), method=method)
    pre = f"glm__{method}_"
    for ours, theirs in [(G.beta, "beta"), (G.se, "se"), (G.tstat, "tstat"), (G.cov, "cov"), (G.r2, "R2")]:
        np.testing.assert_allclose(ours, glm_fx[pre + theirs], rtol=1e-9, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(G.pval, glm_fx[pre + "pval"], rtol=1e-7, atol=1e-15, equal_nan=True)
    if method == "arirls":
        np.testing.assert_allclose(G.X, glm_fx["glm__X"], rtol=1e-12, atol=1e-12)
        assert G.names == list(glm_fx["glm__names"])
        np.testing.assert_array_equal(G.ar_order, glm_fx["glm__arOrder"])
        np.testing.assert_allclose(G.weights[..., 0], glm_fx["glm__weights_hbo"], rtol=1e-9, atol=1e-12,
                                   equal_nan=True)


def test_design_with_derivatives(glm_fx):
    G = glm(recording(glm_fx), method="ols", derivative=True, drift=2, hrf="gamma")
    np.testing.assert_allclose(G.X, glm_fx["glm__X_deriv"], rtol=1e-12, atol=1e-12)
    assert G.names == list(glm_fx["glm__names_deriv"])


@pytest.mark.parametrize("tag,spec", [("diff", "2-Back - 0-Back"), ("mean", "0.5*2-Back + 0.5*0-Back")])
def test_contrast(glm_fx, tag, spec):
    """Estimates match exactly; p values use each channel's own dfe.

    The reference keeps one scalar dfe (the last channel fitted) for every
    channel, so its p values agree with ours exactly where the AR orders agree with
    that channel's, and otherwise differ only through the degrees of freedom.
    """
    G = glm(recording(glm_fx))
    R = contrast(G, spec)
    pre = f"glm__con_{tag}_"
    for col, key in [("Beta", "beta"), ("SE", "se"), ("Tstat", "t")]:
        np.testing.assert_allclose(R[col], glm_fx[pre + key], rtol=1e-9, atol=1e-12, equal_nan=True)
    ref_dfe = float(glm_fx["glm__arirls_dfe"])
    same_dfe = G.dfe.T.ravel() == ref_dfe
    np.testing.assert_allclose(R["Pval"][same_dfe], glm_fx[pre + "p"][same_dfe], rtol=1e-8)
    np.testing.assert_allclose(t_pvalue(R["Tstat"], ref_dfe), glm_fx[pre + "p"], rtol=1e-8, equal_nan=True)


def test_block_average(glm_fx):
    with pytest.warns(UserWarning, match="rejected 1 channel-epochs"):
        ba = block_average(recording(glm_fx), pre=5, post=10, reject=4)
    for k, c in enumerate(ba.conditions, start=1):
        np.testing.assert_allclose(c.time, glm_fx[f"glm__ba{k}_time"], **EXACT)
        np.testing.assert_allclose(c.hbo, glm_fx[f"glm__ba{k}_hbo"], **EXACT)
        np.testing.assert_allclose(c.hbr, glm_fx[f"glm__ba{k}_hbr"], **EXACT)
        np.testing.assert_allclose(c.sem_hbo, glm_fx[f"glm__ba{k}_semHbO"], **EXACT)
        np.testing.assert_array_equal(c.n_trials, glm_fx[f"glm__ba{k}_nTrials"])


# ---------------------------------------------------------------------------
# connectivity
# ---------------------------------------------------------------------------
def test_connectivity():
    d = load("connectivity")
    hb = recording(d, "c_")
    C = connectivity(hb)
    np.testing.assert_allclose(C.R, d["connectivity__R_white"], **EXACT)
    np.testing.assert_allclose(C.Z, d["connectivity__Z_white"], **EXACT)
    assert C.mean_r == pytest.approx(float(d["connectivity__meanR_white"]), rel=1e-12)
    C = connectivity(hb, whiten_ar=False)
    np.testing.assert_allclose(C.R, d["connectivity__R_raw"], **EXACT)
    np.testing.assert_allclose(C.Z, d["connectivity__Z_raw"], **EXACT)
    C = connectivity(hb, whiten_ar=False, kind="spearman", chromophore="hbr")
    np.testing.assert_allclose(C.R, d["connectivity__R_spearman_hbr"], **EXACT)
    C = connectivity(hb, whiten_ar=False, show_short=True, window=(20, 200))
    np.testing.assert_allclose(C.R, d["connectivity__R_short_window"], **EXACT)


# ---------------------------------------------------------------------------
# reliability
# ---------------------------------------------------------------------------
def test_fingerprint_deterministic_core(rel_fx):
    F = reliability(sessions(rel_fx, "fp"), analyses=["fingerprint"], common_only=0.8, rng=0).fingerprint
    for ours, key in [(F.similarity, "similarity"), (F.within, "within"), (F.between, "between"),
                      (F.mean_within, "meanWithin"), (F.mean_between, "meanBetween"),
                      (F.separation, "separation"), (F.id_accuracy, "idAccuracy"), (F.chance, "chance"),
                      (F.n_compared, "nCompared")]:
        np.testing.assert_allclose(ours, rel_fx[f"reliability__fp_{key}"], **EXACT)
    # The permutation p is random, but here the separation is far outside the null in both.
    assert F.pval == pytest.approx(float(rel_fx["reliability__fp_pval"]), abs=2e-3)


def _assert_curve_close(curve: pd.DataFrame, d: dict, prefix: str, n_iter: int) -> None:
    se = np.hypot(curve["sdR"], d[prefix + "sdR"]) / np.sqrt(n_iter)
    assert np.all(np.abs(curve["meanR"] - d[prefix + "meanR"]) < 4 * se)
    np.testing.assert_allclose(curve["sdR"], d[prefix + "sdR"], rtol=0.15)
    for col in ("nSessions", "nChannels", "nReference", "meanOverlap", "nIter"):
        np.testing.assert_array_equal(curve[col], d[prefix + col])
    assert list(curve["Subject"]) == list(d[prefix + "Subject"])


def test_reliability_curve_statistically(rel_fx):
    R = reliability(sessions(rel_fx, "fp"), analyses=["curve"], n_iter=2000, common_only=0.8, rng=1)
    _assert_curve_close(R.curve, rel_fx, "reliability__curve_", 2000)


def test_reliability_curve_ref70_spearman_null(rel_fx):
    R = reliability(sessions(rel_fx, "fp"), analyses=["curve"], n_iter=2000, common_only=0.8, ref_frac=0.7,
                    null_between=True, kind="spearman", rng=2)
    c, pre = R.curve, "reliability__curve70_"
    _assert_curve_close(c, rel_fx, pre, 2000)
    assert np.all(np.abs(c["nullMean"] - rel_fx[pre + "nullMean"]) < 0.01)
    np.testing.assert_allclose(c["pPerm"], rel_fx[pre + "pPerm"], atol=2e-3)


def test_icc21_exact(rel_fx):
    """icc21 against its reference values on four fixed matrices."""
    got = [icc21(rel_fx[f"icc21_M_{i}"]) for i in range(4)]
    np.testing.assert_allclose(got, rel_fx["reliability__icc21"], **EXACT)


def test_icc_curve_statistically(rel_fx):
    R = reliability(sessions(rel_fx, "icc"), analyses=["icc"], rng=3)
    row = R.icc.iloc[0]
    sd = float(rel_fx["reliability__icc_sdICC"])
    assert row["nSessions"] == rel_fx["reliability__icc_nSessions"]
    assert abs(row["ICC"] - float(rel_fx["reliability__icc_ICC"])) < 4 * sd * np.sqrt(2 / 20)


# ---------------------------------------------------------------------------
# session-level analyses
# ---------------------------------------------------------------------------
def _dense_stats(d: dict) -> list[dict]:
    B, S, dose = d["ds_beta"], d["ds_se"], d["ds_dose"]
    n_ch = B.shape[0]
    return [dict(session=s + 1, dose=dose[s], table=pd.DataFrame({
        "Channel": np.arange(1, n_ch + 1), "Source": 1, "Detector": np.arange(1, n_ch + 1),
        "Label": [f"ch{i}" for i in range(1, n_ch + 1)], "Chromophore": "HbO", "Beta": B[:, s], "SE": S[:, s]}))
        for s in range(B.shape[1])]


@pytest.mark.parametrize("i,spec", list(enumerate([
    dict(model="mean"), dict(model="mean", covariates=["sessionNumber"]), dict(model="mean", weight="inverse"),
    dict(model="trend"), dict(model="predictor", predictor="dose")], start=1)))
def test_dense_sampling(ses_fx, i, spec):
    D = dense_sampling(_dense_stats(ses_fx), **spec)
    for col in ("Estimate", "SE", "Tstat", "Pval", "Qval", "N"):
        np.testing.assert_allclose(D[col], ses_fx[f"sessions__ds{i}_{col}"], **EXACT)
    rel = D.attrs["reliability"]
    assert rel["pattern"] == pytest.approx(float(ses_fx[f"sessions__ds{i}_pattern"]), rel=1e-12)
    np.testing.assert_allclose(rel["per_channel"], ses_fx[f"sessions__ds{i}_perChannel"], **EXACT)


def test_change_point(ses_fx):
    C = change_point(series_sessions(ses_fx["cp_M"]), rng=4)
    assert list(C["Model"]) == list(ses_fx["sessions__cp_Model"])
    for col in ("Element", "nSessions", "Slope", "StepAt", "StepSize"):
        np.testing.assert_allclose(C[col], ses_fx[f"sessions__cp_{col}"], **EXACT)
    # Permutation p values from 2000 shuffles each: binomial SE at most ~0.011.
    for col in ("pTrend", "pStep"):
        p, q = C[col].to_numpy(), ses_fx[f"sessions__cp_{col}"]
        se = np.sqrt(np.maximum(p * (1 - p), 1e-4) / 2000) * np.sqrt(2)
        assert np.all(np.abs(p - q) < 4 * se + 1e-3)


def test_iv_metrics(ses_fx):
    T = iv_metrics(series_sessions(ses_fx["iv_M"]))
    for col in T.columns[2:]:
        np.testing.assert_allclose(T[col], ses_fx[f"sessions__iv_{col}"], **EXACT)


def test_state_trait(ses_fx):
    T = pd.DataFrame({"Subject": ses_fx["st_subject"], "SessionNo": ses_fx["st_session"],
                      "Measure": ses_fx["st_measure"], "Value": ses_fx["st_value"],
                      "Condition": ses_fx["st_condition"]})
    V, M = state_trait(T, target={})
    assert list(V["Subject"]) == list(ses_fx["sessions__stv_Subject"])
    assert list(V["Conditions"]) == list(ses_fx["sessions__stv_Conditions"])
    for col in V.columns[2:-1]:
        np.testing.assert_allclose(V[col], ses_fx[f"sessions__stv_{col}"], **EXACT)
    assert list(M["Measure"]) == list(ses_fx["sessions__stm_Measure"])
    for col in M.columns[1:]:
        tol = dict(rtol=1e-5) if col == "ICC_between" else EXACT        # ML optimum
        np.testing.assert_allclose(M[col], ses_fx[f"sessions__stm_{col}"], **tol)


def test_growth_curve_random_slope(ses_fx):
    G = growth_curve(sessions(ses_fx, "gc"))
    assert G.adequate and G.fixed["random_slope_fitted"]
    for col in ("Intercept", "Slope", "SlopeSE", "SlopeP", "SlopeQ", "SlopeVar", "ICC"):
        np.testing.assert_allclose(G.table[col], ses_fx[f"sessions__gc_{col}"], rtol=1e-4, atol=1e-7)
    np.testing.assert_array_equal(G.table["nObs"], ses_fx["sessions__gc_nObs"])
    np.testing.assert_allclose(G.per_subject["Slope"], ses_fx["sessions__gcps_Slope"], **EXACT)


def test_growth_curve_shape_comparison(ses_fx):
    G = growth_curve(sessions(ses_fx, "gc5"), shape="compare")
    assert list(G.table["Shape"]) == list(ses_fx["sessions__gc5_Shape"])
    for col in ("Intercept", "Slope", "SlopeSE", "SlopeP", "ICC"):
        np.testing.assert_allclose(G.table[col], ses_fx[f"sessions__gc5_{col}"], rtol=1e-4, atol=1e-7,
                                   equal_nan=True)


def test_channel_behaviour(ses_fx):
    B = ses_fx["cr_beta"]
    tables = [pd.DataFrame({"Channel": np.tile(np.arange(1, 7), 2), "Source": 1,
                            "Detector": np.tile(np.arange(1, 7), 2),
                            "Label": [f"S1-D{i}" for i in range(1, 7)] * 2,
                            "Chromophore": ["HbO"] * 6 + ["HbR"] * 6, "Beta": B[i].T.ravel()}) for i in range(len(B))]
    R = channel_behaviour(tables, ses_fx["cr_y"])
    for col, key in [("Rho", "rho"), ("Pval", "p"), ("Qval", "q"), ("N", "N")]:
        np.testing.assert_allclose(R[col], ses_fx[f"sessions__cr_{key}"], **EXACT)
    R = channel_behaviour(tables, ses_fx["cr_y"], kind="pearson", covariates=ses_fx["cr_cov"])
    np.testing.assert_allclose(R["Rho"], ses_fx["sessions__crp_rho"], **EXACT)
    np.testing.assert_allclose(R["Pval"], ses_fx["sessions__crp_p"], **EXACT)


@pytest.mark.parametrize("tag", ["gt", "gt1"])
def test_gtheory(ses_fx, tag):
    g = gtheory(sessions(ses_fx, tag))[0]
    for comp, v in g.var.items():
        assert v == pytest.approx(float(ses_fx[f"sessions__{tag}_{comp}"]), rel=1e-4, abs=1e-8)
    for ours, key in [(g.Rc, "Rc"), (g.R1F, "R1F"), (g.Rc_single, "RcSingle"), (g.R1F_single, "R1FSingle")]:
        # same tolerance as the components: the REML surface is flat along some
        # directions, so optimisers agree on the likelihood long before the ratios
        np.testing.assert_allclose(ours, ses_fx[f"sessions__{tag}_{key}"], rtol=1e-4, equal_nan=True)


def test_mixed_survives_near_singular_fixed_effects():
    """A stepped-wedge fit (period dummies, random intercept and slope) whose
    fixed-effect cross-product loses positive definiteness to cancellation;
    the plain Cholesky used to raise LinAlgError on it."""
    from scipy import sparse
    from wnirs.analysis.mixed import RandomTerm, fit_mixed
    d = np.load(FIXTURES / "mixed_near_singular.npz")
    term = RandomTerm("subject", sparse.csr_matrix(d["Z"]), int(d["n_effects"]))
    fit = fit_mixed(d["y"], d["X"], [term])
    assert np.all(np.isfinite(fit.beta)) and np.all(np.isfinite(fit.se))
    assert fit.sigma2 > 0
