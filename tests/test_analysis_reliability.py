"""Reliability curve, fingerprinting and ICC."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthetic import correlation_sessions, gtheory_sessions
from wnirs.analysis import SessionMeasure, icc21, reliability
from wnirs.analysis.reliability import rows_finite


@pytest.fixture(scope="module")
def six_people():
    return correlation_sessions(3, n_subjects=6, n_sessions=10, n_channels=17, noise=1.2)


def test_curve_rises_and_people_are_identifiable(six_people):
    R = reliability(six_people, n_iter=60, n_perm=500, rng=0)
    g = R.curve.groupby("nSessions")["meanR"].mean()
    assert len(g) >= 3
    assert g.iloc[-1] > g.iloc[0]
    assert np.all(np.diff(g.to_numpy()) > -0.05)
    F = R.fingerprint
    assert F.mean_within > F.mean_between
    assert F.id_accuracy > F.chance
    assert F.pval < 0.05
    icc = R.icc.sort_values("nSessions")
    assert icc["ICC"].iloc[0] > 0, "a negative ICC means targets and raters were transposed"
    assert icc["ICC"].iloc[-1] > icc["ICC"].iloc[0]


def test_no_individual_pattern_no_fingerprint(six_people):
    rng = np.random.default_rng(5)
    shuffled = [SessionMeasure(s.subject, s.session, np.tanh(rng.standard_normal(s.measure.shape)), s.task)
                for s in six_people]
    F = reliability(shuffled, analyses=["fingerprint"], n_perm=500, rng=1).fingerprint
    assert F.pval > 0.01


def test_dead_channels_are_dropped_not_scored():
    """testFisherNaN: missing channels must not turn into a reproducible constant."""
    sessions = correlation_sessions(21, n_subjects=2, n_sessions=8, n_channels=40, dropout=0.1)
    R = reliability(sessions, analyses=["curve"], n_iter=40, common_only=0.8, rng=0)
    n_pairs = 40 * 39 // 2
    assert 3 <= R.curve["nChannels"].iloc[0] < n_pairs


def test_every_field_present_whatever_ran():
    """testReliabilityShape."""
    rng = np.random.default_rng(11)
    S = [SessionMeasure(f"S{s}", str(e), rng.standard_normal(12), "rest") for s in range(3) for e in range(6)]
    for analyses in (["curve"], ["icc"], ["curve", "icc"], ["curve", "fingerprint", "icc"]):
        R = reliability(S, analyses=analyses, n_iter=20, n_perm=100, rng=0)
        assert isinstance(R.curve, pd.DataFrame) and isinstance(R.icc, pd.DataFrame)
        assert np.isscalar(R.fingerprint.mean_within)
        assert list(R.curve.columns)[:4] == ["Subject", "Task", "nSessions", "meanR"]


def test_fingerprinting_one_person_is_refused_and_chance_counts_the_pool():
    """testFingerprintGuard: 2 subjects x 6 sessions -> chance is 5/11, not 100%."""
    rng = np.random.default_rng(11)
    one = [SessionMeasure("S01", str(k), rng.standard_normal(20), "NBack") for k in range(6)]
    with pytest.raises(ValueError, match="at least 2 subjects"):
        reliability(one, analyses=["fingerprint"])
    two = one + [SessionMeasure("S02", str(k), rng.standard_normal(20), "NBack") for k in range(6)]
    F = reliability(two, analyses=["fingerprint"], n_perm=100, rng=0).fingerprint
    assert np.isfinite(F.mean_between)
    assert F.chance == pytest.approx(5 / 11)


def test_reference_fraction_reports_overlap():
    sessions = correlation_sessions(4, n_subjects=2, n_sessions=10, n_channels=10)
    c = reliability(sessions, analyses=["curve"], n_iter=50, ref_frac=0.7, rng=0).curve
    assert set(c["nReference"]) == {7}
    np.testing.assert_array_equal(c["meanOverlap"], np.maximum(0, c["nSessions"] - 3))


def test_between_person_null_sits_below_own_sessions():
    sessions = correlation_sessions(6, n_subjects=3, n_sessions=8, n_channels=12, noise=0.8)
    c = reliability(sessions, analyses=["curve"], n_iter=200, null_between=True, rng=0).curve
    assert np.all(c["nullMean"] < c["meanR"])
    assert np.all(c["pPerm"] < 0.05)


def test_too_few_sessions():
    with pytest.raises(ValueError, match="several sessions"):
        reliability(correlation_sessions(0, 1, 3, 5))


def test_icc21_edge_cases():
    assert np.isnan(icc21(np.ones((5, 2))))
    assert np.isnan(icc21(np.array([[1.0, 2.0]])))
    M = np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 2.9], [np.nan, 1.0]])
    assert 0.9 < icc21(M) <= 1


def test_rows_finite_fractions():
    V = np.array([[1, 1, 1, 1], [1, np.nan, 1, 1], [np.nan, np.nan, 1, 1.0]])
    np.testing.assert_array_equal(rows_finite(V, True), [True, False, False])
    np.testing.assert_array_equal(rows_finite(V, False), [True, True, True])
    np.testing.assert_array_equal(rows_finite(V, 0.75), [True, True, False])


def test_icc_from_vector_maps():
    S = gtheory_sessions(8, 6, 30, 1.0, 0.1, 0.5, 1.0, 0.3, seed=2)
    icc = reliability(S, analyses=["icc"], rng=0).icc
    assert list(icc["nSessions"]) == [1, 2, 3]
    assert icc["ICC"].is_monotonic_increasing
