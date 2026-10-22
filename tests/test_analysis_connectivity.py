"""Connectivity: the NaN-safe Fisher clamp, whitening, channel selection."""
from __future__ import annotations

import numpy as np
import pytest

from synthetic import analysis_layout, ar1_noise, connectivity_recording
from wnirs.analysis import connectivity, fisher_z
from wnirs.types import Conditions, Hb


def test_excluded_channels_stay_missing():
    """testConnNaNKeepsMissing: a dropped channel must not become atanh(1 - eps) ~ 18.7."""
    rng = np.random.default_rng(0)
    hbo = rng.standard_normal((600, 6))
    hbo[:, [1, 4]] = np.nan
    hb = Hb("T1", "Rest", "1", np.arange(600) / 7, 7.0, hbo, -0.5 * hbo, analysis_layout(6), Conditions(),
            np.full(6, "ok"))
    C = connectivity(hb, whiten_ar=False)
    assert np.isnan(C.Z[1]).all() and np.isnan(C.Z[:, 4]).all()
    assert not np.any(np.isnan(C.R) & np.isfinite(C.Z))
    assert np.std(C.Z[np.isfinite(C.Z)]) > 0


def test_fisher_z_clamps_only_finite_entries():
    R = np.array([[np.nan, 1.0], [-1.0, 0.5]])
    Z = fisher_z(R)
    assert np.isnan(Z[0, 0])
    assert np.isfinite(Z[0, 1]) and Z[0, 1] == -Z[1, 0] > 18
    assert Z[1, 1] == pytest.approx(np.arctanh(0.5))


def test_shorts_are_left_out_and_diagonal_blanked():
    C = connectivity(connectivity_recording())
    assert C.R.shape == (8, 8)
    assert np.all(np.isnan(np.diag(C.R)))
    assert list(C.labels[:2]) == ["S1-D1", "S1-D2"]
    assert connectivity(connectivity_recording(), show_short=True).labels[-1] == "S3-D9 (sc)"


def test_whitening_removes_drift_driven_correlation():
    """Two channels sharing only slow autocorrelated drift look coupled until whitened."""
    rng = np.random.default_rng(1)
    T, fs = 3000, 5.0
    drift = np.cumsum(rng.standard_normal(T)) * 0.05
    hbo = drift[:, None] + 0.2 * ar1_noise(rng, T, 2, 0.5)
    hb = Hb("s", "Rest", "1", np.arange(T) / fs, fs, hbo, -hbo, analysis_layout(2), Conditions(), np.full(2, "ok"))
    raw = connectivity(hb, whiten_ar=False).R[0, 1]
    white = connectivity(hb).R[0, 1]
    assert raw > 0.8
    assert abs(white) < raw / 2


def test_window_must_leave_enough_samples():
    with pytest.raises(ValueError, match="leaves only"):
        connectivity(connectivity_recording(), window=(0, 2))
