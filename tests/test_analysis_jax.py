"""The JAX AR-IRLS must reproduce the NumPy estimator, channel for channel."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from synthetic import glm_recording, null_recording  # noqa: E402
from wnirs.analysis import design_matrix, fit_channels  # noqa: E402
from wnirs.analysis.glm_jax import ar_irls_jax  # noqa: E402

FIELDS = ("beta", "se", "tstat", "pval", "cov", "r2", "dfe", "ar_order", "weights")


def assert_same_fits(ours, ref, rtol=1e-6):
    for name in FIELDS:
        a, b = getattr(ours, name), getattr(ref, name)
        assert a.shape == b.shape, name
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b), err_msg=name)
        np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-10, equal_nan=True, err_msg=name)


@pytest.mark.parametrize("chromophore", ["hbo", "hbr"])
def test_matches_numpy_including_excluded_and_motion_channels(chromophore):
    hb = glm_recording()
    X = design_matrix(hb.time, hb.conditions, hb.fs).X
    Y = getattr(hb, chromophore)
    assert_same_fits(ar_irls_jax(X, Y, fs=hb.fs), fit_channels(X, Y, fs=hb.fs))


def test_batched_sessions_match_per_session_fits():
    """Sessions with their own designs, vmapped together (and split into calls)."""
    hbs = [null_recording(s) for s in range(1, 6)]
    X = np.stack([design_matrix(h.time, h.conditions, h.fs).X for h in hbs])
    Y = np.stack([np.column_stack((h.hbo[:, 0], h.hbr[:, 0], h.hbo[:, 0] + 0.5 * X[i, :, 0]))
                  for i, h in enumerate(hbs)])
    Y[2, :, 1] = 3.0                                  # a flat channel is refused, as in NumPy
    fits = ar_irls_jax(X, Y, fs=hbs[0].fs, session_batch=2)
    for s in range(len(hbs)):
        ref = fit_channels(X[s], Y[s], fs=hbs[0].fs)
        assert_same_fits(type(fits)(**{k: v[s] for k, v in vars(fits).items()}), ref)
    assert np.all(np.isnan(fits.beta[2, :, 1]))


def test_fixed_order_and_no_whitening():
    hb = glm_recording()
    X = design_matrix(hb.time, hb.conditions, hb.fs).X
    for pmax in (0, 3):
        assert_same_fits(ar_irls_jax(X, hb.hbo, pmax=pmax), fit_channels(X, hb.hbo, fs=hb.fs, ar=pmax))


def test_refuses_too_short_recordings():
    with pytest.raises(ValueError, match="cannot support"):
        ar_irls_jax(np.ones((9, 2)), np.random.default_rng(0).standard_normal((9, 1)), pmax=2)
