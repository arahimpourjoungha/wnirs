"""Canonical haemodynamic response function."""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import gammaln

HRFKind = Literal["canonical", "gamma", "derivative"]


def hrf(fs: float, kind: HRFKind = "canonical", duration: float = 32.0) -> tuple[np.ndarray, np.ndarray]:
    """Haemodynamic response kernel sampled at ``fs``.

    The double-gamma shape follows SPM's canonical, with the gammas placed so
    the response peaks at 6 s and undershoots near 16-17 s, the undershoot one
    sixth of the peak. This is not numerically identical to ``spm_hrf``: SPM
    parameterises the gamma by its shape, which puts its canonical peak at 5 s,
    whereas the peak here lands on the stated 6 s. The 1 s difference is
    deliberate -- the fNIRS response is generally slower than the BOLD response
    the SPM shape was fitted to -- but a result from this toolbox should not be
    described as having used the SPM canonical HRF unmodified.

    HbR is modelled with the same shape and comes out with a negative beta,
    which is the expected sign. The kernel is normalised to unit peak, so GLM
    betas read directly in micromolar of peak response.

    Parameters
    ----------
    fs : float
        Sampling rate, Hz.
    kind : {'canonical', 'gamma', 'derivative'}
        Double gamma, single gamma without undershoot, or the temporal
        derivative of the double gamma.
    duration : float
        Kernel length in seconds.

    Returns
    -------
    h, t : numpy.ndarray
        Kernel and its time axis in seconds, both starting at 0.
    """
    n = int(np.floor(duration * fs + 1e-9)) + 1
    t = np.arange(n) / fs
    kind = kind.lower()
    if kind == "canonical":
        h = _double_gamma(t)
    elif kind == "gamma":
        h = _gamma_pdf(t, 6.0, 1.0)
    elif kind == "derivative":
        h = np.concatenate(([0.0], np.diff(_double_gamma(t)))) * fs
    else:
        raise ValueError(f"Unknown HRF {kind!r}. Use canonical, gamma, or derivative.")
    peak = np.max(np.abs(h))
    return (h / peak if peak > 0 else h), t


def _double_gamma(t: np.ndarray) -> np.ndarray:
    return _gamma_pdf(t, 6.0, 1.0) - _gamma_pdf(t, 16.0, 1.0) / 6.0


def _gamma_pdf(t: np.ndarray, peak: float, dispersion: float) -> np.ndarray:
    """Gamma density with its mode at ``peak``, defined as 0 at t = 0."""
    a = peak / dispersion
    y = np.zeros_like(t)
    pos = t > 0
    y[pos] = np.exp(a * np.log(t[pos] / dispersion) - t[pos] / dispersion - gammaln(a + 1)) / dispersion
    return y
