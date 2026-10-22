"""Modified Beer-Lambert law: optical density to haemoglobin concentration."""
from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd

#: True emitter wavelengths of the wNIRS headband, in nm (see :func:`mbll`).
WNIRS_WAVELENGTHS: tuple[float, float] = (750.0, 860.0)


def dpf(age: float | np.ndarray, wavelength: float | np.ndarray) -> np.ndarray:
    """Differential pathlength factor (Scholkmann & Wolf, 2013).

    Parameters
    ----------
    age : float or array_like
        Subject age in years.
    wavelength : float or array_like
        Wavelength in nm.

    Returns
    -------
    numpy.ndarray
        The DPF, broadcast over ``age`` and ``wavelength``.

    Notes
    -----
    Valid for roughly 690-830 nm and ages 1-50. Outside that range the formula
    still evaluates but is an extrapolation. The DPF scales haemoglobin
    amplitudes, not their timing or sign, so it changes the micromolar numbers
    but not which channels respond.
    """
    age = np.asarray(age, dtype=float)
    lam = np.asarray(wavelength, dtype=float)
    return (223.3
            + 0.05624 * age ** 0.8493
            - 5.723e-7 * lam ** 3
            + 0.001245 * lam ** 2
            - 0.9025 * lam)


@lru_cache(maxsize=1)
def _prahl() -> pd.DataFrame:
    path = resources.files("wnirs") / "data" / "extinction_prahl.csv"
    with resources.as_file(path) as f:
        # Line 2 of the file holds units, not data.
        return pd.read_csv(f, skiprows=[1], encoding="utf-8-sig").astype(float)


def extinction(wavelengths) -> np.ndarray:
    """Molar extinction coefficients of HbO and HbR (Prahl), in cm^-1/M.

    Parameters
    ----------
    wavelengths : array_like
        Wavelengths in nm. Prahl's table is on a 2 nm grid and is linearly
        interpolated, so any wavelength inside 250-1000 nm works.

    Returns
    -------
    numpy.ndarray
        ``(n_wavelengths, 2)``: column 0 is HbO, column 1 is HbR. These are
        base-10 coefficients, consistent with ``OD = -log10(I / mean(I))``.
    """
    lam = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    table = _prahl()
    grid = table["lambda"].to_numpy()
    if lam.min() < grid[0] or lam.max() > grid[-1]:
        raise ValueError(f"wavelengths {lam} fall outside the extinction table "
                         f"({grid[0]:g}-{grid[-1]:g} nm)")
    return np.column_stack([np.interp(lam, grid, table["Hb02"].to_numpy()),
                            np.interp(lam, grid, table["Hb"].to_numpy())])


def mbll(od: np.ndarray, wavelengths=WNIRS_WAVELENGTHS, sd_distance: float = 30.0,
         age: float = 28.0) -> tuple[np.ndarray, np.ndarray]:
    """Convert optical density to HbO / HbR with the modified Beer-Lambert law.

    Parameters
    ----------
    od : numpy.ndarray
        ``(T, C, W)`` base-10 optical density, ``od[..., k]`` at
        ``wavelengths[k]``.
    wavelengths : sequence of float
        Emitter wavelengths in nm; must match ``od.shape[-1]``.
    sd_distance : float
        Source-detector separation in mm.
    age : float
        Subject age in years, for the DPF.

    Returns
    -------
    hbo, hbr : numpy.ndarray
        ``(T, C)`` concentration changes in micromolar.

    Notes
    -----
    Each wavelength contributes one equation,
    ``OD(l) = (eps_HbO(l) [HbO] + eps_HbR(l) [HbR]) * d * DPF(age, l)``.
    Two wavelengths are inverted exactly; three or more (Shimadzu, some
    Hitachi ETG models) are solved by least squares over all of them rather
    than by discarding the extras, which would return plausible numbers from
    incomplete data. A wavelength count that disagrees with the data is an
    error for the same reason.

    750 and 860 nm are correct for the wNIRS headband even though every export
    labels its columns 740/850: those are firmware tags, not the emitters.
    Using the labels leaves HbO almost unchanged but biases HbR by about 7 %
    and manufactures crosstalk -- a pure 6 uM HbO response comes back as
    5.46 HbO plus 0.60 of an HbR that was never there.

    OD must be base 10 here, matching Prahl's base-10 coefficients. Homer and
    nirs-toolbox use natural-log OD with rescaled spectra, so absolute
    micromolar values differ from theirs by a constant; t statistics,
    correlations and reliability measures are unaffected.

    With one ``sd_distance`` for the whole montage, short channels are scaled
    as if they were long. That is harmless for short-separation regression,
    which fits its own coefficient, but their absolute values mean nothing.
    """
    od = np.asarray(od, dtype=float)
    lam = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    if lam.size < 2:
        raise ValueError(f"two wavelengths are needed to separate HbO from HbR; {lam.size} given")
    if od.shape[-1] != lam.size:
        raise ValueError(f"{lam.size} wavelength(s) given but the optical density has "
                         f"{od.shape[-1]}; a wavelength would be paired with the wrong data")

    path_cm = (sd_distance / 10.0) * dpf(age, lam)
    E = extinction(lam) * path_cm[:, None]
    if np.linalg.matrix_rank(E) < 2:
        raise ValueError(f"wavelengths {lam} are not independent enough to separate HbO from HbR")

    # pinv is the exact inverse for two wavelengths and the least-squares
    # solution for more, so one expression covers both. 1e6: M -> uM.
    conc = od @ np.linalg.pinv(E).T * 1e6
    return conc[..., 0], conc[..., 1]
