"""Preprocessing: raw intensity to haemoglobin concentration.

Every step is a standalone function on plain arrays (plus ``fs`` where it
matters), so it can be reused outside the :class:`~wnirs.Raw` /
:class:`~wnirs.Hb` containers; :func:`preprocess` chains them in the
default order.
"""
from ._util import PreprocWarning
from .baseline import arfit, baseline_shift
from .filtering import DEFAULT_BAND, bandpass, butter_filtfilt, detrend
from .mbll import WNIRS_WAVELENGTHS, dpf, extinction, mbll
from .motion import MOTION_METHODS, motion_correct, pca_correct, savgol_correct
from .optical import UNUSABLE, dark_channels, dark_level, flag_channels, optical_density
from .pipeline import PreprocessOptions, preprocess
from .regression import global_regress, lp_regress, ssr
from .spline import smoothing_spline, spline_correct
from .tddr import tddr
from .wavelet import wavelet_correct

__all__ = [
    "DEFAULT_BAND", "MOTION_METHODS", "UNUSABLE", "WNIRS_WAVELENGTHS",
    "PreprocWarning", "PreprocessOptions",
    "arfit", "bandpass", "baseline_shift", "butter_filtfilt", "dark_channels", "dark_level",
    "detrend", "dpf", "extinction", "flag_channels", "global_regress", "lp_regress", "mbll",
    "motion_correct", "optical_density", "pca_correct", "preprocess", "savgol_correct",
    "smoothing_spline", "spline_correct", "ssr", "tddr", "wavelet_correct",
]
