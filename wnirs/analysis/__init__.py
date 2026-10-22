"""Subject- and session-level statistics: GLM, connectivity, reliability, dense sampling.

NumPy/SciPy implementations, plus a JAX implementation of the per-channel AR-IRLS GLM
(:mod:`wnirs.analysis.glm_jax`, imported on demand so JAX stays optional).
"""
from .ar import arfit, whiten
from .arirls import ARIRLSFit, ar_irls
from .block_average import BlockAverage, ConditionAverage, block_average
from .change_point import change_point
from .connectivity import Connectivity, connectivity, fisher_z
from .contrast import contrast, default_contrasts, parse_contrast
from .correlate import RMCorr, channel_behaviour, rmcorr
from .dense_sampling import dense_sampling, pattern_reliability
from .design import Design, design_matrix
from .ezddm import EZDiffusion, ez_parameters, ezddm
from .fdr import fdr_bh
from .glm import ChannelFits, GLMResult, fit_channels, glm
from .growth_curve import GrowthCurve, growth_curve
from .gtheory import GTheory, gtheory
from .hrf import hrf
from .iv_metrics import iv_metrics
from .mixed import MixedFit, fit_mixed
from .reliability import Fingerprint, ReliabilityResult, fingerprint, icc21, reliability
from .robust import RobustFit, robust_fit
from .sessions import SessionMeasure
from .state_trait import state_trait

__all__ = [
    "ARIRLSFit", "BlockAverage", "ChannelFits", "ConditionAverage", "Connectivity", "Design", "EZDiffusion",
    "Fingerprint", "GLMResult", "GTheory", "GrowthCurve", "MixedFit", "RMCorr", "ReliabilityResult", "RobustFit",
    "SessionMeasure", "ar_irls", "arfit", "block_average", "change_point", "channel_behaviour", "connectivity",
    "contrast", "default_contrasts", "dense_sampling", "design_matrix", "ez_parameters", "ezddm", "fdr_bh",
    "fingerprint", "fisher_z", "fit_channels", "fit_mixed", "glm", "growth_curve", "gtheory", "hrf", "icc21",
    "iv_metrics", "parse_contrast", "pattern_reliability", "reliability", "rmcorr", "robust_fit",
    "state_trait", "whiten",
]
