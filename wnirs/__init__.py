"""wnirs: a Python toolbox for dense-sampling wearable fNIRS.

Reading, quality control, signal processing and precision-imaging statistics
for wearable fNIRS, plus a learning module (wnirs.ml) for self-supervised
representation learning on wearable fNIRS time series and on-device deployment.
"""
from .types import Conditions, Hb, Raw

__version__ = "0.1.0"
__all__ = ["Raw", "Hb", "Conditions"]
