"""Channel quality: metrics, pass/fail gates, and a quick post-session light check.

:func:`channel_qc` computes the metrics in :data:`METRICS` for every channel of
a :class:`~wnirs.Raw` (plus HbCorr and the task metrics when given the
:class:`~wnirs.Hb`) and judges them against the gates in
:data:`DEFAULT_THRESHOLDS`. :func:`session_check` answers the narrower question
asked at the end of a session: did enough light reach the detector?
"""
from .catalogue import METRICS, Metric, metric_table, resolve_metrics
from .channel import channel_qc
from .gates import DEFAULT_THRESHOLDS, GATE_OPS, gate_op, judge, merge_thresholds
from .metrics import mean_dark_level, saturation_ceiling
from .session_check import SessionCheckOptions, session_check, session_check_report

__all__ = [
    "DEFAULT_THRESHOLDS", "GATE_OPS", "METRICS",
    "Metric", "SessionCheckOptions",
    "channel_qc", "gate_op", "judge", "mean_dark_level", "merge_thresholds", "metric_table",
    "resolve_metrics", "saturation_ceiling", "session_check", "session_check_report",
]
