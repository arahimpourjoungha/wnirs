"""The full preprocessing pipeline: Raw intensity to filtered haemoglobin."""
from __future__ import annotations

import dataclasses
import logging
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ..types import Hb, Raw
from ._util import PreprocWarning
from .baseline import baseline_shift
from .filtering import DEFAULT_BAND, bandpass, detrend
from .mbll import mbll
from .motion import motion_correct
from .optical import (UNUSABLE, DarkMethod, dark_channels, dark_level, flag_channels,
                      optical_density)
from .regression import global_regress, lp_regress, ssr

log = logging.getLogger(__name__)

#: Readers that produce native wNIRS recordings (LP reference, unused pairs).
_WNIRS_SOURCES = ("wnirs", "wnirscsv", "wnirsreport")


@dataclass(frozen=True)
class PreprocessOptions:
    """Options of :func:`preprocess`.

    Attributes
    ----------
    dark : {"unused", "quantile", "none"} or float
        Dark-level estimate (see :func:`~wnirs.preproc.dark_level`). The step
        that most affects the results; read why before turning it off.
    sat_level, sat_tol, sat_max_frac : float
        Detector ceiling, how close to it counts as clipped, and the clipped
        fraction above which a channel is dropped rather than repaired
        (see :func:`~wnirs.preproc.flag_channels`).
    lp_regress : {"none", "perchannel", "mean"}
        Low-power reference regression. ``"none"`` by default.
    motion : str or tuple of str
        Motion method(s), see :data:`~wnirs.preproc.MOTION_METHODS`.
    motion_opts : dict
        Options shared by the motion methods and the baseline-shift step.
    baseline_shift : bool
        Remove DC steps before motion correction.
    global_signal : bool
        Regress the montage mean out of every channel: removes systemic
        physiology when shorts are unusable, but also any genuinely global
        response, so it is a connectivity tool rather than an activation one.
    filter : bool
        False skips frequency filtering entirely.
    lowpass : bool
        False keeps the high-pass but produces no low-passed display copy.
    band : (float, float)
        Pass band in Hz.
    median_win : int
        Median-filter length in samples (0 skips).
    detrend : bool
        Remove a linear trend after filtering.
    ssr : {"samesource", "mean", "none"}
        Short-separation regression mode.
    sd_distance : float
        Source-detector separation in mm, for Beer-Lambert.
    age : float
        Subject age in years, for the DPF.
    """
    dark: DarkMethod = "unused"
    sat_level: float = 5.0
    sat_tol: float = 0.02
    sat_max_frac: float = 0.20
    lp_regress: Literal["none", "perchannel", "mean"] = "none"
    motion: str | tuple[str, ...] = "tddr"
    motion_opts: dict = field(default_factory=dict)
    baseline_shift: bool = False
    global_signal: bool = False
    filter: bool = True
    lowpass: bool = True
    band: tuple[float, float] = DEFAULT_BAND
    median_win: int = 0
    detrend: bool = True
    ssr: Literal["samesource", "mean", "none"] = "samesource"
    sd_distance: float = 30.0
    age: float = 28.0


def preprocess(raw: Raw, **opts) -> Hb:
    """Raw headband intensity to filtered haemoglobin.

    Parameters
    ----------
    raw : Raw
        One recording. Optional inputs: ``raw.reference`` (low-power trace,
        for LP regression), ``raw.all_pairs`` with ``raw.all_pairs_index``
        (whose ``in_layout`` column picks out the unused pairs that measure
        the dark level) and ``raw.meta["source"]`` (the reader that produced
        it, default ``"wnirs"``).
    **opts
        Any field of :class:`PreprocessOptions`.

    Returns
    -------
    Hb
        ``hbo``/``hbr`` are the analysis copies (high-passed only);
        ``hbo_display``/``hbr_display`` are also low-passed, for figures.
        ``info`` holds the per-step diagnostics and ``steps`` what was done.

    Notes
    -----
    In order:

    1. flag saturated / missing channels (blanked to NaN, not zero) and
       repair intermittently clipped ones;
    2. subtract the detector dark level;
    3. intensity to optical density; channels that saw no light become "dark";
    4. regress out the low-power reference (optional);
    5. remove baseline shifts (optional) -- before motion correction, so the
       spike methods do not try to spline or wavelet through a step;
    6. motion correction on OD;
    7. modified Beer-Lambert;
    8. high-pass the analysis copy; low-pass a separate display copy;
    9. global signal regression (optional);
    10. linear detrend;
    11. short-separation regression with the usable shorts.

    Two copies leave step 8 on purpose. A low-pass makes a block average
    readable but removes part of the residual spectrum the AR estimators need
    to whiten; fitted on low-passed data, the AR coefficients are unstable and
    the t values untrustworthy. The GLM therefore always uses the
    high-passed-only copy.

    Recordings from other instruments have no LP reference and no unused
    pairs; those corrections are switched off (with a warning) rather than
    failing, since the rest of the pipeline is instrument-agnostic.
    """
    o = _adapt_to_source(raw, PreprocessOptions(**opts))
    fs = raw.fs
    layout = raw.layout
    steps: list[str] = []
    info: dict = {"opts": dataclasses.asdict(o)}

    # 1. saturated and dead channels
    status, I, clip_mask = flag_channels(raw.intensity, o.sat_level, o.sat_tol, o.sat_max_frac)
    ref = None
    if raw.reference is not None:
        ref = np.array(raw.reference, dtype=float)
        ref[:, np.isin(status, ("saturated", "missing"))] = np.nan
    _report_flags(status, layout, o.sat_max_frac)
    steps.append("flagged saturated and dead channels")

    # 2. dark level
    dark, info["dark"] = dark_level(raw.intensity, o.dark, _unused_pairs(raw))
    offset = np.where(np.isfinite(dark), dark, 0.0)
    I = I - offset
    if ref is not None:
        ref = ref - offset
    if np.isfinite(dark).any() and not _is(o.dark, "none"):
        steps.append(f"dark level subtracted ({' / '.join(f'{d:g}' for d in dark)}, "
                     f"method: {info['dark']['method']})")
    info["dark"]["level"] = dark

    # 3. optical density
    od = optical_density(I)
    od_ref = None if ref is None else optical_density(ref)
    steps.append("intensity -> optical density")
    # A saturated channel is already NaN and keeps its more informative status.
    status[(status == "ok") & dark_channels(I)] = "dark"

    # 4. low-power regression
    if o.lp_regress != "none":
        od, info["lp"] = lp_regress(od, od_ref, o.lp_regress)
        steps.append(f"low-power reference regressed out ({o.lp_regress})")

    # 5. baseline shifts
    if o.baseline_shift:
        bs_opts = {k: v for k, v in o.motion_opts.items() if k in ("method", "tune", "pmax")}
        info["baseline"] = []
        for w in range(od.shape[2]):
            od[:, :, w], bs = baseline_shift(od[:, :, w], fs, **bs_opts)
            info["baseline"].append(bs)
        n_shift = sum(int(bs["n_shifts"].sum()) for bs in info["baseline"])
        steps.append(f"baseline-shift correction ({n_shift} step(s) removed)")

    # 6. motion correction
    methods = (o.motion,) if isinstance(o.motion, str) else tuple(o.motion)
    if [m.lower() for m in methods] != ["none"]:
        info["motion"] = []
        for w in range(od.shape[2]):
            od[:, :, w], mi = motion_correct(od[:, :, w], fs, methods, **o.motion_opts)
            info["motion"].append(mi)
        steps.append("motion correction on OD: " + " then ".join(methods))

    # 7. Beer-Lambert
    hbo, hbr = mbll(od, raw.wavelengths, o.sd_distance, o.age)
    steps.append(f"modified Beer-Lambert ({'/'.join(f'{w:g}' for w in raw.wavelengths)} nm, "
                 f"d={o.sd_distance:g} mm, age={o.age:g})")

    # 8. filtering: analysis copy high-passed, display copy also low-passed
    display: list[np.ndarray] = []
    if o.filter and o.band:
        lo, hi = o.band
        hbo = bandpass(hbo, fs, (lo, np.nan), o.median_win)
        hbr = bandpass(hbr, fs, (lo, np.nan), o.median_win)
        steps.append(f"high-pass {lo:g} Hz (analysis copy)")
        if o.lowpass and np.isfinite(hi):
            display = [bandpass(h, fs, (np.nan, hi), o.median_win) for h in (hbo, hbr)]
            steps.append(f"additional low-pass {hi:g} Hz (display copy only)")
    else:
        steps.append("no frequency filtering")

    usable = ~np.isin(status, UNUSABLE)
    is_short = layout["Type"].to_numpy() == 2

    # 9. global signal regression
    if o.global_signal:
        include = ~is_short & usable
        (hbo, gi_o), (hbr, gi_r) = global_regress(hbo, include), global_regress(hbr, include)
        display = [global_regress(h, include)[0] for h in display]
        info["global"] = {"hbo": gi_o, "hbr": gi_r}
        steps.append("global signal regressed out")

    # 10. detrend
    if o.detrend:
        hbo, hbr = detrend(hbo), detrend(hbr)
        display = [detrend(h) for h in display]
        steps.append("linear detrend")

    # 11. short-separation regression. A compressed or occasionally clipped
    # short is still a usable regressor: SSR fits a free coefficient, and the
    # fit skips the clipped samples.
    if o.ssr != "none":
        if not (is_short & usable).any():
            warnings.warn(_no_short_message(is_short.sum()), PreprocWarning, stacklevel=2)
        else:
            sources = layout["S"].to_numpy()

            def regress(h):
                return ssr(h, sources, is_short, o.ssr, usable, clip_mask)

            (hbo, si_o), (hbr, si_r) = regress(hbo), regress(hbr)
            display = [regress(h)[0] for h in display]
            info["ssr"] = {"hbo": si_o, "hbr": si_r, "mode": o.ssr}
            steps.append(f"short-separation regression ({o.ssr})")

    _log_summary(raw, status)
    hbo_disp, hbr_disp = display if display else (None, None)
    return Hb(subject=raw.subject, task=raw.task, session=raw.session, time=raw.time, fs=fs,
              hbo=hbo, hbr=hbr, layout=layout, conditions=raw.conditions,
              channel_status=status, od=od, hbo_display=hbo_disp, hbr_display=hbr_disp,
              steps=steps, info=info,
              meta={**raw.meta, "file": raw.file, "stim": raw.stim, "trials": raw.trials,
                    "wavelengths": tuple(raw.wavelengths)})


def _adapt_to_source(raw: Raw, o: PreprocessOptions) -> PreprocessOptions:
    """Switch off the corrections a non-wNIRS recording cannot support."""
    src = str(raw.meta.get("source", "wnirs")).lower()
    if src in _WNIRS_SOURCES:
        return o
    changes = {}
    if o.lp_regress != "none" and raw.reference is None:
        warnings.warn(f"{src.upper()} data carries no low-power reference, so LP regression "
                      "is switched off", PreprocWarning, stacklevel=3)
        changes["lp_regress"] = "none"
    if _is(o.dark, "unused") and _unused_pairs(raw) is None:
        warnings.warn(f"{src.upper()} data has no unused source-detector pairs; the dark "
                      "level falls back to the 1st percentile. If the export is already "
                      "dark-corrected, pass dark='none'", PreprocWarning, stacklevel=3)
        changes["dark"] = "quantile"
    if o.ssr != "none" and not (raw.layout["Type"] == 2).any():
        warnings.warn("no short-separation channels in this montage, so SSR is switched "
                      "off; consider global_signal=True", PreprocWarning, stacklevel=3)
        changes["ssr"] = "none"
    return dataclasses.replace(o, **changes)


def _is(option, name: str) -> bool:
    """True when a string-or-number option is the named mode."""
    return isinstance(option, str) and option.lower() == name


def _unused_pairs(raw: Raw) -> np.ndarray | None:
    """Intensity of the source-detector pairs that are not montage channels."""
    if raw.all_pairs is None or raw.all_pairs_index is None:
        return None
    return raw.all_pairs[:, ~raw.all_pairs_index["in_layout"].to_numpy(dtype=bool)]


def _labels(layout: pd.DataFrame, mask: np.ndarray) -> list[str]:
    if "Label" in layout:
        return list(layout.loc[mask, "Label"].astype(str))
    sub = layout.loc[mask]
    return [f"S{s}-D{d}" for s, d in zip(sub["S"], sub["D"])]


_FLAG_TEXT = {
    "saturated": "are clipped at the detector ceiling on more than {frac:.0%} of samples "
                 "and excluded",
    "intermittentClip": "clip at the ceiling on some samples; kept, with the clipped "
                        "stretches interpolated",
    "nearCeiling": "sit within 10 % of the ceiling without clipping; kept, but their "
                   "response is compressed -- lower the LED drive to record them properly",
}


def _report_flags(status: np.ndarray, layout: pd.DataFrame, sat_max_frac: float) -> None:
    for tag, what in _FLAG_TEXT.items():
        mask = status == tag
        if mask.any():
            what = what.format(frac=sat_max_frac)
            warnings.warn(f"{mask.sum()} channel(s) {what}: {', '.join(_labels(layout, mask))}",
                          PreprocWarning, stacklevel=3)


def _no_short_message(n_short: int) -> str:
    if n_short == 0:
        return "this layout has no short-separation channels, so SSR was skipped"
    return (f"all {n_short} short-separation channel(s) are saturated or dark in this "
            "recording, so SSR was skipped; scalp haemodynamics remain in the data")


def _log_summary(raw: Raw, status: np.ndarray) -> None:
    flagged = Counter(s for s in status if s != "ok")
    n_dropped = sum(flagged[s] for s in UNUSABLE)
    if n_dropped or flagged["nearCeiling"]:
        log.warning("%s / %s: %d of %d channels excluded, %d flagged (%s)", raw.subject,
                    raw.task, n_dropped, status.size, flagged["nearCeiling"],
                    ", ".join(f"{n} {s}" for s, n in sorted(flagged.items())))
