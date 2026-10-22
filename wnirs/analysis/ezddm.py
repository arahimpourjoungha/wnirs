"""EZ-diffusion model (Wagenmakers, van der Maas & Grasman 2007)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EZDiffusion:
    """EZ-diffusion parameters for one set of trials.

    v    drift rate -- how fast evidence accumulates; the closest thing to
         "ability" on the task
    a    boundary separation -- how much evidence is required; caution
    ter  non-decision time (encoding plus motor), seconds
    pc   observed proportion correct (before any edge correction)
    mrt, vrt  mean and variance of correct-trial RT, seconds
    """
    v: float = np.nan
    a: float = np.nan
    ter: float = np.nan
    pc: float = np.nan
    mrt: float = np.nan
    vrt: float = np.nan
    n: int = 0
    n_correct: int = 0
    edge_corrected: bool = False


def ez_parameters(pc: float, vrt: float, mrt: float, s: float = 0.1) -> tuple[float, float, float]:
    """Closed-form EZ inversion from accuracy and correct-RT mean and variance.

    ``pc`` must already be edge corrected (strictly between 0.5 and 1).
    Returns ``(v, a, ter)``, NaN where the inversion is undefined.
    """
    s2 = s * s
    logit = np.log(pc / (1 - pc))
    x = logit * (logit * pc**2 - logit * pc + pc - 0.5) / vrt
    if not np.isfinite(x) or x <= 0:
        return np.nan, np.nan, np.nan
    v = np.sign(pc - 0.5) * s * x**0.25
    if not np.isfinite(v) or v == 0:
        return np.nan, np.nan, np.nan
    a = s2 * logit / v
    y = -v * a / s2
    mdt = (a / (2 * v)) * (1 - np.exp(y)) / (1 + np.exp(y))
    return float(v), float(a), float(mrt - mdt)


def ezddm(rt, correct, *, s: float = 0.1, min_trials: int = 10,
          rt_range: tuple[float, float] = (0.15, 5.0)) -> EZDiffusion:
    """EZ-diffusion parameters from trial RTs (seconds) and accuracy.

    EZ inverts the diffusion model in closed form from three summaries, so
    its three parameters are identified from the 60-170 trials a session
    has, where a hierarchical fit would want hundreds per cell.

    THE CEILING PROBLEM. EZ is undefined at Pc = 1 because the logit
    diverges. The standard edge correction Pc = 1 - 1/(2n) is applied and
    flagged in ``edge_corrected``; read a corrected drift rate as a lower
    bound. When most sessions of a task are edge corrected, drift carries
    little information and a and Ter are the parameters worth reading. Below
    Pc = 0.5 the model is not interpretable and NaNs are returned.

    Parameters
    ----------
    rt : array_like
        Response times in seconds. Zero means "no response recorded" and very
        long RTs are lapses; both are dropped by ``rt_range``.
    correct : array_like of bool
    s : float
        Diffusion scaling parameter; 0.1 is the convention the published
        formulas assume. Changing it rescales v and a together.
    min_trials : int
        Fewer valid trials return NaNs.
    rt_range : (lo, hi)
    """
    rt = np.asarray(rt, float).ravel()
    correct = np.asarray(correct, bool).ravel()
    if rt.size != correct.size:
        raise ValueError("rt and correct differ in length.")
    ok = np.isfinite(rt) & (rt >= rt_range[0]) & (rt <= rt_range[1])
    rt, correct = rt[ok], correct[ok]
    out = EZDiffusion(n=rt.size, n_correct=int(correct.sum()))
    if out.n < min_trials:
        return out
    pc = out.pc = float(correct.mean())
    if pc >= 1:
        pc = 1 - 1 / (2 * out.n)
        out.edge_corrected = True
    elif pc <= 0.5:
        return out
    rtc = rt[correct]
    if rtc.size < 3:
        return out
    out.mrt, out.vrt = float(rtc.mean()), float(np.var(rtc, ddof=1))
    if not np.isfinite(out.vrt) or out.vrt <= 0:
        return out
    out.v, out.a, out.ter = ez_parameters(pc, out.vrt, out.mrt, s)
    return out
