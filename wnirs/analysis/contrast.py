"""Linear contrasts of GLM regressors, and the contrast each task is reported with."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

from ._stats import EPS, channel_labels, t_pvalue
from .fdr import fdr_bh
from .glm import CHROMOPHORES, GLMResult

_COEF = re.compile(r"^\s*([0-9.]+)\s*\*\s*(.+)$")


def contrast(G: GLMResult, spec: str | Sequence[float], name: str | None = None) -> pd.DataFrame:
    """Test a linear combination of GLM regressors, per channel and chromophore.

    Parameters
    ----------
    G : GLMResult
    spec : str or sequence of float
        A formula over ``G.names`` -- ``'2-Back'`` (that condition against
        zero), ``'2-Back - 0-Back'``, ``'0.5*2-Back + 0.5*3-Back - 0-Back'`` --
        or an explicit weight vector with one entry per regressor. Names may
        themselves contain hyphens.
    name : str, optional
        Display name; defaults to the formula.

    Returns
    -------
    pandas.DataFrame
        One row per channel per chromophore with columns Channel, Source,
        Detector, Label, Chromophore, Beta, SE, Tstat, Pval, Qval. ``attrs``
        holds the weight vector, name and regressor names.

        Qval is Benjamini-Hochberg over the channels of each chromophore
        separately: HbO and HbR are not independent tests of different things,
        so pooling them would be too conservative.
    """
    if isinstance(spec, str):
        c = parse_contrast(spec, G.names)
        name = spec if name is None else name
    else:
        c = np.asarray(spec, float).ravel()
        if c.size != len(G.names):
            raise ValueError(f"Weight vector has {c.size} entries but the model has {len(G.names)} regressors.")
        name = "contrast" if name is None else name

    n_ch = G.beta.shape[1]
    # Channels with any non-finite beta or covariance entry are left untested.
    ok = np.all(np.isfinite(G.beta), axis=0) & np.all(np.isfinite(G.cov), axis=(0, 1))     # (n_ch, 2)
    with np.errstate(invalid="ignore"):
        est = np.einsum("p,pkc->kc", c, G.beta)
        se = np.sqrt(np.maximum(np.einsum("p,pqkc,q->kc", c, G.cov, c), 0))
    est[~ok], se[~ok] = np.nan, np.nan
    t = est / np.maximum(se, EPS)
    p = t_pvalue(t, G.dfe)

    labels = channel_labels(G.layout)
    frames = [pd.DataFrame({
        "Channel": np.arange(1, n_ch + 1), "Source": G.layout["S"].to_numpy(),
        "Detector": G.layout["D"].to_numpy(), "Label": labels, "Chromophore": chromo,
        "Beta": est[:, ci], "SE": se[:, ci], "Tstat": t[:, ci], "Pval": p[:, ci], "Qval": fdr_bh(p[:, ci]),
    }) for ci, chromo in enumerate(CHROMOPHORES)]
    out = pd.concat(frames, ignore_index=True)
    out.attrs = dict(contrast=c.tolist(), name=name, names=list(G.names), subject=G.subject, task=G.task)
    return out


def parse_contrast(spec: str, names: Sequence[str]) -> np.ndarray:
    """Weight vector for a contrast formula over regressor ``names``.

    The formula is split on + and -, and any fragment that is not a
    recognisable term is re-joined with the next, so a name like '2-Back'
    survives the split.
    """
    names = list(names)
    tokens = _rejoin(re.findall(r"[+-]?[^+-]+", spec.strip()), names)
    c = np.zeros(len(names))
    found = False
    for tok in tokens:
        s = tok.strip()
        if not s:
            continue
        sign = -1.0 if s[0] == "-" else 1.0
        if s[0] in "+-":
            s = s[1:].strip()
        coef = 1.0
        m = _COEF.match(s)
        if m:
            coef, s = float(m.group(1)), m.group(2).strip()
        if s not in names:
            raise ValueError(f'"{s}" is not a regressor in this model.\nAvailable: {", ".join(names)}')
        c[names.index(s)] += sign * coef
        found = True
    if not found:
        raise ValueError(f'Could not parse the contrast "{spec}".')
    return c


def _is_term(s: str, names: list[str]) -> bool:
    s = s.strip()
    if not s:
        return False
    if s[0] in "+-":
        s = s[1:].strip()
    m = _COEF.match(s)
    return (m.group(2).strip() if m else s) in names


def _rejoin(tokens: list[str], names: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(tokens):
        s = tokens[i].strip()
        while i < len(tokens) - 1 and not _is_term(s, names):
            i += 1
            s += tokens[i].strip()
        out.append(s)
        i += 1
    return out


def default_contrasts(condition_names: Sequence[str]) -> list[str]:
    """The contrast(s) a task is normally reported with.

    In a GLM, "condition versus zero" is really "condition versus everything
    not modelled" -- the implicit baseline. These tasks run at a high duty cycle
    (the pilot N-Back keeps a stimulus on screen 48 percent of the time), so
    there is little unmodelled rest for that comparison to lean on, and the
    intercept and drift absorb most of what is left. The differential contrast
    is the one the design supports, and it also cancels the systemic and scalp
    signal common to both conditions.

    Recognised, in order: n-back (highest load minus lowest, and every
    intermediate load minus lowest, which shows whether the response scales
    with load), ``incongruent - congruent``, ``NoGo - Go``. Anything else falls
    back to every condition against zero, so an unfamiliar design still
    produces something.
    """
    names = [str(n) for n in condition_names]
    lower = [n.lower() for n in names]
    specs: list[str] = []
    loads = [(int(m.group(1)), n) for n in names if (m := re.fullmatch(r"(\d+)-Back", n))]
    if len(loads) >= 2:
        levels = [lv for lv, _ in loads]
        lo = loads[int(np.argmin(levels))][1]
        hi = loads[int(np.argmax(levels))][1]
        specs.append(f"{hi} - {lo}")
        specs += [f"{n} - {lo}" for lv, n in loads if min(levels) < lv < max(levels)]
    if "incongruent" in lower and "congruent" in lower:
        specs.append("incongruent - congruent")
    if "nogo" in lower and "go" in lower:
        specs.append("NoGo - Go")
    return specs or names
