"""Task performance from the tablet's behavioural workbooks.

The tablet writes one workbook per task per session, with a single sheet named
``TEST`` holding a row per trial (stimuliPresented, stimuliType or block,
accuracy, omission, responseTime, commissionError). The summaries here are
computed from those trials directly.

Behaviour comes off the tablet and never touches the optics, so it is immune
to whatever limits the haemodynamic measurements in a session.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .names import parse_brain_games_name


def summarise_trials(trials: pd.DataFrame, task: str) -> dict[str, float]:
    """Accuracy, response time and the task's condition contrasts from one TEST sheet.

    Accuracy, omission and false-alarm rates are percentages of trials; RTs
    are in milliseconds (the sheet stores seconds). A response time of zero
    means no response was recorded, not an instant answer, so it is left out
    of every RT mean. A blank accuracy cell counts as an error.

    Task-specific measures:

    Flanker  ``Acc_/RT_congruent``, ``Acc_/RT_incongruent`` and ``RTcost``
             (incongruent minus congruent RT).
    NBack    ``Acc_<block>`` per load level, made a valid identifier
             (``0-Back`` -> ``Acc_x0_Back``), and
             ``LoadCost`` (0-back minus 2-back accuracy).
    GoNogo   ``FalseAlarmRate`` (commission errors on the X), ``Acc_NoGo``,
             ``Acc_Go`` and ``RT_Go``.

    Returns an empty dict when the sheet has no accuracy column.
    """
    cols = [str(c) for c in trials.columns]
    acc = _numeric_col(trials, cols, "accuracy")
    if acc is None:
        return {}
    rt = _numeric_col(trials, cols, "responseTime")
    rt_ok = np.full(acc.size, np.nan) if rt is None else np.where(rt > 0, rt, np.nan)
    om = _numeric_col(trials, cols, "omission")
    com = _numeric_col(trials, cols, "commissionError")

    m = {"Accuracy": _pct(acc == 1)}
    if np.isfinite(rt_ok).any():
        m["RT"] = 1000 * np.nanmean(rt_ok)
    if om is not None:
        m["Omission"] = _pct(om == 1)

    if task == "Flanker":
        kind = _text_col(trials, cols, "stimuliType")
        if kind is not None:
            kind = np.array([k.lower() for k in kind])
            for level in ("congruent", "incongruent"):
                k = kind == level
                if not k.any():
                    continue
                m[f"Acc_{level}"] = _pct(acc[k] == 1)
                if np.isfinite(rt_ok[k]).any():
                    m[f"RT_{level}"] = 1000 * np.nanmean(rt_ok[k])
            if "RT_incongruent" in m and "RT_congruent" in m:
                m["RTcost"] = m["RT_incongruent"] - m["RT_congruent"]
    elif task == "NBack":
        block = _text_col(trials, cols, "block")
        if block is not None:
            for level in pd.unique(block):
                m[f"Acc_{valid_name(level)}"] = _pct(acc[block == level] == 1)
            if "Acc_x0_Back" in m and "Acc_x2_Back" in m:
                m["LoadCost"] = m["Acc_x0_Back"] - m["Acc_x2_Back"]
    elif task == "GoNogo":
        shown = _text_col(trials, cols, "stimuliPresented")
        if shown is not None:
            nogo = np.array([s.upper() == "X" for s in shown])
            if nogo.any() and com is not None:
                # The point of the task: withholding on the rare X.
                m["FalseAlarmRate"] = _pct(com[nogo] == 1)
                m["Acc_NoGo"] = _pct(acc[nogo] == 1)
            if (~nogo).any():
                m["Acc_Go"] = _pct(acc[~nogo] == 1)
                if np.isfinite(rt_ok[~nogo]).any():
                    m["RT_Go"] = 1000 * np.nanmean(rt_ok[~nogo])
    return m


def behaviour_table(data_root: str | Path) -> pd.DataFrame:
    """One row per subject and session, every task's summary side by side.

    Reads every ``*BehavioralData*.xlsx`` under ``data_root``. Columns are
    ``Subject``, ``Session`` and ``<Task>_<measure>`` from
    :func:`summarise_trials`, e.g. ``Flanker_RTcost``; a measure a session
    lacks is NaN. The tablet's demo account (``testuser``) parses like a
    participant and is skipped. When a session has two workbooks for one
    task, the later file (by name, so by timestamp) wins.

    Rows are sorted by subject, then numerically by session.
    """
    rows: dict[tuple[str, str], dict[str, object]] = {}
    for f in sorted(Path(data_root).rglob("*BehavioralData*.xlsx")):
        if f.name.startswith((".", "~$")) or "testuser" in f.name.lower():
            continue
        info = parse_brain_games_name(f.stem)
        if not info.task or not info.subject:
            continue
        try:
            sheet = pd.read_excel(f, sheet_name="TEST")
        except ValueError:
            continue
        summary = summarise_trials(sheet, info.task)
        if not summary:
            continue
        row = rows.setdefault((info.subject, info.session),
                              {"Subject": info.subject, "Session": info.session})
        row.update({f"{info.task}_{k}": v for k, v in summary.items()})

    if not rows:
        warnings.warn(f"No behavioural sheets read under {data_root}.", stacklevel=2)
        return pd.DataFrame(columns=["Subject", "Session"])
    table = pd.DataFrame(list(rows.values()))
    order = pd.to_numeric(table.Session.str.replace(r"\D", "", regex=True), errors="coerce")
    return (table.assign(_n=order).sort_values(["Subject", "_n"], kind="stable")
            .drop(columns="_n").reset_index(drop=True))


def valid_name(text: str) -> str:
    """A valid identifier (at most 63 characters) from a label the tablet writes.

    Whitespace is dropped and the next character capitalised, other invalid
    characters become ``_``, and a name not starting with a letter gets an
    ``x`` prefix (``'0-Back'`` -> ``'x0_Back'``), so column names stay stable.
    """
    s = re.sub(r"\s+(\w)", lambda m: m.group(1).upper(), text.strip())
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    if not s or not s[0].isalpha():
        s = "x" + s
    return s[:63]


def _pick(cols: list[str], name: str) -> Optional[str]:
    """Case-insensitive exact column name, else the first that contains it."""
    lower = [c.lower() for c in cols]
    if name.lower() in lower:
        return cols[lower.index(name.lower())]
    return next((c for c, lc in zip(cols, lower) if name.lower() in lc), None)


def _numeric_col(df: pd.DataFrame, cols: list[str], name: str) -> Optional[np.ndarray]:
    # Some columns arrive as text because one cell in the sheet is not numeric.
    col = _pick(cols, name)
    return None if col is None else pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def _text_col(df: pd.DataFrame, cols: list[str], name: str) -> Optional[np.ndarray]:
    col = _pick(cols, name)
    if col is None:
        return None
    return np.array(["" if pd.isna(v) else str(v).strip() for v in df[col]], dtype=object)


def _pct(hit: np.ndarray) -> float:
    return 100 * float(np.mean(hit)) if hit.size else np.nan
