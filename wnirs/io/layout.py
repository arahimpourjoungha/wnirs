"""Headband channel layouts."""
from __future__ import annotations

import warnings
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("S", "D", "C", "Row", "Column", "Type")


def builtin_layouts() -> list[str]:
    """Names of the layouts shipped with the package (``'full'``, ...)."""
    folder = resources.files("wnirs") / "data"
    return sorted(p.name[len("wnirs_layout_"):-len(".csv")]
                  for p in folder.iterdir() if p.name.startswith("wnirs_layout_"))


def load_layout(spec: str | Path | pd.DataFrame = "full") -> pd.DataFrame:
    """Load a wNIRS headband channel layout.

    The device writes every source x detector combination whether or not that
    pair is a real optode pair, so the layout is what defines the montage. Row
    order is the canonical channel order used everywhere downstream.

    Parameters
    ----------
    spec : str, Path or DataFrame
        A built-in name (``'full'``, ``'nirxsubset'``), a path to a CSV, or a
        table that is already loaded. A table that already carries ``Index``,
        ``Label`` and ``GridSlot`` is passed straight back, so the
        display-slot warnings are not repeated for every recording.

    Returns
    -------
    DataFrame
        The input columns (at least S, D, C, Row, Column, Type; Type 1 is a
        long channel and Type 2 a short-separation channel) plus

        ``Index``     1..n, the channel number
        ``Label``     ``'S1-D10'``, with ``' (sc)'`` appended for short channels
        ``GridSlot``  C, moved where needed so no two channels share a slot

    Raises
    ------
    FileNotFoundError
        When a name matches no built-in layout and is not a file.
    ValueError
        When a required column is missing.
    """
    if isinstance(spec, pd.DataFrame):
        if {"Index", "Label", "GridSlot"} <= set(spec.columns):
            return spec
        table = spec.copy()
    else:
        table = pd.read_csv(_resolve(spec))

    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"Layout is missing required column(s): {', '.join(missing)}")

    layout = table.reset_index(drop=True)
    for col in REQUIRED_COLUMNS:
        layout[col] = layout[col].astype(int)
    layout["Index"] = np.arange(1, len(layout) + 1)
    layout["Label"] = [f"S{s}-D{d}" + (" (sc)" if t == 2 else "")
                       for s, d, t in zip(layout.S, layout.D, layout.Type)]
    layout["GridSlot"] = _deconflict_grid(layout)

    if not (layout.Type == 2).any():
        warnings.warn("This layout declares no short-separation channels (Type == 2), "
                      "so short-separation regression will be unavailable.", stacklevel=2)
    return layout


def _resolve(spec: str | Path) -> Path:
    path = Path(spec)
    if path.is_file():
        return path
    folder = resources.files("wnirs") / "data"
    for name in (f"wnirs_layout_{spec}.csv", str(spec), f"{spec}.csv"):
        candidate = folder / name
        if candidate.is_file():
            return Path(str(candidate))
    raise FileNotFoundError(
        f'Layout "{spec}" not found. Built-in layouts: {", ".join(builtin_layouts())}. '
        "You can also pass a path to your own CSV.")


def _deconflict_grid(layout: pd.DataFrame) -> np.ndarray:
    """Give every channel its own display slot.

    Two channels sharing a slot would silently overwrite each other in the
    subplot grids. The first keeps it; a later one moves to the nearest free
    slot on the same display row (the lower slot on a tie).
    """
    slots = layout.C.to_numpy().copy()
    n_cols = int(layout.Column.max()) + 1
    n_rows = int(layout.Row.max())
    taken = np.zeros(n_rows * n_cols, dtype=bool)

    for i, s in enumerate(slots):
        if 1 <= s <= taken.size and not taken[s - 1]:
            taken[s - 1] = True
            continue
        row_start = (max(1, int(np.ceil(s / n_cols))) - 1) * n_cols
        row = taken[row_start:row_start + min(n_cols, taken.size - row_start)]
        free = np.flatnonzero(~row) + 1
        if free.size:
            new = row_start + free[np.argmin(np.abs(free + row_start - s))]
        elif (~taken).any():
            new = int(np.flatnonzero(~taken)[0]) + 1
        else:
            continue
        warnings.warn(f"Channels {layout.Label[i]} and an earlier channel both claim display "
                      f"slot {s}; {layout.Label[i]} moved to slot {new}.", stacklevel=3)
        slots[i] = new
        taken[new - 1] = True
    return slots
