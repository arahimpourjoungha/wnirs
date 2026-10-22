"""Core containers passed between pipeline stages.

A recording moves through three shapes:

    Raw   light intensity as read from the device   (io  -> qc, preproc)
    Hb    haemoglobin concentration changes          (preproc -> analysis)
    then per-session results (GLM betas, connectivity) that analysis returns
    as plain DataFrames / arrays.

Each stage takes and returns one of these, so any step can be swapped or run
on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Conditions:
    """Task events, one row per condition."""
    name: list[str] = field(default_factory=list)
    onsets: list[np.ndarray] = field(default_factory=list)      # seconds, same clock as Raw.time
    durations: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.name)

    @property
    def n_blocks(self) -> list[int]:
        return [len(o) for o in self.onsets]


@dataclass
class Raw:
    """One recording of one task, as read from disk.

    intensity : (n_samples, n_channels, n_wavelengths), detector reading per layout channel
    reference : same shape as intensity, the low-power (LP) reading where the device has one
    led_power : (n_samples, n_channels, n_wavelengths) LED drive level, NaN if absent
    layout    : DataFrame with columns S, D, C, Row, Column, Type (1 long, 2 short), Comparison
    """
    subject: str
    task: str
    session: str
    file: Path
    time: np.ndarray
    fs: float
    intensity: np.ndarray
    layout: pd.DataFrame
    wavelengths: tuple[float, float] = (750.0, 860.0)
    stim: Optional[np.ndarray] = None
    reference: Optional[np.ndarray] = None
    led_power: Optional[np.ndarray] = None
    all_pairs: Optional[np.ndarray] = None     # every S-D combination incl. unused, for the dark level
    conditions: Conditions = field(default_factory=Conditions)
    trials: Optional[pd.DataFrame] = None
    trimmed_at: Optional[float] = None
    meta: dict = field(default_factory=dict)
    all_pairs_index: Optional[pd.DataFrame] = None   # one row per all_pairs column: S, D, in_layout
    reference_led_power: Optional[np.ndarray] = None  # LED drive level of the LP reference

    @property
    def n_channels(self) -> int:
        return self.intensity.shape[1]


@dataclass
class Hb:
    """Concentration changes (micromolar) after preprocessing.

    hbo / hbr are the ANALYSIS copies (high-passed only); the GLM always uses
    these. hbo_display / hbr_display are additionally low-passed, for figures.
    channel_status is one string per channel: "ok", "nearCeiling", "saturated",
    "missing", "dark", or a failed QC gate name.
    """
    subject: str
    task: str
    session: str
    time: np.ndarray
    fs: float
    hbo: np.ndarray
    hbr: np.ndarray
    layout: pd.DataFrame
    conditions: Conditions
    channel_status: np.ndarray
    od: Optional[np.ndarray] = None
    hbo_display: Optional[np.ndarray] = None
    hbr_display: Optional[np.ndarray] = None
    steps: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)    # ssr / motion / dark-level diagnostics
    meta: dict = field(default_factory=dict)

    @property
    def usable(self) -> np.ndarray:
        bad = {"saturated", "missing", "dark"}
        return np.array([s not in bad for s in self.channel_status])

    @property
    def is_long(self) -> np.ndarray:
        return self.layout["Type"].to_numpy() == 1
