"""Sensor-data tables: the column-per-optode CSV the headband writes.

Every source x detector combination is written at two wavelengths and two LED
drive levels:

    RP  "regular power", the measurement
    LP  "low power", the same optical path driven at a low LED level

LP is a reference, not a separate channel and not a short channel. With far
less emitted light it mostly carries ambient light, detector dark current and
drift, which is what makes it useful to regress out of RP.

Headers appear as ``S1_D1_740nm_RP`` (CSV export) or ``S1-D1-740nm-RP``
(tablet and Excel report), with or without leading spaces; both are matched.
The 740/850 in the header are the firmware's labels, not the true emitter
wavelengths (750/860), so they are used only to find columns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

WAVELENGTH_TAGS = ("740", "850")


@dataclass
class SensorTable:
    """A sensor CSV as read: canonical column keys and a numeric matrix.

    ``keys`` are the header names reduced to one form (upper case, ``-`` to
    ``_``, spaces removed) so CSV and Excel exports match the same way.
    """
    keys: list[str]
    data: np.ndarray

    def column(self, name: str) -> Optional[np.ndarray]:
        """The first column whose key equals ``name`` (case-insensitive), or None."""
        want = normalise_key(name)
        for j, k in enumerate(self.keys):
            if k == want:
                return self.data[:, j]
        return None

    def take_rows(self, mask: np.ndarray) -> SensorTable:
        return SensorTable(self.keys, self.data[mask])

    @property
    def n_rows(self) -> int:
        return self.data.shape[0]


@dataclass
class SensorChannels:
    """The layout's channels pulled out of a :class:`SensorTable`.

    All arrays are ``(n_samples, n_channels, 2)``, NaN where the column is absent.
    """
    intensity: np.ndarray
    reference: np.ndarray
    led_power: np.ndarray
    reference_led_power: np.ndarray
    missing: list[str]


def normalise_key(name: str) -> str:
    return name.strip().upper().replace("-", "_").replace(" ", "")


def read_sensor_table(path: str | Path, max_rows: Optional[int] = None) -> SensorTable:
    """Read a sensor CSV, optionally only its first ``max_rows`` data rows.

    Short rows are padded with NaN, extra fields are ignored and non-numeric
    cells become NaN, so a recording cut off mid-line still reads. Numbers are
    parsed with correctly rounded conversion, so every value is the double
    ``float()`` would produce; the faster default parser is off by
    one ulp on a few percent of the readings.
    """
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline().rstrip("\r\n").split(",")
    n = len(header)
    frame = pd.read_csv(path, header=None, skiprows=1, names=range(n), usecols=range(n),
                        index_col=False, nrows=max_rows, skipinitialspace=True,
                        float_precision="round_trip", low_memory=False)
    for col in frame.columns[[not pd.api.types.is_numeric_dtype(t) for t in frame.dtypes]]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return SensorTable([normalise_key(h) for h in header], frame.to_numpy(dtype=float))


def parse_sensor(table: SensorTable, layout: pd.DataFrame,
                 wl_tags: tuple[str, str] = WAVELENGTH_TAGS) -> SensorChannels:
    """Pull the layout channels out of a sensor table.

    Every layout channel is read from its RP columns, short channels included;
    LP goes to ``reference``. The LED drive level for a column is the
    ``ledPowerLevel`` column with the same wavelength and panel, which the
    export places just after the intensity pair.
    """
    n, n_ch = table.n_rows, len(layout)
    out = {name: np.full((n, n_ch, 2), np.nan)
           for name in ("intensity", "reference", "led_power", "reference_led_power")}
    missing: list[str] = []

    for i, (s, d, label) in enumerate(zip(layout.S, layout.D, layout.Label)):
        for w, tag in enumerate(wl_tags):
            rp = _find_column(table.keys, f"S{s}_D{d}_{tag}NM_RP")
            lp = _find_column(table.keys, f"S{s}_D{d}_{tag}NM_LP")
            if rp is not None:
                out["intensity"][:, i, w] = table.data[:, rp]
                out["led_power"][:, i, w] = _led_for(table, rp, tag, "RP")
            elif w == 0:
                missing.append(label)
            if lp is not None:
                out["reference"][:, i, w] = table.data[:, lp]
                out["reference_led_power"][:, i, w] = _led_for(table, lp, tag, "LP")
    return SensorChannels(**out, missing=missing)


def all_pairs(table: SensorTable, layout: pd.DataFrame,
              wl_tags: tuple[str, str] = WAVELENGTH_TAGS) -> tuple[np.ndarray, pd.DataFrame]:
    """Every source-detector combination present in the file, used by the montage or not.

    The unused pairs are never illuminated, so they are a direct measurement
    of the detector's zero-light level.

    Returns
    -------
    intensity : ndarray
        ``(n_samples, n_pairs, 2)`` RP intensity, pairs in file order.
    index : DataFrame
        One row per pair: ``S``, ``D`` and ``in_layout``.
    """
    pattern = re.compile(r"^S(\d+)_D(\d+)_(\d+)NM_RP")
    hits = [(j, m) for j, k in enumerate(table.keys) if (m := pattern.match(k))]
    pairs: dict[tuple[int, int], int] = {}
    for _, m in hits:
        pairs.setdefault((int(m.group(1)), int(m.group(2))), len(pairs))

    intensity = np.full((table.n_rows, len(pairs), 2), np.nan)
    for j, m in hits:
        if m.group(3) in wl_tags:
            pair = pairs[(int(m.group(1)), int(m.group(2)))]
            intensity[:, pair, wl_tags.index(m.group(3))] = table.data[:, j]

    used = set(zip(layout.S, layout.D))
    index = pd.DataFrame(list(pairs), columns=["S", "D"])
    index["in_layout"] = [p in used for p in pairs]
    return intensity, index


def _find_column(keys: list[str], name: str) -> Optional[int]:
    try:
        return keys.index(name)
    except ValueError:
        # A reader that de-duplicates names appends a suffix to the repeat.
        return next((j for j, k in enumerate(keys) if k.startswith(name + "_")), None)


def _led_for(table: SensorTable, signal_col: int, tag: str, panel: str) -> np.ndarray:
    want = f"LEDPOWERLEVEL_{tag}NM_{panel}"
    near = range(signal_col, min(signal_col + 4, len(table.keys)))
    col = next((c for c in near if table.keys[c].startswith(want)), None)
    if col is None:
        col = next((c for c, k in enumerate(table.keys) if k.startswith(want)), None)
    return table.data[:, col] if col is not None else np.full(table.n_rows, np.nan)
