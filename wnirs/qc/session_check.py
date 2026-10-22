"""A seconds-long check, run right after a session, that the optics were working."""
from __future__ import annotations

import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..io.layout import load_layout
from ..io.names import parse_brain_games_name, parse_name
from ..io.sensor import parse_sensor, read_sensor_table


@dataclass(frozen=True)
class SessionCheckOptions:
    """Levels for :func:`session_check`, in the file's units.

    dark : detector reading with no light.
    ceiling : detector maximum.
    margin : how far above ``dark`` counts as light.
    n_samples : rows read per file (500 is about 70 s at 7 Hz).
    min_long_ok : fraction of long channels that must carry light to pass.
    """
    dark: float = 1.514
    ceiling: float = 5.0
    margin: float = 0.02
    n_samples: int = 500
    min_long_ok: float = 0.5


def session_check(target: str | Path, layout: str | Path | pd.DataFrame = "full",
                  options: SessionCheckOptions = SessionCheckOptions()
                  ) -> tuple[bool, pd.DataFrame]:
    """Did enough light reach the detector for these recordings to be worth analysing?

    Reads only the first minute or so of each file, so a whole session checks
    in about the time it takes to take the headband off.

    It exists because on this study the long channels sat at the detector's
    dark floor while the short channels were pinned at its ceiling (one LED
    drive level cannot serve an 8 mm and a 30 mm separation at once), and
    nothing flagged it until the analysis, weeks later. The two faults need
    opposite fixes, so they are reported separately: dark long channels mean
    RAISE the drive, saturated short channels mean LOWER it.

    The verdict is about light, not about the brain: failing means do not
    bother analysing, passing means the optics were working.

    Parameters
    ----------
    target : str or Path
        A sensor CSV, or a folder searched recursively for ``*SensorData*.csv``
        and ``dataLog*.csv``.
    layout : str, Path or DataFrame
        See :func:`wnirs.io.load_layout`.
    options : SessionCheckOptions

    Returns
    -------
    ok : bool
        Every recording passed.
    table : DataFrame
        ``File, Task, nLong, nLongDark, nShort, nShortSaturated, MedianLong,
        MedianShort, Verdict`` (PASS / MARGINAL / FAIL) and ``Advice``.
    """
    files = _gather(Path(target))
    if not files:
        raise FileNotFoundError(f"No sensor recordings under {target}. Point this at a session "
                                "folder or a CSV.")
    layout = load_layout(layout)
    is_long = layout.Type.to_numpy() == 1
    rows = [r for f in files if (r := _check_one(f, layout, is_long, options)) is not None]
    if not rows:
        raise ValueError(f"None of the {len(files)} file(s) could be read.")
    table = pd.DataFrame(rows)
    return bool((table.Verdict == "PASS").all()), table


def session_check_report(table: pd.DataFrame,
                         options: SessionCheckOptions = SessionCheckOptions()) -> str:
    """The check as printable text, with each kind of fault's advice given once."""
    lines = ["Session check: did enough light reach the detector?",
             f"floor {options.dark:.3f}   ceiling {options.ceiling:g}   (a long channel needs to "
             f"clear {options.dark + options.margin:.3f})", "",
             f"{'task':<10} {'long dark':>16} {'short saturated':>16}   verdict", "-" * 72]
    for r in table.itertuples():
        lines.append(f"{r.Task:<10} {r.nLongDark:>8d} of {r.nLong:<5d} "
                     f"{r.nShortSaturated:>8d} of {r.nShort:<5d}   {r.Verdict}")
    bad = table[table.Verdict != "PASS"]
    if not bad.empty:
        lines += ["", "What to do"]
        seen = set()
        for advice in bad.Advice:
            kind = _advice_kind(advice)
            if kind not in seen:
                seen.add(kind)
                lines.append(textwrap.fill(advice, 74, initial_indent=" - ",
                                           subsequent_indent="   "))
    if bad.empty:
        lines += ["", "PASS: the optics were working. Keep the session."]
    elif (table.Verdict == "FAIL").any():
        lines += ["", f"FAIL: {int((table.Verdict == 'FAIL').sum())} recording(s) have almost no "
                  "light on the long channels. Nothing downstream can recover this. Fix the "
                  "drive and record again."]
    else:
        lines += ["", "MARGINAL: usable, but see above before the next session."]
    return "\n".join(lines)


def _check_one(file: Path, layout: pd.DataFrame, is_long: np.ndarray,
               opt: SessionCheckOptions) -> dict | None:
    try:
        table = read_sensor_table(file, max_rows=opt.n_samples)
    except (OSError, ValueError, pd.errors.ParserError) as err:
        warnings.warn(f"{file.stem} could not be read ({err}).", stacklevel=3)
        return None
    if table.n_rows == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an absent channel's median is NaN
        level = np.nanmin(np.nanmedian(parse_sensor(table, layout).intensity, axis=0), axis=1)

    floor, ceiling = opt.dark + opt.margin, opt.ceiling - 0.1
    long_dark = is_long & (level < floor)
    short_sat = ~is_long & (level > ceiling)
    long_sat = is_long & (level > ceiling)
    n_long, n_short = int(is_long.sum()), int((~is_long).sum())
    frac_dark = long_dark.sum() / max(n_long, 1)

    # Ordered by cost: a dark long channel is lost signal; a saturated short
    # one only costs short-separation regression, which has alternatives.
    if frac_dark > 1 - opt.min_long_ok:
        verdict, advice = "FAIL", (
            f"{100 * frac_dark:.0f}% of long channels are at the detector floor: almost no light "
            "is crossing 30 mm. RAISE the long-channel LED drive, or reseat the headband through "
            "the hair, and record again.")
    elif frac_dark > 0.25:
        verdict, advice = "MARGINAL", (
            f"{100 * frac_dark:.0f}% of long channels are dark. Usable, but check the fit before "
            "continuing: this is how a study ends up with a third of its channels missing.")
    elif long_sat.any():
        verdict, advice = "MARGINAL", (
            f"{int(long_sat.sum())} long channel(s) are clipped at the ceiling: LOWER their LED "
            "drive.")
    elif n_short > 0 and short_sat[~is_long].all():
        verdict, advice = "MARGINAL", (
            "Every short channel is pinned at the ceiling, so short-separation regression cannot "
            "run. LOWER the short-channel LED drive. The long channels are fine, so the "
            "recording is still analysable.")
    else:
        verdict, advice = "PASS", ""

    task = parse_brain_games_name(file.stem).task or parse_name(file.stem, "task")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_long, median_short = np.nanmedian(level[is_long]), np.nanmedian(level[~is_long])
    return {"File": file.stem, "Task": task, "nLong": n_long, "nLongDark": int(long_dark.sum()),
            "nShort": n_short, "nShortSaturated": int(short_sat.sum()),
            "MedianLong": median_long, "MedianShort": median_short,
            "Verdict": verdict, "Advice": advice}


def _gather(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        return []
    found = [*target.rglob("*SensorData*.csv"), *target.rglob("dataLog*.csv")]
    return sorted(f for f in found if not f.name.startswith((".", "~$")))


def _advice_kind(advice: str) -> str:
    """Which fault a piece of advice is about, ignoring the numbers in it."""
    a = advice.lower()
    if "raise" in a:
        return "long-dark"
    if "long channel(s) are clipped" in a:
        return "long-saturated"
    if "short-channel led" in a:
        return "short-saturated"
    if "check the fit" in a:
        return "long-marginal"
    return a
