"""Read, QC and preprocess every recording once, and cache the learning windows.

    python scripts/build_windows.py /path/to/data/Data --out runs/windows_default.npz
    python scripts/build_windows.py /path/to/data/Data --variant no_ssr --out runs/windows_no_ssr.npz

Each recording goes through the same steps as the activation analysis:
Brain Games reader (with over-long recordings trimmed),
the default preprocessing, and the four-gate channel QC. Windows keep every
long channel that has data; the QC verdict travels with them in
``WindowSet.quality`` so the quality experiment can learn it and the other
experiments can hide failing channels.

Variants change exactly one preprocessing step, so their results can be
compared as controlled ablations.
"""
from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wnirs.io import find_sessions, read_brain_games
from wnirs.ml.data import WindowSet, windows_from_arrays
from wnirs.preproc import preprocess
from wnirs.qc import channel_qc

VARIANTS = {
    "default": {},
    "no_ssr": {"ssr": "none"},
    "no_motion": {"motion": "none"},
    "no_filter": {"filter": False},
}


def one_recording(args: tuple) -> WindowSet | None:
    row, variant, win_s, hop_s = args
    warnings.simplefilter("ignore")
    try:
        raw = read_brain_games(row["File"])
        hb = preprocess(raw, **VARIANTS[variant])
        qc = channel_qc(raw, hb)
    except Exception as e:                          # one unreadable file must not stop the build
        print(f"skip {Path(row['File']).name}: {e}")
        return None
    long = hb.is_long
    has_data = hb.usable[long]
    passed = qc["Pass"].to_numpy(bool)[long].astype(np.float32)
    meta = {"subject": str(row["Subject"]), "session": str(row["Session"]), "task": row["Task"],
            "recording": f"{row['Subject']}_{row['Session']}_{row['Task']}"}
    return windows_from_arrays(hb.hbo[:, long], hb.hbr[:, long], has_data, hb.fs,
                               quality=passed, meta=meta, win_s=win_s, hop_s=hop_s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", choices=VARIANTS, default="default")
    ap.add_argument("--win", type=float, default=30.0)
    ap.add_argument("--hop", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    sessions = find_sessions(a.data_root)
    jobs = [(r, a.variant, a.win, a.hop) for r in sessions.to_dict("records")]
    with ProcessPoolExecutor(a.workers) as ex:
        sets = [s for s in ex.map(one_recording, jobs) if s is not None and len(s)]
    ws = WindowSet.concat(sets)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ws.save(a.out)
    m = ws.meta
    print(f"{len(sets)} recordings, {len(ws)} windows, {ws.x.shape[1]} signals x {ws.x.shape[2]} samples; "
          f"{m.subject.nunique()} subjects, {m.groupby('subject').session.nunique().to_dict()} sessions; "
          f"QC pass rate {np.nanmean(ws.quality[ws.mask]):.2f}")


if __name__ == "__main__":
    main()
