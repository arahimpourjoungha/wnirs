"""End-to-end example on a synthetic recording: read, QC, preprocess, fit, connect.

    python examples/quickstart.py

Writes a 5-minute Flanker recording in the headband's file format (with a
known haemodynamic response, a head movement and four faulty channels) to a
temporary folder, then runs the standard pipeline on it.
"""
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # for tests.synthetic

from tests.synthetic import write_tablet_recording
from wnirs.analysis import connectivity, contrast, default_contrasts, glm
from wnirs.io import find_sessions, read_brain_games
from wnirs.preproc import preprocess
from wnirs.qc import channel_qc

warnings.simplefilter("ignore")

with tempfile.TemporaryDirectory() as tmp:
    write_tablet_recording(Path(tmp), "Flanker", duration=300.0)
    sessions = find_sessions(tmp)
    raw = read_brain_games(sessions.File[0])
    print(f"read {raw.task}: {len(raw.time)} samples at {raw.fs:.2f} Hz, "
          f"{raw.n_channels} channels, conditions {raw.conditions.name}")

    qc = channel_qc(raw)
    print(f"QC: {qc.Pass.sum()} of {len(qc)} channels pass; failures: "
          f"{qc.loc[~qc.Pass, 'Reason'].value_counts().to_dict()}")

    hb = preprocess(raw)
    print("preprocessing:", " -> ".join(hb.steps))

    fit = glm(hb)                                    # AR-IRLS, canonical HRF
    print(f"GLM regressors: {fit.names[:fit.n_condition_regressors]}")
    for spec in default_contrasts(hb.conditions.name):
        c = contrast(fit, spec)
        hbo = c[c.Chromophore.str.lower() == "hbo"].dropna(subset=["Tstat"])
        top = hbo.nlargest(3, "Tstat")[["Label", "Beta", "Tstat", "Qval"]]
        print(f"contrast {spec!r}: strongest HbO channels\n{top.round(3).to_string(index=False)}")

    conn = connectivity(hb)
    print(f"connectivity: {conn.Z.shape[0]} long channels, mean r = {conn.mean_r:.2f} (AR-prewhitened)")
