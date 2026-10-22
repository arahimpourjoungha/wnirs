"""Run the learning experiments on cached windows and write a results table.

    python scripts/run_experiments.py runs/windows_default.npz --out runs/results
    python scripts/run_experiments.py ... --only task --seeds 0 1 2

Experiments (all splits hold out each person's *latest* sessions):

  task        which of Flanker / Go-NoGo / N-Back / Rest a recording is
  identity    which person a recording came from ("fingerprinting")
  quality     whether a channel passes the QC gates, per channel and window

Each is run for:
  baseline_lr     logistic regression on hand-crafted DSP + connectivity features
  scratch         the encoder trained end-to-end from random init
  pretrained_ft   masked-reconstruction pretraining, then fine-tuning
  linear_probe    pretraining, then only the head is trained
and over several seeds. Self-supervised pretraining only ever sees the
training sessions; the test sessions are untouched until scoring.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wnirs.ml.data import TASKS, WindowSet, label_codes, session_split
from wnirs.ml.features import spectral_features, window_features
from wnirs.ml.models import WNIRSNet, count_macs, count_params
from wnirs.ml.train import (TrainConfig, finetune, finetune_quality, inner_val_split,
                            predict, pretrain, quality_auc, seed_everything, session_vote)


def analysis_view(ws: WindowSet) -> WindowSet:
    """For task/identity: only channels that have data AND pass QC are visible."""
    good = ws.mask & (np.nan_to_num(ws.quality, nan=1.0) > 0.5)
    return WindowSet(ws.x * good[..., None], good, ws.quality, ws.meta, ws.fs)


def labels(ws: WindowSet, target: str) -> tuple[np.ndarray, list]:
    if target == "task":
        return label_codes(ws.meta["task"], TASKS)
    return label_codes(ws.meta["subject"].astype(str))


def score(pred: np.ndarray, y: np.ndarray, ws: WindowSet) -> dict:
    rec = ws.meta["recording"].to_numpy()
    keys, vote = session_vote(pred, rec)
    truth = np.array([y[rec == k][0] for k in keys])
    return {"window_acc": float((pred == y).mean()), "recording_acc": float((vote == truth).mean()),
            "n_test_windows": int(len(y)), "n_test_recordings": int(len(keys))}


_PRETRAINED: dict[tuple, dict] = {}


def pretrained_encoder(ws: WindowSet, seed: int, epochs: int, cache: Path | None) -> dict:
    """Pretrain once per seed on the training sessions and reuse it for every experiment.

    Valid because every experiment uses the same session split (each person's
    latest sessions held out), so the pretraining data never include a test session.
    """
    key = (seed, epochs)
    if key in _PRETRAINED:
        return _PRETRAINED[key]
    path = cache / f"pretrained_s{seed}_e{epochs}.pt" if cache else None
    if path is not None and path.exists():
        _PRETRAINED[key] = torch.load(path)
        return _PRETRAINED[key]
    seed_everything(seed)
    tr, _ = session_split(ws.meta)
    model = WNIRSNet(ws.x.shape[1])
    losses = pretrain(model, ws.subset(tr), TrainConfig(epochs=epochs))
    state = {k: v for k, v in model.state_dict().items() if k.startswith("encoder.")}
    _PRETRAINED[key] = state
    if path is not None:
        torch.save(state, path)
        path.with_suffix(".loss.json").write_text(json.dumps(losses))
    return state


def new_model(ws: WindowSet, n_classes: int, method: str, seed: int, pre_epochs: int,
              cache: Path | None) -> WNIRSNet:
    seed_everything(seed)
    model = WNIRSNet(ws.x.shape[1], n_classes=n_classes)
    if method in ("pretrained_ft", "linear_probe"):
        model.load_state_dict(pretrained_encoder(ws, seed, pre_epochs, cache), strict=False)
    return model


def run_classification(ws: WindowSet, target: str, method: str, seed: int, epochs: int,
                       pre_epochs: int, cache: Path | None = None) -> dict:
    full = ws
    ws = analysis_view(ws)
    y, classes = labels(ws, target)
    tr, te = session_split(ws.meta)
    if method == "baseline_lr":
        f = window_features(ws.x, ws.mask, ws.fs)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        clf.fit(f[tr], y[tr])
        return score(clf.predict(f[te]), y[te], ws.subset(te))
    tr_in, va = inner_val_split(ws.meta, tr, seed=seed)
    model = new_model(full, len(classes), method, seed, pre_epochs, cache)
    cfg = TrainConfig(epochs=epochs, freeze_encoder=(method == "linear_probe"), patience=5, batch_size=64)
    finetune(model, ws.subset(tr_in), y[tr_in], ws.subset(va), y[va], cfg)
    return score(predict(model, ws.subset(te)), y[te], ws.subset(te))


def run_quality(ws: WindowSet, method: str, seed: int, epochs: int, pre_epochs: int,
                cache: Path | None = None) -> dict:
    seed_everything(seed)
    tr, te = session_split(ws.meta)
    test = ws.subset(te)
    lab_te = test.mask & ~np.isnan(test.quality)
    if method == "baseline_lr":
        f = spectral_features(ws.x, ws.fs)
        lab = ws.mask & ~np.isnan(ws.quality)
        idx_tr = np.zeros_like(lab)
        idx_tr[tr] = True
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        clf.fit(f[lab & idx_tr], ws.quality[lab & idx_tr] > 0.5)
        s = clf.decision_function(spectral_features(test.x, ws.fs)[lab_te])
        return {"auc": float(roc_auc_score(test.quality[lab_te] > 0.5, s))}
    tr_in, va = inner_val_split(ws.meta, tr, seed=seed)
    model = new_model(ws, 4, method, seed, pre_epochs, cache)
    finetune_quality(model, ws.subset(tr_in), ws.subset(va),
                     TrainConfig(epochs=epochs, freeze_encoder=(method == "linear_probe"),
                                 patience=5, batch_size=64))
    return {"auc": quality_auc(model, test)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("windows", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/results"))
    ap.add_argument("--only", nargs="*", default=["task", "identity", "quality"])
    ap.add_argument("--methods", nargs="*", default=["baseline_lr", "scratch", "pretrained_ft", "linear_probe"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--pre-epochs", type=int, default=15)
    ap.add_argument("--tag", default="default", help="label for this window set (preprocessing variant)")
    a = ap.parse_args()

    ws = WindowSet.load(a.windows)
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / f"pretrained_{a.tag}"
    cache.mkdir(exist_ok=True)
    torch.backends.nnpack.set_flags(False)       # slower than the default kernels for these shapes
    rows = []
    for exp in a.only:
        for method in a.methods:
            seeds = a.seeds[:1] if method == "baseline_lr" else a.seeds   # LR is deterministic
            for seed in seeds:
                t0 = time.time()
                if exp == "quality":
                    r = run_quality(ws, method, seed, a.epochs, a.pre_epochs, cache)
                else:
                    r = run_classification(ws, exp, method, seed, a.epochs, a.pre_epochs, cache)
                rows.append({"variant": a.tag, "experiment": exp, "method": method, "seed": seed,
                             "seconds": round(time.time() - t0, 1), **r})
                print(json.dumps(rows[-1]), flush=True)
                pd.DataFrame(rows).to_csv(a.out / f"results_{a.tag}.csv", index=False)

    model = WNIRSNet(ws.x.shape[1])
    (a.out / f"model_size_{a.tag}.json").write_text(json.dumps({
        "params": count_params(model), "macs_per_window": count_macs(model, ws.x.shape[1], ws.x.shape[2]),
        "n_signals": int(ws.x.shape[1]), "window_samples": int(ws.x.shape[2]), "fs": ws.fs}, indent=2))


if __name__ == "__main__":
    main()
