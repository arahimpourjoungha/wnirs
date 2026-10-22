"""Collect runs/results/results_*.csv into the README's results tables (markdown).

    python scripts/summarise_results.py runs/results > runs/results/summary.md
"""
import sys
from pathlib import Path

import pandas as pd

METHODS = ["baseline_lr", "scratch", "pretrained_ft", "linear_probe"]
LABEL = {"baseline_lr": "DSP features + logistic regression", "scratch": "encoder, from scratch",
         "pretrained_ft": "encoder, pretrained + fine-tuned", "linear_probe": "encoder, pretrained + linear probe"}
EXPS = [("quality", "channel quality (AUC)"), ("identity", "identity, per recording (acc.)"),
        ("task", "task state, per recording (acc.)")]


def cell(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "–"
    return f"{s.mean():.3f}" if len(s) == 1 else f"{s.mean():.3f} ± {s.std():.3f}"


def main(folder: Path) -> None:
    d = pd.concat([pd.read_csv(f) for f in sorted(folder.glob("results_*.csv"))], ignore_index=True)
    d["score"] = d["auc"].fillna(d["recording_acc"]) if "auc" in d else d["recording_acc"]
    main_ = d[d.variant == "default"]
    print("| method | " + " | ".join(t for _, t in EXPS) + " |")
    print("|---|" + "---|" * len(EXPS))
    for m in METHODS:
        row = [cell(main_[(main_.method == m) & (main_.experiment == e)].score) for e, _ in EXPS]
        print(f"| {LABEL[m]} | " + " | ".join(row) + " |")
    print()
    print("| preprocessing | method | " + " | ".join(t for _, t in EXPS) + " |")
    print("|---|---|" + "---|" * len(EXPS))
    for v in ["default", "no_ssr", "no_motion"]:
        for m in ["baseline_lr", "pretrained_ft"]:
            q = d[(d.variant == v) & (d.method == m) & (d.seed == 0)]
            if q.empty:
                continue
            row = [cell(q[q.experiment == e].score) for e, _ in EXPS]
            print(f"| {v} | {LABEL[m]} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/results"))
