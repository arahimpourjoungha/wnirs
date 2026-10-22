#!/usr/bin/env bash
# Full experiment grid. Main comparison on the default preprocessing, then
# one-step preprocessing ablations (no short-separation regression, no motion
# correction) with the strongest learned method against the classical baseline.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/run_experiments.py runs/windows_default.npz --out runs/results --tag default --seeds 0 1 2
for v in no_ssr no_motion; do
  $PY scripts/run_experiments.py runs/windows_$v.npz --out runs/results --tag $v --seeds 0 1 2 \
      --methods baseline_lr pretrained_ft
done
