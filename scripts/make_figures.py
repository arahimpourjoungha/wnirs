"""Figures for the README, all built from synthetic data.

    python scripts/make_figures.py

Writes docs/figures/. The JAX benchmark figure is drawn from the output of
scripts/bench_glm.py saved to runs/bench/bench_glm.txt, when present.
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "figures"

C_GOOD, C_BAD, C_ACC1, C_ACC2 = "#2b6cb0", "#c53030", "#dd6b20", "#2f855a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 150})


def panel_label(ax, s):
    ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


# ---------------------------------------------------------------- synthetic
def fig_pipeline() -> None:
    """Raw light -> motion repair -> haemodynamic response -> channel map, on a synthetic recording."""
    from tests.synthetic import motion_signals, write_tablet_recording
    from wnirs.analysis import block_average, contrast, glm
    from wnirs.io import find_sessions, read_brain_games
    from wnirs.preproc import preprocess, tddr
    from wnirs.qc import channel_qc

    with tempfile.TemporaryDirectory() as tmp:
        rec = write_tablet_recording(Path(tmp), "Flanker", duration=300.0)
        raw = read_brain_games(find_sessions(tmp).File[0])
    hb = preprocess(raw)
    qc = channel_qc(raw, hb)
    fit = glm(hb)
    con = contrast(fit, "incongruent - congruent")
    ba = block_average(hb, pre=5, post=15)

    t = raw.time - raw.time[0]
    good = np.flatnonzero(qc.Pass.to_numpy() & (raw.layout.Type.to_numpy() == 1))
    hbo_t = con[con.Chromophore.str.lower() == "hbo"].set_index("Channel")["Tstat"]
    best = int(hbo_t.loc[good + 1].idxmax()) - 1

    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    for ch, col, lab in ((best, C_GOOD, "good channel"),
                         (rec.roles["saturated"], C_ACC1, "clipped at the detector ceiling"),
                         (rec.roles["dark"], "0.35", "no light (optode not coupled)")):
        ax.plot(t, raw.intensity[:, ch, 1], color=col, lw=0.7, label=lab)
    ax.set(xlabel="time (s)", ylabel="detector reading (V), 860 nm", title="Raw light: three channels QC must tell apart")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    fs_m = 10.0
    x, clean = motion_signals(fs=fs_m)
    tm = np.arange(len(x)) / fs_m
    fixed = tddr(x[:, 3], fs_m)
    ax.plot(tm, x[:, 3], color=C_BAD, lw=0.8, label="with spikes and baseline shifts")
    ax.plot(tm, fixed - np.median(fixed - clean[:, 3]), color="k", lw=0.9, label="after TDDR")
    ax.plot(tm, clean[:, 3], color=C_ACC2, lw=0.8, ls="--", label="ground truth")
    ax.set(xlabel="time (s)", ylabel="ΔOD", title="Motion repair (TDDR) against known truth")
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    for cond, col in zip(ba.conditions, (C_ACC1, C_GOOD)):
        y, e = cond.hbo[:, best], cond.sem_hbo[:, best]
        ax.plot(cond.time, y, color=col, lw=1.6, label=f"{cond.name} (n={int(cond.n_trials[best])})")
        ax.fill_between(cond.time, y - e, y + e, color=col, alpha=0.2, lw=0)
    ax.axvspan(0, ba.conditions[0].duration, color="0.92", zorder=0)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set(xlabel="time from trial onset (s)", ylabel="HbO (µM)",
           title=f"Block-averaged response, channel {raw.layout.Label[best]}")
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    lay = raw.layout
    tvals = hbo_t.to_numpy()
    lim = np.nanmax(np.abs(tvals[lay.Type.to_numpy() == 1]))
    for i, r in lay.iterrows():
        if r.Type != 1:
            continue
        ok = bool(qc.Pass.iloc[i])
        colour = plt.cm.RdBu_r((tvals[i] / lim + 1) / 2) if ok and np.isfinite(tvals[i]) else "white"
        ax.scatter(r.Column, -r.Row, s=150, c=[colour], edgecolors="k" if ok else C_BAD,
                   linewidths=0.6 if ok else 1.8, marker="s" if ok else "X")
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-lim, lim))
    fig.colorbar(sm, ax=ax, shrink=0.8, label="t, incongruent − congruent (HbO)")
    ax.set(title="Channel map: GLM contrast, QC failures marked ✕", xticks=[], yticks=[],
           xlim=(lay.Column.min() - 1, lay.Column.max() + 1), ylim=(-lay.Row.max() - 0.7, -lay.Row.min() + 0.7))
    ax.spines[:].set_visible(False)
    panel_label(ax, "d")

    fig.suptitle("One synthetic Flanker recording through the pipeline", fontsize=12, fontweight="bold")
    fig.savefig(OUT / "pipeline_synthetic.png", bbox_inches="tight")
    plt.close(fig)


def fig_streaming() -> None:
    """Streaming kernel vs PyTorch on a synthetic window: float is exact, int8 is close."""
    import torch
    from wnirs.ml.models import WNIRSNet
    from wnirs.ml.streaming import StreamingEncoder, calibrate_activation_scales
    from wnirs.ml.train import TrainConfig, pretrain, seed_everything
    from wnirs.ml.data import windows_from_arrays, WindowSet

    seed_everything(0)
    rng = np.random.default_rng(0)
    sets = []
    for k in range(12):
        n, tt = 1500, np.arange(1500) / 7.1
        hbo = rng.normal(0, 0.3, (n, 31)) + np.sin(2 * np.pi * 0.04 * tt)[:, None]
        sets.append(windows_from_arrays(hbo, -0.4 * hbo + rng.normal(0, 0.2, (n, 31)), np.ones(31, bool), 7.1,
                                        meta=dict(subject="S", session=str(k), task="T", recording=str(k))))
    ws = WindowSet.concat(sets)
    model = WNIRSNet(62)
    pretrain(model, ws, TrainConfig(epochs=2))
    model.eval()
    x, m = ws.x[0], ws.mask[0]
    with torch.no_grad():
        ref = model(torch.from_numpy(x[None]), torch.from_numpy(m[None]))["quality"][0].numpy()
    scales = calibrate_activation_scales(model, ws.x[:64], ws.mask[:64])
    out = {}
    for name, s in (("float", StreamingEncoder(model, m)),
                    ("int8", StreamingEncoder(model, m, int8=True, act_scales=scales))):
        trace = np.array([s.step(x[:, i]) for i in range(x.shape[1])])
        out[name] = (trace, s.memory_bytes())

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axs[0]
    lo, hi = ref.min(), ref.max()
    ax.plot([lo, hi], [lo, hi], color="0.7", lw=0.8)
    ax.scatter(ref, out["float"][0][-1], s=14, color=C_GOOD, label="streaming float32")
    ax.scatter(ref, out["int8"][0][-1], s=14, color=C_ACC1, label="streaming int8", marker="^")
    ax.set(xlabel="PyTorch (whole window)", ylabel="streaming kernel (one sample at a time)",
           title="Quality logits, 62 signals")
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, "a")
    ax = axs[1]
    labels = ["weights", "ring-buffer state"]
    f32 = [out["float"][1]["weights"] / 1024, out["float"][1]["state"] / 1024]
    i8 = [out["int8"][1]["weights"] / 1024, out["int8"][1]["state"] / 1024]
    xx = np.arange(2)
    ax.bar(xx - 0.18, f32, 0.36, color=C_GOOD, label="float32")
    ax.bar(xx + 0.18, i8, 0.36, color=C_ACC1, label="int8")
    for xi, (a, b) in enumerate(zip(f32, i8)):
        ax.text(xi - 0.18, a, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.18, b, f"{b:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set(xticks=xx, xticklabels=labels, ylabel="KB", yscale="log", title="On-device memory")
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, "b")
    fig.tight_layout()
    fig.savefig(OUT / "streaming_kernel.png", bbox_inches="tight")
    plt.close(fig)


def fig_jax(bench_txt: Path) -> None:
    txt = bench_txt.read_text()
    t_np = float(re.search(r"NumPy loop\s+([\d.]+)", txt).group(1))
    t_jx = float(re.search(r"JAX jit \+ vmap\s+([\d.]+)", txt).group(1))
    head = txt.splitlines()[0]
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.barh(["NumPy channel loop", "JAX jit + vmap\n+ shard_map"], [t_np, t_jx], color=["0.6", C_GOOD])
    for y, v in enumerate([t_np, t_jx]):
        ax.text(v, y, f" {v:.1f} s", va="center")
    ax.set(xlabel="wall time (s)", title=f"AR-IRLS GLM, {t_np / t_jx:.1f}× faster, identical results")
    ax.text(0, -0.55, head.split(";")[0], transform=ax.get_xaxis_transform(), fontsize=7, color="0.4")
    ax.set_xlim(0, t_np * 1.2)
    fig.tight_layout()
    fig.savefig(OUT / "jax_glm_benchmark.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    warnings.simplefilter("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_pipeline()
    fig_streaming()
    bench = ROOT / "runs" / "bench" / "bench_glm.txt"
    if bench.exists():
        fig_jax(bench)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
