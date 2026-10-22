"""Benchmark the AR-IRLS GLM: NumPy channel loop vs. jitted, vmapped JAX.

Synthetic dense-sampling data: every session is a 5 Hz, 300 s recording with
its own randomised two-condition block design and AR(1) noise (rho 0.9) on
every channel. Both implementations fit the same data; the script reports
wall time per implementation, the speedup, and the largest disagreement in
the betas and t statistics.

    python scripts/bench_glm.py                       # 60 channels x 200 sessions
    python scripts/bench_glm.py --sessions 20 --numpy-sessions 5 --devices 1

On CPU, XLA exposes a single device unless told otherwise; ``--devices`` splits
the cores into that many host devices so independent sessions run on separate
cores (``shard_map``) while each session's channels are vmapped. The NumPy
reference is the single-threaded channel loop the GLM runs in practice.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy.signal import lfilter

from wnirs.analysis import design_matrix, fit_channels
from wnirs.types import Conditions


def make_sessions(n_sessions: int, n_channels: int, fs: float, duration: float, seed: int):
    """Designs (S, T, p) and data (S, T, C) with a true response on half the channels."""
    rng = np.random.default_rng(seed)
    T = int(round(duration * fs))
    t = np.arange(T) / fs
    Xs, Ys = [], []
    for _ in range(n_sessions):
        starts = np.sort(rng.choice(np.arange(15.0, duration - 45.0, 30.0), 8, replace=False))
        labels = rng.permutation(np.repeat([0, 1], 4))
        cond = Conditions(["easy", "hard"], [starts[labels == 0], starts[labels == 1]],
                          [np.full(4, 20.0), np.full(4, 20.0)])
        X = design_matrix(t, cond, fs).X
        beta = np.zeros((X.shape[1], n_channels))
        beta[1, : n_channels // 2] = rng.uniform(0.2, 1.0, n_channels // 2)
        noise = lfilter([1.0], [1.0, -0.9], 0.3 * rng.standard_normal((T + 100, n_channels)), axis=0)
        Xs.append(X)
        Ys.append(X @ beta + noise[100:])
    return np.stack(Xs), np.stack(Ys)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", type=int, default=60)
    ap.add_argument("--sessions", type=int, default=200)
    ap.add_argument("--numpy-sessions", type=int, default=None,
                    help="time NumPy on this many sessions and scale up (default: all)")
    ap.add_argument("--devices", type=int, default=4,
                    help="XLA host devices on CPU (default 4, the performance cores of an M-series laptop)")
    ap.add_argument("--batch", type=int, default=None, help="sessions per compiled call (default: one per device)")
    ap.add_argument("--fs", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # XLA reads the host device count when JAX initialises, so it is set
    # before anything imports jax.
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + f" --xla_force_host_platform_device_count={args.devices}").strip()
    import jax

    from wnirs.analysis.glm_jax import ar_irls_jax

    X, Y = make_sessions(args.sessions, args.channels, args.fs, args.duration, args.seed)
    S, T, p = X.shape
    print(f"{S} sessions x {args.channels} channels, T = {T} samples, {p} regressors, AR order searched up to "
          f"{round(4 * args.fs)}; {os.cpu_count()} CPU cores, JAX devices: {jax.local_device_count()}")

    n_np = S if args.numpy_sessions is None else min(args.numpy_sessions, S)
    t0 = time.perf_counter()
    ref = [fit_channels(X[s], Y[s], fs=args.fs) for s in range(n_np)]
    t_np = (time.perf_counter() - t0) * S / n_np
    scaled = "" if n_np == S else f" (measured on {n_np} sessions, scaled)"
    print(f"NumPy loop      {t_np:8.2f} s{scaled}")

    warm = args.batch or args.devices
    t0 = time.perf_counter()
    ar_irls_jax(X[:warm], Y[:warm], fs=args.fs, session_batch=args.batch)
    print(f"JAX first call  {time.perf_counter() - t0:8.2f} s  (compilation + {warm} sessions)")
    t0 = time.perf_counter()
    fits = ar_irls_jax(X, Y, fs=args.fs, session_batch=args.batch)
    t_jax = time.perf_counter() - t0
    print(f"JAX jit + vmap  {t_jax:8.2f} s")
    print(f"speedup         {t_np / t_jax:8.1f} x")

    beta = np.stack([r.beta for r in ref])
    tstat = np.stack([r.tstat for r in ref])
    orders = np.stack([r.ar_order for r in ref])
    print(f"max |beta diff| {np.max(np.abs(fits.beta[:n_np] - beta)):.1e}, "
          f"max |t diff| {np.max(np.abs(fits.tstat[:n_np] - tstat)):.1e}, "
          f"AR orders identical: {np.array_equal(fits.ar_order[:n_np], orders)}")


if __name__ == "__main__":
    main()
