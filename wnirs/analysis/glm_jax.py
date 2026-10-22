"""AR-IRLS GLM in JAX, compiled once and vectorised over channels and sessions.

The NumPy estimator (:func:`wnirs.analysis.ar_irls`) is a loop over channels
with data-dependent control flow at three levels: the AR order chosen by BIC,
the robust IRLS iterations, and the outer whiten-refit iterations. Each is
rewritten here as fixed-shape computation so the whole fit is one jitted
function that ``vmap`` can batch:

* the Levinson-Durbin recursion runs over the full order grid 1..pmax with a
  ``lax.scan``; orders past the first unstable reflection coefficient are
  masked out of the BIC search, and the chosen filter is zero-padded to
  pmax + 1 taps, so filtering with it is the same FIR whatever the order;
* whitening is a contraction of a precomputed stack of lagged copies of the
  design (shared by all channels of a session) with each channel's taps;
* the filter start-up samples, whose number depends on the chosen order, are
  excluded with a row mask instead of slicing: masked rows are zeroed in the
  design, carry zero leverage and zero weight, and the MAD scale is the median
  of the valid residuals, read from a sorted vector at a traced index (the
  sort runs on the integer bit patterns, which order non-negative floats
  identically and sort several times faster);
* both iteration loops are ``lax.while_loop`` with the NumPy convergence
  rules. Under ``vmap`` a loop runs until every lane has converged, with
  converged lanes held fixed, so each channel still gets exactly its own
  number of iterations;
* sessions are independent, so with several devices each call's sessions are
  split across them with ``shard_map`` and no communication.

Results match the NumPy implementation to rounding error (float64 is enabled
for the duration of each call). Channels with any non-finite sample, or a
constant one, come back NaN, as in the NumPy GLM.
"""
from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P

from ._stats import t_pvalue
from .arirls import default_pmax
from .glm import ChannelFits
from .robust import BISQUARE_TUNE

_EPS = float(np.finfo(np.float64).eps)
_SQRT_EPS = float(np.sqrt(_EPS))


def ar_irls_jax(X: np.ndarray, Y: np.ndarray, *, fs: float | None = None, pmax: int | None = None,
                tune: float = BISQUARE_TUNE, max_iter: int = 10, irls_max_iter: int = 50,
                session_batch: int | None = None) -> ChannelFits:
    """AR-IRLS fit of every channel (and session) at once.

    Channels of a session are vmapped over one shared design. Sessions are
    vmapped too, and when JAX sees several devices (GPUs, or CPU cores exposed
    with ``XLA_FLAGS=--xla_force_host_platform_device_count=N``) each call's
    sessions are split across them with ``shard_map``.

    Parameters
    ----------
    X : (T, p) or (S, T, p) array
        Design matrix, or one per session. Sessions are batched, so they must
        share T and p; group recordings by length to batch unequal ones.
    Y : (T, C) or (S, T, C) array
        Channels to fit with that design.
    fs, pmax, tune, max_iter
        As for :func:`wnirs.analysis.ar_irls`.
    irls_max_iter : int
        Iteration limit of the inner robust fit.
    session_batch : int, optional
        Sessions per compiled call, rounded up to a multiple of the device
        count. Every lane of a call iterates until its slowest lane has
        converged, so small batches waste fewer iterations; the default, one
        session per device, was fastest on CPU.

    Returns
    -------
    ChannelFits
        Same fields and layout as :func:`wnirs.analysis.fit_channels`, with a
        leading session axis when ``X`` is 3-D.
    """
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    batched = X.ndim == 3
    if not batched:
        X, Y = X[None], Y[None]
    S, T, p = X.shape
    pmax = default_pmax(fs) if pmax is None else int(pmax)
    pmax = max(0, min(pmax, T // 4))
    if T - pmax < max(10, 2 * p):
        raise ValueError(f"{T} samples cannot support an AR({pmax}) whitening of a {p}-column design.")

    ok = np.all(np.isfinite(Y), axis=1) & np.all(np.isfinite(X), axis=(1, 2))[:, None]      # (S, C)
    ok &= np.any(Y != Y[:, :1], axis=1)
    Y = np.where(ok[:, None, :], Y, 0.0)

    n_dev = jax.local_device_count()
    step = -(-(session_batch or n_dev) // n_dev) * n_dev
    pad = -S % step                     # repeat the last session to fill the final call
    X = np.concatenate([X, np.repeat(X[-1:], pad, axis=0)])
    Y = np.concatenate([Y, np.repeat(Y[-1:], pad, axis=0)])
    fit = _compiled(n_dev, pmax, float(tune), int(max_iter), int(irls_max_iter))
    with jax.enable_x64(True):
        parts = [fit(jnp.asarray(X[i:i + step]), jnp.asarray(Y[i:i + step])) for i in range(0, S + pad, step)]
        out = {k: np.concatenate([np.asarray(part[k]) for part in parts])[:S] for k in parts[0]}

    def channels_last(a: np.ndarray) -> np.ndarray:
        """(S, C, ...) -> (S, ..., C), NaN for channels that were not fitted."""
        a = np.where(np.expand_dims(ok, tuple(range(2, a.ndim))), a, np.nan)
        return np.moveaxis(a, 1, -1)

    beta, covb = channels_last(out["beta"]), channels_last(out["covb"])
    se = np.sqrt(np.maximum(np.diagonal(covb, axis1=1, axis2=2), 0)).swapaxes(1, 2)
    tstat = beta / np.maximum(se, _EPS)
    dfe = np.where(ok, out["dfe"], np.nan)
    fits = ChannelFits(beta=beta, se=se, tstat=tstat, pval=t_pvalue(tstat, np.maximum(dfe, 1)[:, None, :]),
                       cov=covb, r2=np.where(ok, out["r2"], np.nan), dfe=dfe,
                       ar_order=np.where(ok, out["order"], np.nan), weights=channels_last(out["w"]))
    if not batched:
        fits = ChannelFits(**{k: v[0] for k, v in vars(fits).items()})
    return fits


@lru_cache(maxsize=None)
def _compiled(n_devices: int, pmax: int, tune: float, max_iter: int, irls_max_iter: int):
    """The jitted batch fit, sharded over sessions when there are several devices."""
    fn = partial(_fit_sessions, pmax=pmax, tune=tune, max_iter=max_iter, irls_max_iter=irls_max_iter)
    if n_devices > 1:
        mesh = Mesh(np.array(jax.local_devices()), ("session",))
        # Lanes are independent, so the per-device bodies need no collectives;
        # check_vma is off because the loop carries start out device-invariant.
        fn = jax.shard_map(fn, mesh=mesh, in_specs=(P("session"), P("session")), out_specs=P("session"),
                           check_vma=False)
    return jax.jit(fn)


def _fit_sessions(X, Y, *, pmax, tune, max_iter, irls_max_iter):
    one = partial(_ar_irls, pmax=pmax, tune=tune, max_iter=max_iter, irls_max_iter=irls_max_iter)

    def session(X, Y):
        # The lagged copies of the design are shared by every channel of the
        # session, so they are built once; whitening is then a contraction
        # with each channel's own filter taps.
        return jax.vmap(one, in_axes=(None, None, 1))(X, _lags(X, pmax), Y)

    return jax.vmap(session)(X, Y)


def _delay(z, i):
    """z delayed by i samples along axis 0, zero-filled."""
    return jnp.pad(z, [(i, 0)] + [(0, 0)] * (z.ndim - 1))[: z.shape[0]]


def _lags(Z, pmax):
    """(T, pmax + 1, ...) stack of Z delayed by 0..pmax samples."""
    return jnp.stack([_delay(Z, i) for i in range(pmax + 1)], axis=1)


def _fir(f, z):
    """Causal FIR filter with taps f (``lfilter(f, 1, z)``)."""
    return sum(f[i] * _delay(z, i) for i in range(f.shape[0]))


# ---------------------------------------------------------------------------
# One channel
# ---------------------------------------------------------------------------
def _ar_irls(X, X_lags, y, *, pmax, tune, max_iter, irls_max_iter):
    T, p = X.shape
    B0 = _lstsq(X, y)
    shapes = jax.eval_shape(_robust_fit, X, y, jnp.ones(T, bool), tune, irls_max_iter)
    empty = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), shapes)     # loop carry before the first fit

    def cond(state):
        it, done = state[0], state[1]
        return (it < max_iter) & ~done

    def body(state):
        it, _, B, B_prev, _, _, _ = state
        f, order = _levinson_bic(y - X @ B, pmax)
        use = jnp.arange(T) >= order                      # drop the filter start-up
        fit = _robust_fit(jnp.einsum("tip,i->tp", X_lags, f), _fir(f, y), use, tune, irls_max_iter)
        B_new = fit["beta"]
        done = jnp.linalg.norm(B_new - B_prev) / jnp.maximum(jnp.linalg.norm(B_prev), _EPS) < 1e-2
        return it + 1, done, B_new, B, fit, order, use

    zero = jnp.zeros((), int)
    init = (zero, jnp.array(False), B0, jnp.full(p, jnp.inf), empty, zero, jnp.ones(T, bool))
    _, _, B, _, fit, order, use = lax.while_loop(cond, body, init)
    return dict(beta=B, covb=fit["covb"], dfe=fit["dfe"], r2=fit["r2"], order=order,
                w=jnp.where(use, fit["w"], jnp.nan))


def _lstsq(X, y, w=None):
    """Weighted least squares by the normal equations (p is a handful of regressors)."""
    Xw = X if w is None else X * w[:, None]
    return jnp.linalg.solve(Xw.T @ X, Xw.T @ y)


def _levinson_bic(x, pmax):
    """Whitening filter [1, -phi], zero-padded to pmax + 1 taps, and its BIC-chosen order."""
    n = x.shape[0]
    x = x - x.mean()
    g = jnp.stack([x @ _delay(x, k) for k in range(pmax + 1)]) / n
    j = jnp.arange(pmax + 1)
    log_n = jnp.log(n)

    def step(carry, p):
        a, err, active, best_score, best_a, best_order = carry
        lower = (j >= 1) & (j < p)
        k = (g[p] - jnp.sum(jnp.where(lower, a * g[jnp.clip(p - j, 0, pmax)], 0.0))) / err
        a_new = jnp.where(lower, a - k * a[jnp.clip(p - j, 0, pmax)], jnp.where(j == p, k, a))
        err_new = err * (1 - k * k)
        ok = active & jnp.isfinite(k) & (jnp.abs(k) < 1) & (err_new > 0)
        score = n * jnp.log(jnp.where(ok, err_new, 1.0)) + p * log_n
        better = ok & (score < best_score)
        return (jnp.where(ok, a_new, a), jnp.where(ok, err_new, err), ok, jnp.where(better, score, best_score),
                jnp.where(better, a_new, best_a), jnp.where(better, p, best_order)), None

    a0 = (j == 0).astype(x.dtype)
    init = (a0, g[0], g[0] > 0, jnp.asarray(jnp.inf, x.dtype), a0, jnp.zeros((), j.dtype))
    (_, _, _, _, best_a, order), _ = lax.scan(step, init, jnp.arange(1, pmax + 1))
    return jnp.where(j == 0, 1.0, -best_a), order


# ---------------------------------------------------------------------------
# Robust bisquare fit on the rows in `use` (as robust.robust_fit, full-rank design)
# ---------------------------------------------------------------------------
def _bisquare(u):
    return jnp.where(jnp.abs(u) < 1, (1 - u**2) ** 2, 0.0)


def _robust_fit(X, y, use, tune, max_iter):
    p = X.shape[1]
    m = use.astype(X.dtype)
    X, y = X * m[:, None], y * m
    n = m.sum()
    L = jnp.linalg.cholesky(X.T @ X)
    b = jax.scipy.linalg.cho_solve((L, True), X.T @ y)
    XtX_inv = jax.scipy.linalg.cho_solve((L, True), jnp.eye(p))
    h = jnp.minimum(0.9999, jnp.sum((X @ XtX_inv) * X, axis=1))
    adj = 1 / jnp.sqrt(1 - h)
    dfe = n - p
    ols_s = jnp.linalg.norm(y - X @ b) / jnp.sqrt(dfe)
    y_mean = y.sum() / n
    sd_y = jnp.sqrt(jnp.sum(m * (y - y_mean) ** 2) / (n - 1))
    tiny_s = jnp.where(sd_y > 0, 1e-6 * sd_y, 1.0)

    def mad_sigma(r):
        # Median of |r| over the valid rows, omitting the p - 1 smallest. For
        # non-negative floats the IEEE bit pattern read as an integer is
        # monotone, and XLA sorts int64 several times faster than float64.
        bits = jnp.sort(lax.bitcast_convert_type(jnp.where(use, jnp.abs(r), jnp.inf), jnp.int64), stable=False)
        count = n.astype(int) - p + 1
        lo, hi = p - 1 + (count - 1) // 2, p - 1 + count // 2
        pair = lax.bitcast_convert_type(jnp.stack((bits[lo], bits[hi])), r.dtype)
        return (pair[0] + pair[1]) / 2 / 0.6745

    def cond(state):
        it, b, b0, _ = state
        moving = jnp.any(jnp.abs(b - b0) > _SQRT_EPS * jnp.maximum(jnp.abs(b), jnp.abs(b0)))
        return (it == 0) | (moving & (it < max_iter))

    def body(state):
        it, b, _, _ = state
        radj = (y - X @ b) * adj
        w = jnp.where(use, _bisquare(radj / (jnp.maximum(mad_sigma(radj), tiny_s) * tune)), 0.0)
        return it + 1, _lstsq(X, y, w), b, w

    _, b, _, w = lax.while_loop(cond, body, (jnp.zeros((), int), b, jnp.zeros(p), m))

    r = (y - X @ b) * m
    radj = r * adj
    st = jnp.maximum(mad_sigma(radj), tiny_s) * tune
    u = radj / st
    delta = 1e-4
    phi = u * _bisquare(u)
    dphi = ((u + delta) * _bisquare(u + delta) - (u - delta) * _bisquare(u - delta)) / (2 * delta)
    m1 = jnp.sum(dphi * m) / n
    m2 = jnp.sum((1 - h) * phi**2 * m) / (n - p)
    robust_s = (1 + (p / n) * (1 - m1) / m1) * jnp.sqrt(m2) * st / m1
    # All weights 0 or 1: the fit is OLS on a subset, and its own s applies.
    binary = jnp.all(~use | (w < _SQRT_EPS) | (w > 1 - _SQRT_EPS))
    inc = use & (w > 1 - _SQRT_EPS)
    robust_s = jnp.where(binary, jnp.linalg.norm(jnp.where(inc, r, 0.0)) / jnp.sqrt(inc.sum() - p), robust_s)
    sigma = jnp.maximum(robust_s, jnp.sqrt((ols_s**2 * p**2 + robust_s**2 * n) / (p**2 + n)))

    wm = jnp.sum(w * y) / jnp.maximum(w.sum(), _EPS)
    r2 = 1 - jnp.sum(w * r**2) / jnp.maximum(jnp.sum(w * (y - wm) ** 2), _EPS)
    return dict(beta=b, covb=XtX_inv * sigma**2, w=w, dfe=dfe, r2=r2)
