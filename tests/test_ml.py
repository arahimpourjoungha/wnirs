"""Tests for wnirs.ml on small synthetic windows (fast; no real data)."""
import numpy as np

import pytest
import torch

from wnirs.ml.data import WindowSet, session_split, windows_from_arrays
from wnirs.ml.features import connectivity_features, spectral_features, window_features
from wnirs.ml.models import WNIRSNet, count_params
from wnirs.ml.streaming import StreamingEncoder, calibrate_activation_scales, compare_to_torch
from wnirs.ml.train import TrainConfig, finetune, inner_val_split, pretrain, seed_everything

FS = 5.0
S = 8  # 4 channels x HbO/HbR


def _recording(rng, n=900, task_freq=0.05):
    t = np.arange(n) / 7.1
    hbo = rng.normal(0, 0.3, (n, 4)) + np.sin(2 * np.pi * task_freq * t)[:, None]
    hbr = -0.4 * hbo + rng.normal(0, 0.2, (n, 4))
    return hbo, hbr


@pytest.fixture(scope="module")
def ws():
    rng = np.random.default_rng(0)
    sets = []
    for subj in ("A", "B"):
        for ses in range(1, 7):
            for task, f in (("Flanker", 0.03), ("Rest", 0.0)):
                hbo, hbr = _recording(rng, task_freq=f)
                usable = np.array([True, True, ses % 3 != 0, True])
                sets.append(windows_from_arrays(hbo, hbr, usable, 7.1, fs_out=FS, win_s=20, hop_s=10,
                                                meta=dict(subject=subj, session=str(ses), task=task,
                                                          recording=f"{subj}_{ses}_{task}")))
    return WindowSet.concat(sets)


def test_windows_shape_and_mask(ws):
    assert ws.x.shape[1:] == (S, 100)
    assert ws.x.dtype == np.float32
    dropped = ws.meta.session.astype(int) % 3 == 0
    assert not ws.mask[dropped.to_numpy()][:, [2, 6]].any()
    assert np.all(ws.x[~ws.mask] == 0)


def test_session_split_holds_out_latest_sessions(ws):
    tr, te = session_split(ws.meta, test_frac=0.34)
    test_ses = set(ws.meta.iloc[te].session.astype(int))
    assert test_ses == {5, 6}
    keys = ws.meta.subject + ws.meta.session
    assert set(keys.iloc[tr]).isdisjoint(set(keys.iloc[te]))


def test_save_load_roundtrip(ws, tmp_path):
    ws.save(tmp_path / "w.npz")
    back = WindowSet.load(tmp_path / "w.npz")
    np.testing.assert_array_equal(back.x, ws.x)
    assert list(back.meta.session) == list(ws.meta.session)


def test_features_finite_and_connectivity_clamped(ws):
    f = spectral_features(ws.x[:5], ws.fs)
    assert f.shape == (5, S, 8) and np.isfinite(f).all()
    same = np.repeat(ws.x[:1, :1], S, axis=1)                       # identical signals: r = 1
    c = connectivity_features(same, np.ones((1, S), bool))
    assert np.isfinite(c).all()
    assert window_features(ws.x[:3], ws.mask[:3], ws.fs).shape[0] == 3


def test_model_is_causal():
    torch.manual_seed(0)
    m = WNIRSNet(S).eval()
    x = torch.randn(1, S, 100)
    mask = torch.ones(1, S, dtype=torch.bool)
    h1 = m.encoder(x, mask)
    x2 = x.clone()
    x2[..., 60:] = torch.randn(1, S, 40)
    h2 = m.encoder(x2, mask)
    assert torch.allclose(h1[..., :60], h2[..., :60], atol=1e-6)
    assert not torch.allclose(h1[..., 60:], h2[..., 60:])


def test_masked_channel_does_not_leak():
    torch.manual_seed(0)
    m = WNIRSNet(S).eval()
    x = torch.randn(1, S, 100)
    mask = torch.ones(1, S, dtype=torch.bool)
    mask[0, 3] = False
    x2 = x.clone()
    x2[0, 3] = 100.0
    assert torch.allclose(m(x, mask)["embedding"], m(x2, mask)["embedding"], atol=1e-6)


def test_model_is_small():
    assert count_params(WNIRSNet(62)) < 5000


def test_pretrain_reduces_loss_and_finetune_learns(ws):
    seed_everything(0)
    model = WNIRSNet(S, n_classes=2)
    losses = pretrain(model, ws, TrainConfig(epochs=6))
    assert losses[-1] < losses[0]
    y = (ws.meta.task == "Flanker").astype(int).to_numpy()
    tr, te = session_split(ws.meta, test_frac=0.34)
    tr, va = inner_val_split(ws.meta, tr)
    model = finetune(model, ws.subset(tr), y[tr], ws.subset(va), y[va], TrainConfig(epochs=15))
    from wnirs.ml.train import accuracy
    assert accuracy(model, ws.subset(te), y[te]) > 0.8


def test_streaming_matches_torch_exactly(ws):
    seed_everything(1)
    model = WNIRSNet(S)
    pretrain(model, ws, TrainConfig(epochs=2))      # non-trivial BN statistics
    stream = StreamingEncoder(model, ws.mask[0])
    d = compare_to_torch(model, stream, ws.x[0])
    assert d["quality_max_abs"] < 1e-5
    assert d["embedding_max_abs"] < 1e-5


def test_streaming_int8_close_to_float(ws):
    seed_everything(1)
    model = WNIRSNet(S)
    pretrain(model, ws, TrainConfig(epochs=2))
    scales = calibrate_activation_scales(model, ws.x[:32], ws.mask[:32])
    s8 = StreamingEncoder(model, ws.mask[0], int8=True, act_scales=scales)
    d = compare_to_torch(model, s8, ws.x[0])
    assert d["embedding_rel"] < 0.1
    s32 = StreamingEncoder(model, ws.mask[0])
    assert s8.memory_bytes()["weights"] < s32.memory_bytes()["weights"] / 2


def test_onnx_export_and_int8(ws, tmp_path):
    pytest.importorskip("onnxruntime")
    from wnirs.ml.export import export_onnx, onnx_predict, quantize_onnx
    seed_everything(2)
    model = WNIRSNet(S).eval()
    p = export_onnx(model, S, 100, tmp_path / "m.onnx")
    logits, q = onnx_predict(p, ws.x[:4], ws.mask[:4])
    with torch.no_grad():
        ref = model(torch.from_numpy(ws.x[:4]), torch.from_numpy(ws.mask[:4]))["logits"].numpy()
    np.testing.assert_allclose(logits, ref, atol=1e-4)
    p8 = quantize_onnx(p, ws.x[:16], ws.mask[:16], tmp_path / "m8.onnx")
    l8, _ = onnx_predict(p8, ws.x[:4], ws.mask[:4])
    assert l8.shape == ref.shape
