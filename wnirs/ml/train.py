"""Training loops: self-supervised pretraining, supervised fine-tuning, evaluation.

Kept deliberately plain (no trainer framework) so every step that affects a
result is visible: the seed, the split, what is frozen, what the loss sees.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import WindowSet
from .models import WNIRSNet, masked_recon_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 3e-3
    weight_decay: float = 1e-4
    channel_dropout: float = 0.1     # extra signals hidden per batch, mimicking lost channels
    freeze_encoder: bool = False     # linear probe when True
    patience: int = 8                # early stopping on the validation split
    device: str = "cpu"
    log: list = field(default_factory=list)


def _loader(ws: WindowSet, y: np.ndarray | None, batch: int, shuffle: bool) -> DataLoader:
    t = [torch.from_numpy(ws.x), torch.from_numpy(ws.mask)]
    if y is not None:
        t.append(torch.as_tensor(y))
    return DataLoader(TensorDataset(*t), batch_size=batch, shuffle=shuffle, drop_last=False)


def _drop_channels(mask: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return mask
    return mask & (torch.rand_like(mask, dtype=torch.float32) > p)


def pretrain(model: WNIRSNet, ws: WindowSet, cfg: TrainConfig) -> list[float]:
    """Masked causal reconstruction on unlabelled windows. Returns loss per epoch."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    losses = []
    model.train()
    for _ in range(cfg.epochs):
        tot, n = 0.0, 0
        for x, m in _loader(ws, None, cfg.batch_size, True):
            m = _drop_channels(m, cfg.channel_dropout)
            loss = masked_recon_loss(x, m, model)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        sched.step()
        losses.append(tot / n)
    return losses


def finetune(model: WNIRSNet, train: WindowSet, y_train: np.ndarray,
             val: WindowSet, y_val: np.ndarray, cfg: TrainConfig) -> WNIRSNet:
    """Fit the classifier head (and the encoder unless frozen), early-stopping on val accuracy.

    Validation must be sessions disjoint from both train and test; the caller
    is responsible for that split (see ``inner_val_split``).
    """
    for p in model.encoder.parameters():
        p.requires_grad = not cfg.freeze_encoder
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best, best_acc, wait = copy.deepcopy(model.state_dict()), -1.0, 0
    for epoch in range(cfg.epochs):
        model.train()
        if cfg.freeze_encoder:
            model.encoder.eval()             # keep BN statistics from pretraining
        for x, m, y in _loader(train, y_train, cfg.batch_size, True):
            m = _drop_channels(m, cfg.channel_dropout)
            loss = loss_fn(model(x, m)["logits"], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = accuracy(model, val, y_val)
        cfg.log.append({"epoch": epoch, "val_acc": acc})
        if acc > best_acc:
            best, best_acc, wait = copy.deepcopy(model.state_dict()), acc, 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best)
    return model


@torch.no_grad()
def predict(model: WNIRSNet, ws: WindowSet, batch: int = 128) -> np.ndarray:
    model.eval()
    out = [model(x, m)["logits"].argmax(-1) for x, m in _loader(ws, None, batch, False)]
    return torch.cat(out).numpy()


@torch.no_grad()
def embed(model: WNIRSNet, ws: WindowSet, batch: int = 128) -> np.ndarray:
    model.eval()
    return torch.cat([model.embed(x, m) for x, m in _loader(ws, None, batch, False)]).numpy()


def accuracy(model: WNIRSNet, ws: WindowSet, y: np.ndarray) -> float:
    return float((predict(model, ws) == y).mean()) if len(ws) else float("nan")


def session_vote(pred: np.ndarray, sessions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Majority vote of window predictions within each session.

    A wearable reads many windows per session; the per-session decision is
    what a product would report, and it is what fingerprinting is scored on.
    """
    keys = np.unique(sessions)
    return keys, np.array([np.bincount(pred[sessions == k]).argmax() for k in keys])


def inner_val_split(meta, train_idx: np.ndarray, frac: float = 0.2, seed: int = 0):
    """Hold out whole sessions of the training set for early stopping."""
    rng = np.random.default_rng(seed)
    key = (meta["subject"].astype(str) + "|" + meta["session"].astype(str)).to_numpy()
    ses = np.unique(key[train_idx])
    k = max(1, int(round(frac * len(ses))))
    val_ses = set(rng.choice(ses, k, replace=False))
    is_val = np.array([key[i] in val_ses for i in train_idx])
    return train_idx[~is_val], train_idx[is_val]


def finetune_quality(model: WNIRSNet, train: WindowSet, val: WindowSet, cfg: TrainConfig) -> WNIRSNet:
    """Train the per-signal quality head against QC-gate labels (``WindowSet.quality``).

    Every signal that has data is shown to the model -- including the ones QC
    rejects, since telling them apart is the point -- and the loss is taken
    only where a label exists.
    """
    for p in model.encoder.parameters():
        p.requires_grad = not cfg.freeze_encoder
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos = float(np.nanmean(train.quality))
    # QC failures are the minority class: down-weight passes so both count equally
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-3)))
    q_train = torch.from_numpy(train.quality)
    best, best_auc, wait = copy.deepcopy(model.state_dict()), -1.0, 0
    for epoch in range(cfg.epochs):
        model.train()
        for idx in torch.randperm(len(train)).split(cfg.batch_size):
            x, m, q = torch.from_numpy(train.x[idx]), torch.from_numpy(train.mask[idx]), q_train[idx]
            lab = m & ~torch.isnan(q)
            loss = loss_fn(model(x, m)["quality"][lab], q[lab])
            opt.zero_grad()
            loss.backward()
            opt.step()
        auc = quality_auc(model, val)
        cfg.log.append({"epoch": epoch, "val_auc": auc})
        if auc > best_auc:
            best, best_auc, wait = copy.deepcopy(model.state_dict()), auc, 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best)
    return model


@torch.no_grad()
def quality_scores(model: WNIRSNet, ws: WindowSet, batch: int = 128) -> np.ndarray:
    model.eval()
    return torch.cat([model(x, m)["quality"] for x, m in _loader(ws, None, batch, False)]).numpy()


def quality_auc(model: WNIRSNet, ws: WindowSet) -> float:
    from sklearn.metrics import roc_auc_score
    s = quality_scores(model, ws)
    lab = ws.mask & ~np.isnan(ws.quality)
    y = ws.quality[lab] > 0.5
    return float(roc_auc_score(y, s[lab])) if 0 < y.sum() < y.size else float("nan")
