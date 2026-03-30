
# -*- coding: utf-8 -*-
"""
fn_focal_attackhead.py
------------------------------------------------------------------
Drop-in utilities to add FN-Focal loss and an optional Attack Pattern Head
to your existing runner_combine pipeline *without changing your model combos*.

What this gives you
- FNFocalLoss: BCE-with-logits + focal factor, with *extra* emphasis on
  false negatives (missed attacks). Tunable via (alpha, gamma, fn_lambda, fn_gamma).
- AttackPatternHead (optional): a tiny MLP head that predicts attack
  categories from the **same tabular features**. Its loss is added as an
  auxiliary term (lambda_attack * CE) *only for samples with known pattern*.
- train_with_fnfocal(): a training loop you can call instead of your
  plain BCE loop. It supports DataLoaders that optionally provide pattern ids.

How to use (minimal edits in runner_combine):
1) Place this file next to your runner (same folder).
2) In runner_combine.py, add:
       from fn_focal_attackhead import FNFocalLoss, AttackPatternHead, train_with_fnfocal
3) Replace your train_torch(...) call with train_with_fnfocal(...)
   and pass flags like use_fn_focal=True, attack_head_nclass=K (or 0), etc.
   If you don't have pattern ids yet, pass y_pat=None when building the dataset
   (it will automatically disable the attack head term).

Author: ChatGPT
------------------------------------------------------------------
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# -------------------------------
# Losses
# -------------------------------

class FNFocalLoss(nn.Module):
    """
    Binary FN-focused focal loss with logits.
    - Standard focal:  FL = - alpha * (1-pt)^gamma * y*log(pt) - (1-alpha) * pt^gamma * (1-y)*log(1-pt)
    - FN emphasis: multiply positive term by (1 + fn_lambda * (1 - pt_pos)^fn_gamma)
      where pt_pos = sigmoid(logit) on positive class (y=1).
    Args
    ----
    alpha: float in (0,1). Weight on positives (y=1). Default 0.5.
    gamma: standard focal focusing parameter (>=0). Default 2.0.
    fn_lambda: extra weight on (potential) false negatives (>=0). Default 1.0.
    fn_gamma: curvature for FN emphasis (>=0). Default 2.0.
    reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0,
                 fn_lambda: float = 1.0, fn_gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.fn_lambda = float(fn_lambda)
        self.fn_gamma = float(fn_gamma)
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE (element-wise)
        bce = self.bce(logits, targets)

        # pt = prob of the target class (like focal)
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1.0 - p)

        # focal factor
        focal_factor = (1.0 - pt).clamp(min=1e-8).pow(self.gamma)

        # alpha factor: positives get alpha, negatives get (1-alpha)
        alpha_factor = torch.where(targets > 0.5,
                                   torch.full_like(bce, self.alpha),
                                   torch.full_like(bce, 1.0 - self.alpha))

        # FN emphasis: only for positives — larger when model is unsure (pt small)
        # weight_fn = 1 + fn_lambda * (1 - pt_pos)^fn_gamma
        pt_pos = p  # prob of class=1
        weight_fn = 1.0 + self.fn_lambda * (1.0 - pt_pos).clamp(min=1e-8).pow(self.fn_gamma)
        weight_fn = torch.where(targets > 0.5, weight_fn, torch.ones_like(weight_fn))

        loss = alpha_factor * focal_factor * bce * weight_fn

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# -------------------------------
# Attack Pattern Head (auxiliary)
# -------------------------------

class AttackPatternHead(nn.Module):
    """
    A tiny MLP head that consumes the SAME tabular feature vector x (B, D)
    and predicts a categorical 'attack pattern id' (0..K-1).
    If K=0 or None, you can skip creating this head.
    """
    def __init__(self, in_feats: int, n_classes: int,
                 hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.n_classes = int(n_classes)
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_classes)
        )
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)  # -1: missing pattern

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) tabular features
        return self.net(x)

    def loss(self, logits: torch.Tensor, y_pat: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, y_pat)


# -------------------------------
# Train Loop (drop-in)
# -------------------------------

@dataclass
class TrainCfg:
    device: str = "cuda"
    epochs: int = 12
    lr: float = 1e-3
    pos_weight: Optional[float] = None  # optional BCE pos_weight; not used by FNFocal
    use_fn_focal: bool = True
    alpha: float = 0.5       # FNFocal alpha
    gamma: float = 2.0       # FNFocal gamma
    fn_lambda: float = 1.0   # extra FN emphasis
    fn_gamma: float = 2.0    # curvature for FN emphasis
    attack_lambda: float = 0.2   # weight for attack-head loss
    print_every: int = 1


def _compute_pos_weight(y_numpy: np.ndarray) -> float:
    pos = float((y_numpy > 0.5).sum())
    neg = float((y_numpy <= 0.5).sum())
    if pos < 1.0:
        return 1.0
    return neg / (pos + 1e-6)


def train_with_fnfocal(
    model: nn.Module,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    cfg: TrainCfg,
    pattern_head: Optional[AttackPatternHead] = None,
) -> nn.Module:
    """
    Train `model` with FN-Focal loss for the main binary task.
    Optionally attach a pattern head fed with the SAME input features.
    DataLoader must yield (xb, yb) or (xb, yb, y_pat). If y_pat is missing,
    set to None or -1.
    """
    device = cfg.device
    model = model.to(device)
    params = list(model.parameters())
    if pattern_head is not None:
        pattern_head = pattern_head.to(device)
        params += list(pattern_head.parameters())

    opt = torch.optim.Adam(params, lr=cfg.lr)

    # Build loss
    if cfg.use_fn_focal:
        criterion = FNFocalLoss(alpha=cfg.alpha, gamma=cfg.gamma,
                                fn_lambda=cfg.fn_lambda, fn_gamma=cfg.fn_gamma,
                                reduction="mean")
    else:
        # fall back to weighted BCE
        pos_weight = None
        if cfg.pos_weight is None:
            # try derive from training data
            if hasattr(dl_tr, "dataset") and hasattr(dl_tr.dataset, "y"):
                y_np = dl_tr.dataset.y if isinstance(dl_tr.dataset.y, np.ndarray) else None
                if y_np is not None:
                    pos_weight = _compute_pos_weight(y_np)
        else:
            pos_weight = cfg.pos_weight
        pos_w_t = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_t)

    best = {"auc": -1.0, "state": None}

    for ep in range(1, cfg.epochs + 1):
        model.train()
        if pattern_head is not None:
            pattern_head.train()

        for batch in dl_tr:
            if len(batch) == 2:
                xb, yb = batch
                y_pat = None
            else:
                xb, yb, y_pat = batch
                if isinstance(y_pat, np.ndarray):
                    y_pat = torch.from_numpy(y_pat)

            xb = xb.to(device)
            yb = yb.to(device)

            # main forward
            logits = model(xb)

            # main loss
            loss_main = criterion(logits, yb)

            # aux loss
            loss_aux = 0.0
            if pattern_head is not None:
                pat_logits = pattern_head(xb)  # use same features
                if y_pat is None:
                    # fabricate -1 (ignore) if not provided
                    y_pat_t = torch.full((xb.size(0),), -1, dtype=torch.long, device=device)
                else:
                    y_pat_t = y_pat.to(device).long()
                loss_aux = pattern_head.loss(pat_logits, y_pat_t)

            loss = loss_main + (cfg.attack_lambda * loss_aux if pattern_head is not None else 0.0)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            y_true, y_prob = [], []
            for batch in dl_va:
                xb, yb = batch[0], batch[1]
                prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
                y_true.append(yb.numpy())
                y_prob.append(prob)
            y_true = np.concatenate(y_true)
            y_prob = np.concatenate(y_prob)
            # ROC-AUC as early stopping metric (compute defensively)
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                auc = float(roc_auc_score(y_true, y_prob))
                ap = float(average_precision_score(y_true, y_prob))
            except Exception:
                auc, ap = -1.0, -1.0

        if cfg.print_every and ep % cfg.print_every == 0:
            print(f"[val] ep{ep:02d} ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}")

        if auc > best["auc"]:
            best["auc"] = auc
            best["state"] = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "pattern_head": (
                    {k: v.detach().cpu().clone() for k, v in pattern_head.state_dict().items()}
                    if pattern_head is not None else None
                )
            }

    # restore best
    if best["state"] is not None:
        model.load_state_dict(best["state"]["model"])
        if pattern_head is not None and best["state"]["pattern_head"] is not None:
            pattern_head.load_state_dict(best["state"]["pattern_head"])

    return model
