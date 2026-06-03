"""
train_all_models.py
====================

Unified training / evaluation pipeline for the three project models
(EfficientNet-B0, CvT-13, AudioMAE) on the pooled WESAD + UBFC-Phys
scalogram dataset.

For EACH selected model the pipeline runs three stages:

  Stage 1 — CV hyperparameter sweep
      GroupKFold cross-validation (grouped by subject) over a hyperparameter
      grid, fit on (train + val). The combo with the best mean fold F1 wins.

  Stage 2 — Final test evaluation
      Refit on TRAIN with the best params, early-stop / checkpoint on VAL,
      then evaluate once on the held-out TEST split. Saves metrics,
      confusion matrix and ROC curve.

  Stage 3 — Leave-One-Subject-Out (LOSO)
      Pool TRAIN + VAL + TEST, and for every subject train on all the others
      (best params) and predict the held-out subject. Aggregates window-level
      metrics over the full pooled prediction set plus a per-subject table.

Data contract (produced by pool_datasets.py):
    trainSINGLE.npz / valSINGLE.npz / testSINGLE.npz
    keys: X (N, H, W, C) float in [0,1], y (N,), subjects (N,),
          filenames (N,), and one of trial_ids / window_ids.

Each model declares its own input size, channel count and normalization, so
the same data array is reshaped appropriately per model:
    EfficientNet-B0 : 224x224, 3ch, ImageNet norm
    CvT-13          : 224x224, 3ch, ImageNet norm
    AudioMAE        : 128x128, 1ch, dataset standardization  (pluggable)

Usage:
    python train_all_models.py
Edit the CONFIG block to choose models, splits, grids and toggles.
"""

import os
import json
import argparse
import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
# Paths. On Modal the persistent volume is mounted at /data; when running
# locally that path doesn't exist (and / is read-only), so fall back to the
# directory this script lives in. Override any of these on the CLI
# (--train/--val/--test/--out-root) if your layout differs.
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = "/data" if os.path.isdir("/data") else BASE_DIR

TRAIN_PATH = os.path.join("trainSINGLE.npz")
VAL_PATH   = os.path.join("valSINGLE.npz")
TEST_PATH  = os.path.join("testSINGLE.npz")
OUT_ROOT   = os.path.join( "results", "all_models")

# Which models to run by default (override with --models on the CLI).
MODELS_TO_RUN = ["efficientnet", "cvt", "audiomae"]

# Which stages to run by default (override with --stages on the CLI).
# Stages run independently: "cv" sweeps + saves best_params.json; "test" and
# "loso" reuse a saved best_params.json if "cv" isn't part of the same run.
STAGES = ["cv", "test", "loso"]

CV_FOLDS = 5            # GroupKFold splits for the sweep (capped to #subjects)
PHASE2_BLOCKS = 0       # >0 = also unfreeze the last N backbone blocks (head-only if 0)

# Debug: subsample subjects to make a fast end-to-end smoke test
QUICK_TEST = False
QUICK_FRACTION = 0.3

SEED = 91

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

CLASS_NAMES = ["No Stress", "Stress"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ADAPTERS
#  Every adapter exposes the same API so the training code is model-agnostic:
#    forward(x) -> logits (N, 2)
#    freeze_backbone()
#    unfreeze_top_blocks(n)
#    trainable_params()
# ══════════════════════════════════════════════════════════════════════════════
class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 with a 2-class head and freeze/unfreeze controls."""

    def __init__(self, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 2),
        )

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        for p in self.model.features.parameters():
            p.requires_grad = False

    def unfreeze_top_blocks(self, n: int):
        n_blocks = len(self.model.features)
        cutoff = n_blocks - n
        for i, block in enumerate(self.model.features):
            for p in block.parameters():
                p.requires_grad = (i >= cutoff)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


class CvTClassifier(nn.Module):
    """
    CvT-13 (HuggingFace `microsoft/cvt-13`) with a 2-class head.

    Requires:  pip install transformers
    """

    def __init__(self, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        try:
            from transformers import CvtForImageClassification, CvtConfig
        except ImportError as e:
            raise ImportError(
                "CvT needs the `transformers` package. Add it to your "
                "Modal image pip_install list."
            ) from e

        if pretrained:
            self.base = CvtForImageClassification.from_pretrained(
                "microsoft/cvt-13", num_labels=2, ignore_mismatched_sizes=True,
            )
        else:
            self.base = CvtForImageClassification(CvtConfig(num_labels=2))

        in_f = self.base.classifier.in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_f, 2),
        )

    def forward(self, x):
        return self.base(pixel_values=x).logits

    def freeze_backbone(self):
        for p in self.base.cvt.parameters():
            p.requires_grad = False

    def unfreeze_top_blocks(self, n: int):
        # CvT-13 has 3 transformer stages; unfreeze the last n of them.
        stages = self.base.cvt.encoder.stages
        cutoff = len(stages) - n
        for i, stage in enumerate(stages):
            for p in stage.parameters():
                p.requires_grad = (i >= cutoff)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


class AudioMAEClassifier(nn.Module):
    """
    AudioMAE wrapper.

    Uses the timm port of the AudioMAE ViT-B/16 weights (pretrained on
    AudioSet-2M, no classification head):
        gaunernst/vit_base_patch16_1024_128.audiomae_as2m
    loaded via load_audiomae_encoder(). The encoder maps (N, 1, 1024, 128) to a
    pooled (N, 768) feature vector; the learned head does the 2-class call.

    Requires:  pip install timm
    """

    def __init__(self, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.encoder, embed_dim = load_audiomae_encoder(pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, x):
        feats = self.encoder(x)      # (N, 768): timm returns pooled features
        if feats.ndim == 3:          # safety: (N, tokens, dim) -> mean-pool
            feats = feats.mean(dim=1)
        return self.head(feats)

    def freeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_top_blocks(self, n: int):
        blocks = getattr(self.encoder, "blocks", None)
        if blocks is None:
            return
        cutoff = len(blocks) - n
        for i, blk in enumerate(blocks):
            for p in blk.parameters():
                p.requires_grad = (i >= cutoff)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


def load_audiomae_encoder(pretrained: bool = True):
    """
    Load the AudioMAE ViT-B/16 backbone via timm (no classification head).

    Source: gaunernst/vit_base_patch16_1024_128.audiomae_as2m
    Returns (encoder, embed_dim). With num_classes=0 the encoder's __call__
    returns pooled features (N, 768).
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "AudioMAE needs the `timm` package. Add it to your Modal image "
            "pip_install list."
        ) from e

    name = "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m"
    # Current timm: let the repo's config set up pooling (it uses the checkpoint's
    # `norm` + mean-pool). Do NOT pass global_pool="avg" here — that forces timm
    # to expect an `fc_norm` layer the AudioMAE checkpoint doesn't have, which is
    # the "Missing fc_norm / Unexpected norm" load error. Only old timm (<0.9.11)
    # needs the explicit override, so try clean first and fall back.
    try:
        enc = timm.create_model(name, pretrained=pretrained, num_classes=0)
    except RuntimeError:
        enc = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                global_pool="avg")
    # Do NOT override img_size / in_chans: the config fixes the native
    # (1, 1024, 128) input (512 tokens on a 64x8 grid). Inputs are fed at that
    # native shape — see the AudioMAE entry in REGISTRY (input_size=(1024, 128)).
    return enc, enc.num_features   # 768 for ViT-B


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ModelSpec:
    name: str
    builder: Callable[[float, bool], nn.Module]
    input_size: int | tuple   # int -> square; (H, W) -> non-square (AudioMAE)
    in_channels: int
    mean: tuple
    std: tuple
    grid: dict
    supports_phase2: bool = True
    standardize: bool = False   # per-image zero-mean/unit-std instead of fixed mean/std


REGISTRY: dict[str, ModelSpec] = {
    "efficientnet": ModelSpec(
        name="EfficientNet-B0",
        builder=EfficientNetClassifier,
        input_size=224, in_channels=3,
        mean=IMAGENET_MEAN, std=IMAGENET_STD,
        grid={
            "lr":           [1e-4, 3e-4, 1e-3],
            "batch_size":   [8, 16],
            "weight_decay": [1e-5, 1e-4],
            "dropout":      [0.2, 0.3, 0.5],
            "epochs":       [10],
        },
    ),
    "cvt": ModelSpec(
        name="CvT-13",
        builder=CvTClassifier,
        input_size=224, in_channels=3,
        mean=IMAGENET_MEAN, std=IMAGENET_STD,
        # 20M params + transformer => keep the sweep modest
        grid={
            "lr":           [1e-4, 3e-4],
            "batch_size":   [8, 16],
            "weight_decay": [1e-5, 1e-4],
            "dropout":      [0.2, 0.3],
            "epochs":       [10],
        },
    ),
    "audiomae": ModelSpec(
        name="AudioMAE",
        builder=AudioMAEClassifier,
        input_size=(1024, 128), in_channels=1,   # native AudioMAE shape (time, mel)
        mean=(0.0,), std=(1.0,),    # unused: standardize=True does per-image norm
        standardize=True,           # more robust under the scalogram->spectrogram shift
        grid={
            "lr":           [1e-4, 3e-4, 1e-3],
            "batch_size":   [16, 32],
            "weight_decay": [1e-5, 1e-4],
            "dropout":      [0.2, 0.3],
            "epochs":       [10],
        },
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_split(npz_path: str):
    """Return X (N, H, W, C) float32 in [0,1], y, subjects, ids."""
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    subjects = data["subjects"].astype(np.int64)
    if "trial_ids" in data:
        ids = data["trial_ids"].astype(np.int64)
    elif "window_ids" in data:
        ids = data["window_ids"].astype(np.int64)
    else:
        ids = np.zeros(len(y), dtype=np.int64)
    if X.ndim != 4 or X.shape[-1] not in (1, 3):
        raise ValueError(f"Unexpected X shape {X.shape} in {npz_path}")
    return X, y, subjects, ids


def _resize_batch(X: np.ndarray, size, chunk: int = 256) -> np.ndarray:
    """X: (N, C, H, W) -> (N, C, target_H, target_W) via bilinear, in chunks.

    `size` is an int (square) or an (H, W) tuple (non-square, e.g. AudioMAE).
    """
    H, W = (size, size) if isinstance(size, int) else size
    if X.shape[-2] == H and X.shape[-1] == W:
        return X
    out = np.empty((X.shape[0], X.shape[1], H, W), dtype=np.float32)
    for i in range(0, len(X), chunk):
        t = torch.from_numpy(X[i:i + chunk])
        t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        out[i:i + chunk] = t.numpy()
    return out


def preprocess(X: np.ndarray, spec: ModelSpec) -> np.ndarray:
    """(N, H, W, C) in [0,1]  ->  (N, in_channels, size, size) normalized."""
    X = np.transpose(X, (0, 3, 1, 2))          # (N, C, H, W)
    C = X.shape[1]
    if spec.in_channels == 3 and C == 1:
        X = np.repeat(X, 3, axis=1)
    elif spec.in_channels == 1 and C == 3:
        X = X[:, :1]                           # grayscale: channels are identical
    X = _resize_batch(X, spec.input_size)
    if spec.standardize:
        # per-image zero-mean / unit-std (robust to the scalogram->spectrogram shift)
        m = X.mean(axis=(1, 2, 3), keepdims=True)
        s = X.std(axis=(1, 2, 3), keepdims=True) + 1e-6
        return ((X - m) / s).astype(np.float32)
    mean = np.array(spec.mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std  = np.array(spec.std,  dtype=np.float32).reshape(1, -1, 1, 1)
    return ((X - mean) / std).astype(np.float32)


def maybe_subsample(X, y, subjects, ids, fraction):
    rng = np.random.default_rng(SEED)
    uniq = np.unique(subjects)
    n = max(2, int(len(uniq) * fraction))
    chosen = rng.choice(uniq, size=n, replace=False)
    m = np.isin(subjects, chosen)
    print(f"  [QUICK] {m.sum()} samples from {n}/{len(uniq)} subjects")
    return X[m], y[m], subjects[m], ids[m]


def make_loader(Xp, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(Xp, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def class_weighted_criterion(y_train):
    counts = np.bincount(y_train, minlength=2)
    w = len(y_train) / (2 * np.maximum(counts, 1))
    return nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32).to(DEVICE)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN / EVAL CORE
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, probs, labels = [], [], []
    for xb, yb in loader:
        logits = model(xb.to(DEVICE))
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.extend(p)
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        labels.extend(yb.numpy())
    return np.array(labels), np.array(preds), np.array(probs)


def compute_metrics(labels, preds, probs):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")
    return {
        "acc":          float(accuracy_score(labels, preds)),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "f1":           float(f1_score(labels, preds, zero_division=0)),
        "sensitivity":  float(sens),
        "specificity":  float(spec),
        "auroc":        float(auroc),
    }


def build(spec, dropout, pretrained=True, phase2_blocks=0):
    m = spec.builder(dropout, pretrained)
    m.freeze_backbone()
    if phase2_blocks > 0 and spec.supports_phase2:
        m.unfreeze_top_blocks(phase2_blocks)
    return m.to(DEVICE)


def fit(model, train_loader, criterion, lr, epochs, weight_decay,
        val_loader=None, patience=None, verbose=False):
    """
    Train `model`. If val_loader is given, checkpoint the best val-F1 state in
    memory and (optionally) early-stop. Returns the model with best/last weights.
    """
    opt = torch.optim.AdamW(model.trainable_params(), lr=lr, weight_decay=weight_decay)
    best_f1, best_state, no_improve = -1.0, None, 0

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, criterion)
        if val_loader is not None:
            yl, yp, pr = predict(model, val_loader)
            vf1 = f1_score(yl, yp, zero_division=0)
            if verbose:
                print(f"    epoch {ep:>2}/{epochs}  train_loss={tr_loss:.4f}  val_f1={vf1:.4f}")
            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if patience is not None and no_improve >= patience:
                    if verbose:
                        print(f"    early stop @ epoch {ep}")
                    break
        elif verbose:
            print(f"    epoch {ep:>2}/{epochs}  train_loss={tr_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — CV HYPERPARAMETER SWEEP
# ══════════════════════════════════════════════════════════════════════════════
def cv_sweep(spec, X, y, subjects, out_dir):
    print(f"\n{'='*70}\n  [{spec.name}] STAGE 1 — CV hyperparameter sweep\n{'='*70}")
    Xp = preprocess(X, spec)

    keys = list(spec.grid.keys())
    combos = list(itertools.product(*spec.grid.values()))
    k = min(CV_FOLDS, len(np.unique(subjects)))
    gkf = GroupKFold(n_splits=k)

    rows = []
    for ci, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"\n  Combo {ci}/{len(combos)}: {params}")
        for fold, (tr, va) in enumerate(gkf.split(Xp, y, subjects), 1):
            train_loader = make_loader(Xp[tr], y[tr], int(params["batch_size"]), True)
            val_loader   = make_loader(Xp[va], y[va], int(params["batch_size"]), False)
            criterion = class_weighted_criterion(y[tr])

            model = build(spec, float(params["dropout"]), pretrained=True,
                          phase2_blocks=PHASE2_BLOCKS)
            model, best_f1 = fit(
                model, train_loader, criterion,
                lr=float(params["lr"]), epochs=int(params["epochs"]),
                weight_decay=float(params["weight_decay"]),
                val_loader=val_loader,
            )
            yl, yp, pr = predict(model, val_loader)
            m = compute_metrics(yl, yp, pr)
            print(f"    fold {fold}/{k}  f1={m['f1']:.4f}  bal_acc={m['balanced_acc']:.4f}")
            rows.append({**params, "fold": fold, **m})
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(out_dir, "cv_results.csv"), index=False)

    summary = (results.groupby(keys)[["acc", "balanced_acc", "f1",
                                      "sensitivity", "specificity", "auroc"]]
               .mean().reset_index().sort_values("f1", ascending=False))
    summary.to_csv(os.path.join(out_dir, "cv_summary.csv"), index=False)

    _plot_cv_f1(summary, keys, out_dir)

    best_params = summary.iloc[0][keys].to_dict()
    # cast numeric types back
    for kk in ("batch_size", "epochs"):
        if kk in best_params:
            best_params[kk] = int(best_params[kk])
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n  Best params for {spec.name}: {best_params}")
    return best_params


def _plot_cv_f1(summary, keys, out_dir):
    s = summary.sort_values("f1", ascending=True)
    labels = s.apply(
        lambda r: ", ".join(f"{k}={r[k]}" for k in keys if k != "epochs"), axis=1)
    plt.figure(figsize=(10, max(4, 0.4 * len(s))))
    plt.barh(range(len(s)), s["f1"])
    plt.yticks(range(len(s)), labels, fontsize=7)
    plt.xlabel("Mean CV F1")
    plt.title("Hyperparameter search")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_f1_by_combo.png"), dpi=200)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — FINAL TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def final_test(spec, best_params, splits, out_dir):
    print(f"\n{'='*70}\n  [{spec.name}] STAGE 2 — final fit + test eval\n{'='*70}")
    (Xtr, ytr, _, _), (Xva, yva, _, _), (Xte, yte, ste, _) = splits

    # Pool train + val for the final fit. The Stage-1 sweep already used
    # train+val (via subject-grouped CV) to select hyperparameters, so val has
    # done its job; we retrain on all of it for maximum data. No held-out dev
    # set remains, so we train a fixed number of epochs (the count carried in
    # best_params from the sweep) rather than early-stopping.
    Xdev = np.concatenate([Xtr, Xva], axis=0)
    ydev = np.concatenate([ytr, yva], axis=0)
    print(f"  Final fit on pooled train+val: {len(ydev)} samples, "
          f"class dist {np.bincount(ydev)}")

    Xdev_p = preprocess(Xdev, spec)
    Xte_p  = preprocess(Xte, spec)

    bs = int(best_params["batch_size"])
    train_loader = make_loader(Xdev_p, ydev, bs, True)
    test_loader  = make_loader(Xte_p, yte, bs, False)
    criterion = class_weighted_criterion(ydev)

    model = build(spec, float(best_params["dropout"]), pretrained=True,
                  phase2_blocks=PHASE2_BLOCKS)
    model, _ = fit(
        model, train_loader, criterion,
        lr=float(best_params["lr"]), epochs=int(best_params["epochs"]),
        weight_decay=float(best_params["weight_decay"]),
        val_loader=None,   # train the full epoch budget on the pooled set
        verbose=True,
    )
    torch.save(model.state_dict(), os.path.join(out_dir, "final_model.pt"))

    yl, yp, pr = predict(model, test_loader)
    metrics = compute_metrics(yl, yp, pr)
    print("\n  TEST metrics:")
    for k, v in metrics.items():
        print(f"    {k:<14}: {v:.4f}")
    print(classification_report(yl, yp, target_names=CLASS_NAMES, zero_division=0))

    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _plot_confusion(yl, yp, f"{spec.name} — Test", os.path.join(out_dir, "cm_test.png"))
    _plot_roc(yl, pr, f"{spec.name} — Test", os.path.join(out_dir, "roc_test.png"))

    # per-subject (window-level) accuracy on test
    _per_subject_table(yl, yp, pr, ste, os.path.join(out_dir, "per_subject_test.csv"),
                       os.path.join(out_dir, "per_subject_test.png"),
                       f"{spec.name} — Per-subject test accuracy")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — LEAVE-ONE-SUBJECT-OUT
# ══════════════════════════════════════════════════════════════════════════════
def loso(spec, best_params, X, y, subjects, out_dir):
    print(f"\n{'='*70}\n  [{spec.name}] STAGE 3 — LOSO over pooled dataset\n{'='*70}")
    Xp = preprocess(X, spec)
    uniq = np.unique(subjects)
    bs = int(best_params["batch_size"])

    all_labels = np.empty(len(y), dtype=np.int64)
    all_preds  = np.empty(len(y), dtype=np.int64)
    all_probs  = np.empty(len(y), dtype=np.float32)

    for i, s in enumerate(uniq, 1):
        te = subjects == s
        tr = ~te
        print(f"  Fold {i}/{len(uniq)}  hold-out subject {s}  "
              f"(train={tr.sum()}, test={te.sum()})")

        train_loader = make_loader(Xp[tr], y[tr], bs, True)
        test_loader  = make_loader(Xp[te], y[te], bs, False)
        criterion = class_weighted_criterion(y[tr])

        model = build(spec, float(best_params["dropout"]), pretrained=True,
                      phase2_blocks=PHASE2_BLOCKS)
        # No internal val here: best params already chosen -> fixed epochs.
        model, _ = fit(
            model, train_loader, criterion,
            lr=float(best_params["lr"]), epochs=int(best_params["epochs"]),
            weight_decay=float(best_params["weight_decay"]),
        )
        yl, yp, pr = predict(model, test_loader)
        idx = np.where(te)[0]
        all_labels[idx], all_preds[idx], all_probs[idx] = yl, yp, pr
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Aggregate over the full pooled prediction set (global AUROC well-defined)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    print("\n  LOSO aggregate metrics:")
    for k, v in metrics.items():
        print(f"    {k:<14}: {v:.4f}")

    with open(os.path.join(out_dir, "loso_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"subject": subjects, "label": all_labels,
                  "pred": all_preds, "prob": all_probs}).to_csv(
        os.path.join(out_dir, "loso_predictions.csv"), index=False)

    _plot_confusion(all_labels, all_preds, f"{spec.name} — LOSO",
                    os.path.join(out_dir, "cm_loso.png"))
    _plot_roc(all_labels, all_probs, f"{spec.name} — LOSO",
              os.path.join(out_dir, "roc_loso.png"))
    _per_subject_table(all_labels, all_preds, all_probs, subjects,
                       os.path.join(out_dir, "per_subject_loso.csv"),
                       os.path.join(out_dir, "per_subject_loso.png"),
                       f"{spec.name} — Per-subject LOSO accuracy")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _plot_confusion(labels, preds, title, path):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(2), yticks=range(2),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True", title=title)
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_roc(labels, probs, title, path):
    if len(np.unique(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, lw=2, label=f"AUROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _per_subject_table(labels, preds, probs, subjects, csv_path, png_path, title):
    rows = []
    for s in np.unique(subjects):
        m = subjects == s
        true_lbl = int(np.round(labels[m].mean()))
        pred_lbl = int(np.round(preds[m].mean() >= 0.5))
        rows.append({
            "subject": int(s),
            "n_windows": int(m.sum()),
            "true_label": true_lbl,
            "pred_label": pred_lbl,
            "mean_prob": float(probs[m].mean()),
            "correct": int(true_lbl == pred_lbl),
            "window_acc": float(accuracy_score(labels[m], preds[m])),
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    d = df.sort_values("window_acc")
    colors = ["steelblue" if c == 1 else "tomato" for c in d["correct"]]
    fig, ax = plt.subplots(figsize=(max(6, len(d) * 0.4), 4))
    ax.bar(range(len(d)), d["window_acc"], color=colors)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([f"S{int(s)}" for s in d["subject"]], rotation=45, ha="right", fontsize=7)
    ax.axhline(0.5, color="gray", linestyle="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Window-level accuracy")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main(models=None, stages=None, cv_folds=None, phase2_blocks=None,
         quick=None, out_root=None, train_path=None, val_path=None, test_path=None):
    """
    Run the pipeline. All args default to the module-level CONFIG values, so
    `main()` with no args runs every stage for every model. The CLI (see
    parse_args) and Modal wrappers pass overrides here.
    """
    global CV_FOLDS, PHASE2_BLOCKS, QUICK_TEST, OUT_ROOT, TRAIN_PATH, VAL_PATH, TEST_PATH

    models = models or MODELS_TO_RUN
    stages = stages or STAGES
    if cv_folds      is not None: CV_FOLDS      = cv_folds
    if phase2_blocks is not None: PHASE2_BLOCKS = phase2_blocks
    if quick         is not None: QUICK_TEST    = quick
    if out_root      is not None: OUT_ROOT      = out_root
    if train_path    is not None: TRAIN_PATH    = train_path
    if val_path      is not None: VAL_PATH      = val_path
    if test_path     is not None: TEST_PATH     = test_path

    run_cv   = "cv"   in stages
    run_test = "test" in stages
    run_loso = "loso" in stages

    print(f"Device: {DEVICE}")
    print(f"Models: {models}  |  Stages: {stages}")
    os.makedirs(OUT_ROOT, exist_ok=True)

    print("Loading splits …")
    train = load_split(TRAIN_PATH)
    val   = load_split(VAL_PATH)
    test  = load_split(TEST_PATH)
    splits = (train, val, test)

    # CV is run on train+val (subject-grouped); test held out for stage 2.
    Xtv = np.concatenate([train[0], val[0]], axis=0)
    ytv = np.concatenate([train[1], val[1]], axis=0)
    stv = np.concatenate([train[2], val[2]], axis=0)
    itv = np.concatenate([train[3], val[3]], axis=0)

    # LOSO pools everything.
    Xall = np.concatenate([train[0], val[0], test[0]], axis=0)
    yall = np.concatenate([train[1], val[1], test[1]], axis=0)
    sall = np.concatenate([train[2], val[2], test[2]], axis=0)

    if QUICK_TEST:
        Xtv, ytv, stv, itv = maybe_subsample(Xtv, ytv, stv, itv, QUICK_FRACTION)
        Xall, yall, sall, _ = maybe_subsample(Xall, yall, sall,
                                              np.zeros_like(yall), QUICK_FRACTION)

    print(f"  train+val : {len(ytv)} samples, {len(np.unique(stv))} subjects, "
          f"class dist {np.bincount(ytv)}")
    print(f"  test      : {len(test[1])} samples, class dist {np.bincount(test[1])}")
    print(f"  pooled    : {len(yall)} samples, {len(np.unique(sall))} subjects")

    overall = {}
    for key in models:
        spec = REGISTRY[key]
        out_dir = os.path.join(OUT_ROOT, key)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n\n########## {spec.name} ##########")

        # Stage 1 — sweep (or load existing best params if cv isn't in this run)
        bp_path = os.path.join(out_dir, "best_params.json")
        if run_cv:
            best_params = cv_sweep(spec, Xtv, ytv, stv, out_dir)
        elif os.path.exists(bp_path):
            best_params = json.load(open(bp_path))
            print(f"  Loaded best params: {best_params}")
        else:
            best_params = {k: v[0] for k, v in spec.grid.items()}
            print(f"  No saved best_params.json (run --stages cv first) "
                  f"-> falling back to grid defaults: {best_params}")

        model_summary = {"best_params": best_params}

        # Stage 2 — test eval
        if run_test:
            model_summary["test"] = final_test(spec, best_params, splits, out_dir)

        # Stage 3 — LOSO
        if run_loso:
            model_summary["loso"] = loso(spec, best_params, Xall, yall, sall, out_dir)

        overall[key] = model_summary
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(model_summary, f, indent=2)

    with open(os.path.join(OUT_ROOT, "all_models_summary.json"), "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\n\nDone. Results in {OUT_ROOT}")


def parse_args():
    p = argparse.ArgumentParser(
        description="CV sweep / held-out test / LOSO pipeline for the three models. "
                    "Stages run independently; 'test' and 'loso' reuse a saved "
                    "best_params.json when 'cv' isn't in the same run."
    )
    p.add_argument("--models", nargs="+", default=MODELS_TO_RUN,
                   choices=list(REGISTRY),
                   help="Models to run (default: all).")
    p.add_argument("--stages", nargs="+", default=STAGES,
                   choices=["cv", "test", "loso"],
                   help="Stages to run (default: all). e.g. --stages loso")
    p.add_argument("--cv-folds", type=int, default=None,
                   help="GroupKFold splits for the sweep.")
    p.add_argument("--phase2-blocks", type=int, default=None,
                   help=">0 also unfreezes the last N backbone blocks.")
    p.add_argument("--quick", action="store_true",
                   help="Subsample subjects for a fast smoke test.")
    p.add_argument("--out-root", default=None)
    p.add_argument("--train", default=None)
    p.add_argument("--val", default=None)
    p.add_argument("--test", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        models=args.models, stages=args.stages,
        cv_folds=args.cv_folds, phase2_blocks=args.phase2_blocks,
        quick=(args.quick or None),
        out_root=args.out_root, train_path=args.train,
        val_path=args.val, test_path=args.test,
    )