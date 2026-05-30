import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
TRAIN_PATH = "/data/data/trainSINGLE.npz"
VAL_PATH   = "/data/data/valSINGLE.npz"
TEST_PATH  = "/data/data/testSINGLE.npz"   # set to None to skip
OUT_DIR    = "/data/results/efficientnet_train_test_phase1only"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "gradcam"), exist_ok=True)

# ── Best hyperparams from cross-validation ────────────────────────────────────
BATCH_SIZE   = 8
DROPOUT      = 0.2
WEIGHT_DECAY = 1e-5

# Phase 1 — head only
PHASE1_LR      = 1e-3
PHASE1_EPOCHS  = 20
PHASE1_PATIENCE = 10

# Phase 2 — head + last 3 backbone blocks (features[6,7,8])
# Use a lower LR to nudge pretrained weights gently
PHASE2_LR       = 1e-4
PHASE2_EPOCHS   = 15
PHASE2_PATIENCE = 6
N_BLOCKS_TO_UNFREEZE = 3

PHASE1_CKPT = os.path.join(OUT_DIR, "best_model_phase1.pt")
PHASE2_CKPT = os.path.join(OUT_DIR, "best_model_phase2.pt")

# Grad-CAM: how many val/test samples to visualise per class
GRADCAM_SAMPLES_PER_CLASS = 5

# Training Mode
RUN_PHASE2 = False

DEVICE = (
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

CLASS_NAMES = ["No Stress", "Stress"]


# ── MODEL ─────────────────────────────────────────────────────────────────────
class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 with two-phase fine-tuning.

    Phase 1: freeze_backbone() — only the classification head trains.
    Phase 2: unfreeze_top_blocks(n) — head + last n feature blocks train.

    EfficientNet-B0 features layout (9 blocks, indices 0–8):
        [0]  stem conv
        [1]  MBConv ×1  (stride 1)
        [2]  MBConv ×2  (stride 2)
        [3]  MBConv ×2  (stride 2)
        [4]  MBConv ×3  (stride 2)
        [5]  MBConv ×3  (stride 1)
        [6]  MBConv ×4  (stride 2)  ← unfrozen with n=3
        [7]  MBConv ×1  (stride 1)  ← unfrozen with n=3
        [8]  head conv              ← unfrozen with n=3
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.freeze_backbone()

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        for param in self.model.features.parameters():
            param.requires_grad = False

    def unfreeze_top_blocks(self, n: int):
        n_blocks      = len(self.model.features)
        unfreeze_from = n_blocks - n
        for i, block in enumerate(self.model.features):
            for param in block.parameters():
                param.requires_grad = (i >= unfreeze_from)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def print_param_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  Trainable : {trainable:>10,} / {total:,} ({100*trainable/total:.1f}%)")
        for i, block in enumerate(self.model.features):
            state = "FROZEN" if not any(p.requires_grad for p in block.parameters()) else "ACTIVE"
            print(f"  features[{i}] : {state}")
        state = "ACTIVE" if any(p.requires_grad for p in self.model.classifier.parameters()) else "FROZEN"
        print(f"  classifier : {state}")


# ── GRAD-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM for EfficientNet-B0.
    Target layer: model.features[-1]  (the final conv block, features[8]).
    """

    def __init__(self, model: EfficientNetClassifier):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self._hook_layer(model.model.features[-1])

    def _hook_layer(self, layer):
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Returns a CAM heatmap (H×W, values in [0,1]) for the input tensor x (1×C×H×W).
        """
        self.model.eval()
        x = x.to(DEVICE).requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global-average-pool the gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam     = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def save_hyperparameters(out_dir, class_weights=None):
    params = {
        "model": "EfficientNet-B0",
        "pretrained": True,

        "batch_size": BATCH_SIZE,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,

        "run_phase2": RUN_PHASE2,
        "n_blocks_to_unfreeze": N_BLOCKS_TO_UNFREEZE,

        "phase1": {
            "lr": PHASE1_LR,
            "epochs": PHASE1_EPOCHS,
            "patience": PHASE1_PATIENCE,
        },

        "phase2": {
            "lr": PHASE2_LR,
            "epochs": PHASE2_EPOCHS,
            "patience": PHASE2_PATIENCE,
        },

        "train_path": TRAIN_PATH,
        "val_path": VAL_PATH,
        "test_path": TEST_PATH,

        "device": DEVICE,
        "class_names": CLASS_NAMES,
    }
    
    if class_weights is not None:
        params["class_weights"] = class_weights.tolist()

    with open(os.path.join(out_dir, "hyperparameters.json"), "w") as f:
        json.dump(params, f, indent=4)


def save_gradcam_figure(
    raw_img: np.ndarray,      # (3, H, W) in [0, 1]  (pre-normalisation)
    cam: np.ndarray,          # (H, W)   in [0, 1]
    true_label: int,
    pred_label: int,
    prob_stress: float,
    out_path: str,
) -> None:
    img_hwc = np.transpose(raw_img, (1, 2, 0))  # (H, W, 3)

    heatmap = cm.jet(cam)[..., :3]              # (H, W, 3)
    overlay = 0.55 * img_hwc + 0.45 * heatmap
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(img_hwc)
    axes[0].set_title("Scalogram")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\nTrue: {CLASS_NAMES[true_label]}  "
        f"Pred: {CLASS_NAMES[pred_label]}  "
        f"P(stress)={prob_stress:.2f}"
    )
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_gradcam(
    model: EfficientNetClassifier,
    X_raw: np.ndarray,        # (N, 3, H, W)  un-normalised [0,1]
    y: np.ndarray,
    subjects: np.ndarray,
    split_name: str,
    out_dir: str,
    n_per_class: int = GRADCAM_SAMPLES_PER_CLASS,
) -> None:
    """
    Generate Grad-CAM images for n_per_class samples from each class.
    Picks a variety of subjects where possible.
    """
    gradcam = GradCAM(model)
    gradcam_dir = os.path.join(out_dir, "gradcam", split_name)
    os.makedirs(gradcam_dir, exist_ok=True)

    X_norm = normalize_imagenet(X_raw)

    for cls in range(2):
        indices = np.where(y == cls)[0]
        # Spread across subjects
        subj_arr = subjects[indices]
        unique_subjs = np.unique(subj_arr)
        chosen = []
        for s in unique_subjs:
            s_idx = indices[subj_arr == s]
            chosen.append(s_idx[0])
            if len(chosen) >= n_per_class:
                break
        # Pad if fewer subjects than n_per_class
        if len(chosen) < n_per_class:
            chosen = indices[:n_per_class].tolist()

        for i, idx in enumerate(chosen[:n_per_class]):
            x_tensor = torch.tensor(X_norm[idx: idx + 1], dtype=torch.float32)
            cam = gradcam.generate(x_tensor, class_idx=cls)

            with torch.no_grad():
                logits = model(x_tensor.to(DEVICE))
                prob   = torch.softmax(logits, dim=1)[0, 1].item()
            pred = logits.argmax(dim=1).item()

            fname = f"class{cls}_{CLASS_NAMES[cls].replace(' ','_')}_sample{i:02d}.png"
            save_gradcam_figure(
                raw_img     = X_raw[idx],
                cam         = cam,
                true_label  = int(y[idx]),
                pred_label  = pred,
                prob_stress = prob,
                out_path    = os.path.join(gradcam_dir, fname),
            )

    print(f"  Grad-CAM images saved → {gradcam_dir}/")


# ── DATA HELPERS ──────────────────────────────────────────────────────────────
def load_split(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)

    X         = data["X"].astype(np.float32)
    y         = data["y"].astype(np.int64)
    filenames = data["filenames"].astype(str)
    subjects  = data["subjects"].astype(np.int64)

    if "trial_ids" in data:
        window_ids = data["trial_ids"].astype(np.int64)
    elif "window_ids" in data:
        window_ids = data["window_ids"].astype(np.int64)
    else:
        window_ids = np.zeros(len(y), dtype=np.int64)

    if X.ndim == 4 and X.shape[-1] in (1, 3):
        X = np.transpose(X, (0, 3, 1, 2))
    else:
        raise ValueError(f"Unexpected X shape: {X.shape}")

    if X.shape[1] == 1:
        X = np.repeat(X, 3, axis=1)

    return X, y, filenames, subjects, window_ids


def normalize_imagenet(X: np.ndarray) -> np.ndarray:
    return (X - IMAGENET_MEAN) / IMAGENET_STD


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(normalize_imagenet(X), dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ── TRAINING & EVALUATION ─────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion) -> dict:
    """
    Returns loss, accuracy, balanced_acc, F1, sensitivity, specificity, AUROC,
    plus raw preds, probs, and labels.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * xb.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    probs  = np.array(all_probs)

    cm_    = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm_.ravel() if cm_.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")

    return {
        "loss":         total_loss / len(loader.dataset),
        "acc":          accuracy_score(labels, preds),
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "f1":           f1_score(labels, preds, zero_division=0),
        "sensitivity":  sensitivity,
        "specificity":  specificity,
        "auroc":        auroc,
        "preds":        preds,
        "probs":        probs,
        "labels":       labels,
    }


def evaluate_per_subject(
    model, X_raw: np.ndarray, y: np.ndarray, subjects: np.ndarray, criterion
) -> pd.DataFrame:
    """Window-level predictions aggregated per subject by majority vote."""
    model.eval()
    loader = make_loader(X_raw, y, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    rows = []
    for subj in np.unique(subjects):
        mask       = subjects == subj
        s_labels   = y[mask]
        s_preds    = np.array(all_preds)[mask]
        s_probs    = np.array(all_probs)[mask]
        true_label = int(np.round(s_labels.mean()))       # majority true label
        pred_label = int(np.round(s_preds.mean() >= 0.5)) # majority vote
        rows.append({
            "subject":      subj,
            "n_windows":    mask.sum(),
            "true_label":   true_label,
            "pred_label":   pred_label,
            "mean_prob":    float(s_probs.mean()),
            "correct":      int(true_label == pred_label),
            "window_acc":   float(accuracy_score(s_labels, s_preds)),
        })
    return pd.DataFrame(rows)


# ── PLOTTING ──────────────────────────────────────────────────────────────────
def plot_training_curves(history: list[dict], out_dir: str) -> None:
    df = pd.DataFrame(history)
    df["global_epoch"] = range(1, len(df) + 1)
    p2_start = df[df["phase"].str.startswith("Phase 2")]["global_epoch"].min()

    metrics = [
        ("loss",         "Loss"),
        ("f1",           "F1"),
        ("balanced_acc", "Balanced Accuracy"),
        ("sensitivity",  "Sensitivity"),
        ("specificity",  "Specificity"),
        ("auroc",        "AUROC"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.plot(df["global_epoch"], df[f"val_{key}"], label="val")
        if f"train_{key}" in df.columns:
            ax.plot(df["global_epoch"], df[f"train_{key}"], label="train", alpha=0.6)
        if not np.isnan(p2_start):
            ax.axvline(p2_start, color="gray", linestyle="--", linewidth=1, label="Phase 2")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)

    plt.suptitle("Two-Phase Training — EfficientNet-B0", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()


def plot_combined_curves(history: list[dict], out_dir: str) -> None:
    """
    Single figure: Loss (left y-axis) + Accuracy + F1 (right y-axis), vs epochs.
    Train lines are slightly transparent; val lines are solid.
    A vertical dashed line marks the Phase 1 → Phase 2 boundary.
    """
    df = pd.DataFrame(history)
    df["global_epoch"] = range(1, len(df) + 1)
    epochs = df["global_epoch"].values

    p2_rows  = df[df["phase"].str.startswith("Phase 2")]["global_epoch"]
    p2_start = p2_rows.min() if not p2_rows.empty else None

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # ── Loss on left axis ────────────────────────────────────────────────────
    l1, = ax1.plot(epochs, df["train_loss"],  color="#e07b54", lw=1.5, alpha=0.5,
                   linestyle="--", label="Train Loss")
    l2, = ax1.plot(epochs, df["val_loss"],    color="#e07b54", lw=2,
                   label="Val Loss")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", color="#e07b54", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#e07b54")
    ax1.set_ylim(bottom=0)

    # ── Accuracy & F1 on right axis ──────────────────────────────────────────
    l3, = ax2.plot(epochs, df["train_f1"],          color="#4c8bbe", lw=1.5, alpha=0.5,
                   linestyle="--", label="Train F1")
    l4, = ax2.plot(epochs, df["val_f1"],            color="#4c8bbe", lw=2,
                   label="Val F1")
    l5, = ax2.plot(epochs, df["train_balanced_acc"],color="#56a868", lw=1.5, alpha=0.5,
                   linestyle="--", label="Train Bal-Acc")
    l6, = ax2.plot(epochs, df["val_balanced_acc"],  color="#56a868", lw=2,
                   label="Val Bal-Acc")
    ax2.set_ylabel("Score (F1 / Balanced Accuracy)", color="#333333", fontsize=11)
    ax2.set_ylim(0, 1.05)

    # ── Phase boundary ───────────────────────────────────────────────────────
    extra_lines = []
    if p2_start is not None:
        vl = ax1.axvline(p2_start, color="gray", linestyle=":", linewidth=1.5,
                         label="Phase 2 start")
        extra_lines = [vl]

    # ── Combined legend ──────────────────────────────────────────────────────
    all_lines = [l1, l2, l3, l4, l5, l6] + extra_lines
    ax1.legend(all_lines, [l.get_label() for l in all_lines],
               loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.22), framealpha=0.9)

    plt.title("EfficientNet-B0 — Loss, Balanced Accuracy & F1 vs Epoch",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "training_curves_combined.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved combined curve → {out_path}")


def plot_confusion_matrix(labels, preds, title: str, out_path: str) -> None:
    cm_ = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(2), yticks=range(2),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted", ylabel="True", title=title,
    )
    for i, j in np.ndindex(cm_.shape):
        ax.text(j, i, str(cm_[i, j]), ha="center", va="center",
                color="white" if cm_[i, j] > cm_.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curve(labels, probs, title: str, out_path: str) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    auc         = roc_auc_score(labels, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, lw=2, label=f"AUROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_per_subject_accuracy(subj_df: pd.DataFrame, title: str, out_path: str) -> None:
    df = subj_df.sort_values("window_acc")
    colors = ["steelblue" if c == 1 else "tomato" for c in df["correct"]]

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.55), 4))
    bars = ax.bar(range(len(df)), df["window_acc"], color=colors)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"S{int(s)}" for s in df["subject"]], rotation=45, ha="right")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Window-level Accuracy")
    ax.set_title(title)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="steelblue", label="Subject correct (majority vote)"),
        Patch(color="tomato",    label="Subject wrong (majority vote)"),
    ], fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── METRICS SUMMARY ───────────────────────────────────────────────────────────
def print_and_save_metrics(metrics: dict, split_name: str, out_dir: str) -> None:
    print(f"\n── {split_name} Metrics ──────────────────────────────────────────")
    for k in ["loss", "acc", "balanced_acc", "f1", "sensitivity", "specificity", "auroc"]:
        print(f"  {k:<16}: {metrics[k]:.4f}")
    print("\nClassification Report:")
    print(classification_report(metrics["labels"], metrics["preds"],
                                target_names=CLASS_NAMES, zero_division=0))

    summary = {k: float(metrics[k]) for k in
               ["loss", "acc", "balanced_acc", "f1", "sensitivity", "specificity", "auroc"]}
    fname = split_name.lower().replace(" ", "_") + "_metrics.json"
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(summary, f, indent=2)


# ── PHASE RUNNER ──────────────────────────────────────────────────────────────
def run_phase(
    phase_name, model, train_loader, val_loader, criterion,
    lr, epochs, patience, checkpoint_path,
) -> list[dict]:
    optimizer = torch.optim.AdamW(model.trainable_params(), lr=lr, weight_decay=WEIGHT_DECAY)

    print(f"\n{'='*60}")
    print(f" {phase_name}")
    print(f"{'='*60}")
    model.print_param_summary()
    print(f"  LR={lr}  Epochs={epochs}  Patience={patience}\n")

    history = []
    best_val_f1       = -1.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss    = train_one_epoch(model, train_loader, optimizer, criterion)
        train_metrics = evaluate(model, train_loader, criterion)
        val_metrics   = evaluate(model, val_loader, criterion)

        row = {
            "phase":             phase_name,
            "epoch":             epoch,
            "train_loss":        train_loss,
            "train_f1":          train_metrics["f1"],
            "train_balanced_acc":train_metrics["balanced_acc"],
            "train_sensitivity": train_metrics["sensitivity"],
            "train_specificity": train_metrics["specificity"],
            "train_auroc":       train_metrics["auroc"],
            "val_loss":          val_metrics["loss"],
            "val_acc":           val_metrics["acc"],
            "val_balanced_acc":  val_metrics["balanced_acc"],
            "val_f1":            val_metrics["f1"],
            "val_sensitivity":   val_metrics["sensitivity"],
            "val_specificity":   val_metrics["specificity"],
            "val_auroc":         val_metrics["auroc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_sens={val_metrics['sensitivity']:.4f} | "
            f"val_spec={val_metrics['specificity']:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1       = val_metrics["f1"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Best val F1 = {best_val_f1:.4f} — saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {epoch} epochs.")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print(f"\n{phase_name} complete. Best val F1 = {best_val_f1:.4f}")
    return history


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device : {DEVICE}")
    print(f"Hyperparams — LR(P1)={PHASE1_LR}, LR(P2)={PHASE2_LR}, "
          f"BS={BATCH_SIZE}, WD={WEIGHT_DECAY}, DO={DROPOUT}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading train split …")
    X_train, y_train, _, s_train, _ = load_split(TRAIN_PATH)
    print("Loading val split …")
    X_val,   y_val,   _, s_val,   _ = load_split(VAL_PATH)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)

    print(f"Train : {len(y_train)} samples  |  Val : {len(y_val)} samples")
    print(f"Train class dist : {np.bincount(y_train)}")
    print(f"Val   class dist : {np.bincount(y_val)}")

    # ── Class-weighted loss ───────────────────────────────────────────────────
    class_counts  = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2 * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    )
    print(f"Class weights : {class_weights}\n")

    # ── Build model ───────────────────────────────────────────────────────────
    model = EfficientNetClassifier(num_classes=2, dropout=DROPOUT, pretrained=True).to(DEVICE)

    # ── Phase 1: head only ────────────────────────────────────────────────────
    history_p1 = run_phase(
        "Phase 1 — Head only",
        model, train_loader, val_loader, criterion,
        lr=PHASE1_LR, epochs=PHASE1_EPOCHS, patience=PHASE1_PATIENCE,
        checkpoint_path=PHASE1_CKPT,
    )

    # ── Phase 2: head + last 3 blocks ─────────────────────────────────────────
    # model.unfreeze_top_blocks(N_BLOCKS_TO_UNFREEZE)
    # history_p2 = run_phase(
    #     f"Phase 2 — Head + last {N_BLOCKS_TO_UNFREEZE} blocks",
    #     model, train_loader, val_loader, criterion,
    #     lr=PHASE2_LR, epochs=PHASE2_EPOCHS, patience=PHASE2_PATIENCE,
    #     checkpoint_path=PHASE2_CKPT,
    # )
    
    history_p2 = []

    if RUN_PHASE2:
        model.unfreeze_top_blocks(N_BLOCKS_TO_UNFREEZE)
    
        history_p2 = run_phase(
            f"Phase 2 — Head + last {N_BLOCKS_TO_UNFREEZE} blocks",
            model, train_loader, val_loader, criterion,
            lr=PHASE2_LR, epochs=PHASE2_EPOCHS, patience=PHASE2_PATIENCE,
            checkpoint_path=PHASE2_CKPT,
        )

    # ── Training curves ───────────────────────────────────────────────────────
    all_history = history_p1 + history_p2
    pd.DataFrame(all_history).to_csv(os.path.join(OUT_DIR, "training_history.csv"), index=False)
    plot_training_curves(all_history, OUT_DIR)
    plot_combined_curves(all_history, OUT_DIR)

    # ── Val evaluation ────────────────────────────────────────────────────────
    val_metrics = evaluate(model, val_loader, criterion)
    print_and_save_metrics(val_metrics, "Validation", OUT_DIR)

    plot_confusion_matrix(
        val_metrics["labels"], val_metrics["preds"],
        title    = "Confusion Matrix — Val",
        out_path = os.path.join(OUT_DIR, "confusion_matrix_val.png"),
    )
    plot_roc_curve(
        val_metrics["labels"], val_metrics["probs"],
        title    = "ROC Curve — Val",
        out_path = os.path.join(OUT_DIR, "roc_curve_val.png"),
    )

    val_subj_df = evaluate_per_subject(model, X_val, y_val, s_val, criterion)
    val_subj_df.to_csv(os.path.join(OUT_DIR, "per_subject_val.csv"), index=False)
    plot_per_subject_accuracy(
        val_subj_df,
        title    = "Per-Subject Window Accuracy — Val",
        out_path = os.path.join(OUT_DIR, "per_subject_accuracy_val.png"),
    )

    print("\nGenerating Grad-CAM (val) …")
    run_gradcam(model, X_val, y_val, s_val, split_name="val", out_dir=OUT_DIR)

    # ── Test evaluation ───────────────────────────────────────────────────────
    if TEST_PATH is not None and os.path.exists(TEST_PATH):
        print("\nLoading test split …")
        X_test, y_test, _, s_test, _ = load_split(TEST_PATH)
        test_loader = make_loader(X_test, y_test, BATCH_SIZE, shuffle=False)
        print(f"Test : {len(y_test)} samples  |  Class dist : {np.bincount(y_test)}")

        test_metrics = evaluate(model, test_loader, criterion)
        print_and_save_metrics(test_metrics, "Test", OUT_DIR)

        plot_confusion_matrix(
            test_metrics["labels"], test_metrics["preds"],
            title    = "Confusion Matrix — Test",
            out_path = os.path.join(OUT_DIR, "confusion_matrix_test.png"),
        )
        plot_roc_curve(
            test_metrics["labels"], test_metrics["probs"],
            title    = "ROC Curve — Test",
            out_path = os.path.join(OUT_DIR, "roc_curve_test.png"),
        )

        test_subj_df = evaluate_per_subject(model, X_test, y_test, s_test, criterion)
        test_subj_df.to_csv(os.path.join(OUT_DIR, "per_subject_test.csv"), index=False)
        plot_per_subject_accuracy(
            test_subj_df,
            title    = "Per-Subject Window Accuracy — Test",
            out_path = os.path.join(OUT_DIR, "per_subject_accuracy_test.png"),
        )

        print("\nGenerating Grad-CAM (test) …")
        run_gradcam(model, X_test, y_test, s_test, split_name="test", out_dir=OUT_DIR)
    else:
        print("\nNo test split found — skipping.")

    save_hyperparameters(OUT_DIR, class_weights)
    
    print(f"\nAll outputs saved to: {OUT_DIR}")

    

if __name__ == "__main__":
    main()