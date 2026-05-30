import os
import json
import itertools
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt


# ── PATHS ─────────────────────────────────────────────────────────────────────
# Point at the pooled dataset produced by pool_datasets.py.
# split_ubfc2.py-style train/val splits should be re-run on the pooled npz.
TRAIN_PATH = "/data/data/trainSINGLE.npz"
VAL_PATH   = "/data/data/valSINGLE.npz"
OUT_DIR    = "/data/results/efficientnet_crossval"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_FRACTION = 1.0

DEVICE = (
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


# ── MODEL ─────────────────────────────────────────────────────────────────────
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_split(npz_path):
    """
    Load a processed .npz split and return X in (N, 3, H, W) float32.

    Processing scripts now output (N, H, W, 1) — single-channel grayscale.
    We transpose to (N, 1, H, W) then repeat to (N, 3, H, W) for EfficientNet
    and CvT, which expect 3-channel ImageNet-style input.

    AudioMAE loaders should NOT call this function — they need (N, 1, H, W)
    and should load X directly without channel expansion.

    Handles both old 3-channel data (shape[-1]==3) and new 1-channel data
    (shape[-1]==1) for backwards compatibility.

    Also handles the trial_ids/window_ids key difference between UBFC and
    WESAD npz files gracefully.
    """
    data = np.load(npz_path, allow_pickle=True)
    
    X        = data["X"].astype(np.float32)          # (N, H, W, C)
    y        = data["y"].astype(np.int64)
    filenames = data["filenames"].astype(str)
    subjects = data["subjects"].astype(np.int64)

    # Handle trial_ids vs window_ids key difference
    if "trial_ids" in data:
        window_ids = data["trial_ids"].astype(np.int64)
    elif "window_ids" in data:
        window_ids = data["window_ids"].astype(np.int64)
    else:
        window_ids = np.zeros(len(y), dtype=np.int64)

    # Transpose (N, H, W, C) -> (N, C, H, W)
    if X.ndim == 4 and X.shape[-1] in (1, 3):
        X = np.transpose(X, (0, 3, 1, 2))   # (N, C, H, W)
    else:
        raise ValueError(f"Unexpected X shape: {X.shape}")

    # Expand single-channel to 3-channel for ImageNet-pretrained models
    # (EfficientNet, CvT). All three channels are identical — this is standard
    # practice for grayscale inputs to ImageNet-pretrained models.
    if X.shape[1] == 1:
        X = np.repeat(X, 3, axis=1)         # (N, 3, H, W)

    # X is still in [0, 1] float32 at this point. ImageNet normalization
    # happens in normalize_imagenet() just before tensor creation.

    return X, y, filenames, subjects, window_ids


def load_trainval():
    X_train, y_train, f_train, s_train, t_train = load_split(TRAIN_PATH)
    X_val,   y_val,   f_val,   s_val,   t_val   = load_split(VAL_PATH)

    X         = np.concatenate([X_train, X_val],   axis=0)
    y         = np.concatenate([y_train, y_val],   axis=0)
    filenames = np.concatenate([f_train, f_val],   axis=0)
    subjects  = np.concatenate([s_train, s_val],   axis=0)
    window_ids = np.concatenate([t_train, t_val],  axis=0)

    return X, y, filenames, subjects, window_ids


def apply_subject_fraction(X, y, filenames, subjects, window_ids, fraction):
    if fraction >= 1.0:
        return X, y, filenames, subjects, window_ids

    rng = np.random.default_rng(91)
    unique_subjects = np.unique(subjects)
    n_subjects = max(2, int(len(unique_subjects) * fraction))
    chosen_subjects = rng.choice(unique_subjects, size=n_subjects, replace=False)
    mask = np.isin(subjects, chosen_subjects)
    print(f"Using {mask.sum()} samples from {n_subjects}/{len(unique_subjects)} subjects")
    return X[mask], y[mask], filenames[mask], subjects[mask], window_ids[mask]


def normalize_imagenet(X):
    """Apply ImageNet mean/std normalization. X must be (N, 3, H, W) in [0,1]."""
    return (X - IMAGENET_MEAN) / IMAGENET_STD


# ── TRAINING ──────────────────────────────────────────────────────────────────
def make_loaders(X, y, train_idx, val_idx, batch_size):
    X_train = normalize_imagenet(X[train_idx])
    X_val   = normalize_imagenet(X[val_idx])

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y[train_idx], dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y[val_idx], dtype=torch.long),
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
    )


def train_one_epoch(model, loader, optimizer, criterion):
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


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * xb.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(yb.cpu().numpy())
    return {
        "loss":         total_loss / len(loader.dataset),
        "acc":          accuracy_score(labels, preds),
        "f1":           f1_score(labels, preds, zero_division=0),
        "balanced_acc": balanced_accuracy_score(labels, preds),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")

    X, y, filenames, subjects, window_ids = load_trainval()
    X, y, filenames, subjects, window_ids = apply_subject_fraction(
        X, y, filenames, subjects, window_ids, DATA_FRACTION
    )

    param_grid = {
        "lr":           [1e-4, 3e-4, 1e-3],
        "batch_size":   [8, 16],
        "weight_decay": [1e-5, 1e-4],
        "dropout":      [0.2, 0.3, 0.5],
        "epochs":       [10],
    }

    keys   = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    n_splits   = min(5, len(np.unique(subjects)))
    group_kfold = GroupKFold(n_splits=n_splits)

    results = []

    for combo in combos:
        params = dict(zip(keys, combo))
        print(f"\nTesting params: {params}")

        for fold, (train_idx, val_idx) in enumerate(
            group_kfold.split(X, y, subjects)
        ):
            print(f"\nFold {fold + 1}/{n_splits}")

            train_loader, val_loader = make_loaders(
                X, y, train_idx, val_idx, batch_size=int(params["batch_size"])
            )

            model = EfficientNetClassifier(
                num_classes=2,
                dropout=float(params["dropout"]),
                pretrained=True,
            ).to(DEVICE)
            
            for param in model.model.features.parameters():
                param.requires_grad = False

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=float(params["lr"]),
                weight_decay=float(params["weight_decay"]),
            )

            class_counts  = np.bincount(y[train_idx], minlength=2)
            class_weights = len(y[train_idx]) / (2 * np.maximum(class_counts, 1))
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
            )

            best_f1, best_metrics = -1.0, None

            for epoch in range(int(params["epochs"])):
                train_loss  = train_one_epoch(model, train_loader, optimizer, criterion)
                val_metrics = evaluate(model, val_loader, criterion)
                print(
                    f"Epoch {epoch+1}/{params['epochs']} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_acc={val_metrics['acc']:.4f} | "
                    f"val_f1={val_metrics['f1']:.4f}"
                )
                if val_metrics["f1"] > best_f1:
                    best_f1, best_metrics = val_metrics["f1"], val_metrics

            results.append({**params, "fold": fold, **best_metrics})

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "crossval_results.csv"), index=False)

    summary = (
        results_df
        .groupby(keys)[["loss", "acc", "f1", "balanced_acc"]]
        .mean()
        .reset_index()
        .sort_values("f1", ascending=False)
    )
    summary.to_csv(os.path.join(OUT_DIR, "cv_summary.csv"), index=False)

    # ── F1 SCORE VISUALIZATION ────────────────────────────────────────────────────

    summary_plot = summary.copy()

    summary_plot["combo"] = summary_plot.apply(
        lambda r: (
            f"lr={r['lr']}\n"
            f"bs={int(r['batch_size'])}\n"
            f"wd={r['weight_decay']}\n"
            f"do={r['dropout']}"
        ),
        axis=1,
    )

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(summary_plot)), summary_plot["f1"])

    plt.xticks(
        range(len(summary_plot)),
        summary_plot["combo"],
        rotation=45,
        ha="right",
    )

    plt.ylabel("Mean CV F1")
    plt.xlabel("Hyperparameter Combination")
    plt.title("EfficientNet Cross-Validation F1 Scores")
    plt.tight_layout()

    plot_path = os.path.join(OUT_DIR, "f1_scores_by_combo.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved plot: {plot_path}")
    
    
    # -- HORIZONTAL BAR CHART --------
    summary_plot = summary.sort_values("f1", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(summary_plot)), summary_plot["f1"])

    plt.yticks(
        range(len(summary_plot)),
        summary_plot.apply(
            lambda r:
            f"lr={r['lr']}, bs={int(r['batch_size'])}, "
            f"wd={r['weight_decay']}, do={r['dropout']}",
            axis=1
        )
    )

    plt.xlabel("Mean CV F1")
    plt.title("Hyperparameter Search Results")
    plt.tight_layout()
    
    plot_path_h = os.path.join(OUT_DIR, "f1_scores_by_combo_horizontal.png")
    plt.savefig(plot_path_h, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path_h}")
    

    best_params = summary.iloc[0][keys].to_dict()
    with open(os.path.join(OUT_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    print("\nBest params:")
    print(best_params)


if __name__ == "__main__":
    main()
