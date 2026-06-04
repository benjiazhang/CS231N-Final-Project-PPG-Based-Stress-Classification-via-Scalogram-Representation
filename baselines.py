"""
baselines.py
============
1D time-series baselines for binary stress classification from raw BVP signal.

Models:
  - LSTM         : 2-layer bidirectional LSTM + classifier head
  - CNN1D        : 4-block 1D CNN + global average pooling + classifier head

Data:
  Loads raw BVP signal directly from WESAD .pkl files and UBFC-Phys .csv files.
  Uses identical windowing (30s, 1920 samples, 80% majority threshold) and
  label mapping as the CWT scalogram pipeline — ensuring a fair comparison.

Evaluation:
  - Stage 1: GroupKFold CV hyperparameter sweep on train+val subjects
  - Stage 2: Final training on train+val, evaluate on held-out test set
  - Stage 3: LOSO over all pooled subjects

Metrics: balanced accuracy, F1, sensitivity, specificity, AUROC
(same as vision pipeline for direct comparison)

Usage:
  python baselines.py --wesad_dir /path/to/WESAD --ubfc_dir /path/to/ubfc_bvp
  python baselines.py --wesad_dir /data/WESAD --ubfc_dir /data/bvp_ubfc --out_root /data/results/baselines
"""

import os
import json
import pickle
import argparse
import itertools
from math import gcd

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report,
)
from scipy.signal import resample_poly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
FS           = 64.0
LABEL_FS     = 700
WINDOW_SEC   = 30
WINDOW_SAMP  = int(FS * WINDOW_SEC)   # 1920

MAJORITY_THRESHOLD = 0.80
STRESS_LABELS      = {2}
NON_STRESS_LABELS  = {1, 3}
VALID_WESAD        = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
UBFC_SUBJECT_OFFSET = 100   # UBFC subjects numbered 101-156 to avoid collision

CV_FOLDS   = 5
SEED       = 91
CLASS_NAMES = ["No Stress", "Stress"]

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

torch.manual_seed(SEED)
np.random.seed(SEED)

GRID = {
    "lstm": {
        "lr":           [1e-4, 3e-4, 1e-3],
        "batch_size":   [16, 32],
        "weight_decay": [1e-5, 1e-4],
        "dropout":      [0.2, 0.3],
        "epochs":       [30],
    },
    "cnn1d": {
        "lr":           [1e-4, 3e-4, 1e-3],
        "batch_size":   [16, 32],
        "weight_decay": [1e-5, 1e-4],
        "dropout":      [0.2, 0.3],
        "epochs":       [30],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _majority_label(label_window):
    unique, counts = np.unique(label_window, return_counts=True)
    best = np.argmax(counts)
    if counts[best] / len(label_window) >= MAJORITY_THRESHOLD:
        return int(unique[best])
    return -1


def load_wesad(wesad_dir):
    """Load raw BVP windows from WESAD .pkl files.
    Returns X (N, 1920), y (N,), subjects (N,)
    """
    all_X, all_y, all_s = [], [], []
    for sid in VALID_WESAD:
        pkl = os.path.join(wesad_dir, f"S{sid}", f"S{sid}.pkl")
        if not os.path.exists(pkl):
            print(f"  [skip] WESAD S{sid}: not found")
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        bvp       = data["signal"]["wrist"]["BVP"].flatten().astype(np.float32)
        labels700 = data["label"].flatten().astype(np.int32)

        g = gcd(int(LABEL_FS), int(FS))
        up, down = int(FS) // g, int(LABEL_FS) // g
        labels64 = np.round(resample_poly(labels700.astype(np.float64), up, down)).astype(np.int32)
        n = min(len(bvp), len(labels64))
        bvp, labels64 = bvp[:n], labels64[:n]

        n_win = n // WINDOW_SAMP
        for i in range(n_win):
            s, e = i * WINDOW_SAMP, (i + 1) * WINDOW_SAMP
            maj = _majority_label(labels64[s:e])
            if maj in STRESS_LABELS:
                y_val = 1
            elif maj in NON_STRESS_LABELS:
                y_val = 0
            else:
                continue
            all_X.append(bvp[s:e])
            all_y.append(y_val)
            all_s.append(sid)
        print(f"  WESAD S{sid}: {n_win} windows processed")

    return (np.array(all_X, dtype=np.float32),
            np.array(all_y, dtype=np.int64),
            np.array(all_s, dtype=np.int64))


def load_ubfc(ubfc_path):
    """Load UBFC-Phys BVP windows.
    ubfc_path can be:
      - a .npz file produced by process_ubfc_baselines.py (preferred)
      - a directory containing bvp_s*.csv raw files
    Returns X (N, 1920), y (N,), subjects (N,)
    """
    # Option 1: pre-processed .npz
    if ubfc_path.endswith(".npz") and os.path.isfile(ubfc_path):
        data = np.load(ubfc_path)
        X, y, subjects = data["X"], data["y"], data["subjects"]
        print(f"  UBFC (npz): {len(y)} windows, {len(np.unique(subjects))} subjects")
        return (X.astype(np.float32), y.astype(np.int64), subjects.astype(np.int64))

    # Option 2: raw CSV directory
    import glob, re
    all_X, all_y, all_s = [], [], []
    csv_files = sorted(glob.glob(os.path.join(ubfc_path, "bvp_s*.csv")))
    if not csv_files:
        print(f"  [warning] No UBFC BVP CSVs found in {ubfc_path}")
        return (np.zeros((0, WINDOW_SAMP), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64))

    for fpath in csv_files:
        basename = os.path.basename(fpath)
        match = re.search(r"s(\d+)_T(\d+)", basename)
        if not match:
            continue
        sid   = int(match.group(1))
        trial = int(match.group(2))
        label = 0 if trial == 1 else 1
        signal = np.loadtxt(fpath).astype(np.float32)
        n_win  = len(signal) // WINDOW_SAMP
        for i in range(n_win):
            s, e = i * WINDOW_SAMP, (i + 1) * WINDOW_SAMP
            all_X.append(signal[s:e])
            all_y.append(label)
            all_s.append(sid + UBFC_SUBJECT_OFFSET)

    print(f"  UBFC (csv): {len(all_X)} windows from {len(csv_files)} files")
    return (np.array(all_X, dtype=np.float32),
            np.array(all_y, dtype=np.int64),
            np.array(all_s, dtype=np.int64))


def normalize(X):
    """Per-window z-score normalization."""
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sigma


def subject_split(X, y, subjects, test_size=0.15, val_size=0.1765, seed=SEED):
    """Subject-level train/val/test split matching CWT pipeline."""
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tv_idx, te_idx = next(gss_test.split(X, y, groups=subjects))

    X_tv, y_tv, s_tv = X[tv_idx], y[tv_idx], subjects[tv_idx]
    X_te, y_te, s_te = X[te_idx], y[te_idx], subjects[te_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(gss_val.split(X_tv, y_tv, groups=s_tv))

    splits = {
        "train": (X_tv[tr_idx], y_tv[tr_idx], s_tv[tr_idx]),
        "val":   (X_tv[va_idx], y_tv[va_idx], s_tv[va_idx]),
        "test":  (X_te, y_te, s_te),
    }

    print("\n===== SPLIT SUMMARY =====")
    for name, (Xs, ys, ss) in splits.items():
        print(f"  {name}: {len(ys)} windows, {len(np.unique(ss))} subjects, "
              f"class dist {np.bincount(ys)}")

    # Leakage check
    tr_s = set(splits["train"][2])
    va_s = set(splits["val"][2])
    te_s = set(splits["test"][2])
    assert not (tr_s & va_s), f"train∩val leakage: {tr_s & va_s}"
    assert not (tr_s & te_s), f"train∩test leakage: {tr_s & te_s}"
    assert not (va_s & te_s), f"val∩test leakage: {va_s & te_s}"
    print("  Leakage check: OK")
    return splits


# ══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════════════════════
class LSTMClassifier(nn.Module):
    """Bidirectional 2-layer LSTM for 1D time series classification."""

    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 dropout=0.3, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),  # *2 for bidirectional
        )

    def forward(self, x):
        # x: (B, 1920) -> (B, 1920, 1)
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        # Use last timestep output
        out = out[:, -1, :]
        return self.head(out)


class CNN1DClassifier(nn.Module):
    """4-block 1D CNN with global average pooling for time series classification."""

    def __init__(self, dropout=0.3, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            # Block 4
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, 1920) -> (B, 1, 1920)
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.mean(dim=-1)   # global average pooling
        return self.head(x)


def build_model(model_key, dropout):
    if model_key == "lstm":
        return LSTMClassifier(dropout=dropout).to(DEVICE)
    elif model_key == "cnn1d":
        return CNN1DClassifier(dropout=dropout).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_key}")


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def class_weighted_criterion(y):
    counts = np.bincount(y, minlength=2)
    w = len(y) / (2 * np.maximum(counts, 1))
    return nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32).to(DEVICE)
    )


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


def fit(model, train_loader, val_loader, criterion, lr, epochs,
        weight_decay, patience=10, verbose=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_f1, best_state, no_improve = -1.0, None, 0
    history = []

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, criterion)

        # Val loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        yl, yp, pr = predict(model, val_loader)
        vm = compute_metrics(yl, yp, pr)
        vf1 = vm["f1"]
        row = {"epoch": ep, "train_loss": tr_loss, "val_loss": val_loss}
        row.update({f"val_{k}": v for k, v in vm.items()})
        history.append(row)

        if verbose:
            print(f"    epoch {ep:>2}/{epochs}  train_loss={tr_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_f1={vf1:.4f}")

        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    early stop @ epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1, history


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STAGES
# ══════════════════════════════════════════════════════════════════════════════
def cv_sweep(model_key, X, y, subjects, out_dir):
    print(f"\n{'='*70}\n  [{model_key.upper()}] STAGE 1 — CV sweep\n{'='*70}")
    Xn = normalize(X)
    grid = GRID[model_key]
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    k = min(CV_FOLDS, len(np.unique(subjects)))
    gkf = GroupKFold(n_splits=k)

    rows = []
    for ci, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"\n  Combo {ci}/{len(combos)}: {params}")
        for fold, (tr, va) in enumerate(gkf.split(Xn, y, subjects), 1):
            train_loader = make_loader(Xn[tr], y[tr], int(params["batch_size"]), True)
            val_loader   = make_loader(Xn[va], y[va], int(params["batch_size"]), False)
            criterion    = class_weighted_criterion(y[tr])
            model = build_model(model_key, float(params["dropout"]))
            model, best_f1, _ = fit(
                model, train_loader, val_loader, criterion,
                lr=float(params["lr"]), epochs=int(params["epochs"]),
                weight_decay=float(params["weight_decay"]), patience=10,
            )
            yl, yp, pr = predict(model, val_loader)
            m = compute_metrics(yl, yp, pr)
            print(f"    fold {fold}/{k}  f1={m['f1']:.4f}  bal_acc={m['balanced_acc']:.4f}")
            rows.append({**params, "fold": fold, **m})
            del model
            if DEVICE == "cuda": torch.cuda.empty_cache()

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(out_dir, "crossval_results.csv"), index=False)

    hp_cols = [c for c in results_df.columns if c in keys]
    summary = (results_df.groupby(hp_cols)[["f1","balanced_acc"]]
               .mean().reset_index().sort_values("f1", ascending=False))
    summary.to_csv(os.path.join(out_dir, "cv_summary.csv"), index=False)

    best_row = summary.iloc[0]
    best_params = {k: best_row[k] for k in hp_cols}
    for kk in ("batch_size", "epochs"):
        if kk in best_params:
            best_params[kk] = int(best_params[kk])

    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n  Best params: {best_params}")
    print(f"  Best mean CV F1: {best_row['f1']:.4f}  bal_acc: {best_row['balanced_acc']:.4f}")
    return best_params


def final_test(model_key, best_params, splits, out_dir):
    print(f"\n{'='*70}\n  [{model_key.upper()}] STAGE 2 — final test eval\n{'='*70}")
    Xtr, ytr, _ = splits["train"]
    Xva, yva, _ = splits["val"]
    Xte, yte, ste = splits["test"]

    Xdev = normalize(np.concatenate([Xtr, Xva]))
    ydev = np.concatenate([ytr, yva])
    Xte_n = normalize(Xte)

    bs = int(best_params["batch_size"])
    train_loader = make_loader(Xdev, ydev, bs, True)
    val_loader   = make_loader(normalize(Xva), yva, bs, False)
    test_loader  = make_loader(Xte_n, yte, bs, False)
    criterion    = class_weighted_criterion(ydev)

    model = build_model(model_key, float(best_params["dropout"]))
    model, _, history = fit(
        model, train_loader, val_loader, criterion,
        lr=float(best_params["lr"]), epochs=int(best_params["epochs"]),
        weight_decay=float(best_params["weight_decay"]),
        patience=10, verbose=True,
    )
    torch.save(model.state_dict(), os.path.join(out_dir, "final_model.pt"))

    if history:
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
        _plot_training_curves(history, model_key.upper(), out_dir)

    yl, yp, pr = predict(model, test_loader)
    metrics = compute_metrics(yl, yp, pr)
    print("\n  TEST metrics:")
    for k, v in metrics.items():
        print(f"    {k:<14}: {v:.4f}")
    print(classification_report(yl, yp, target_names=CLASS_NAMES, zero_division=0))

    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _plot_confusion(yl, yp, f"{model_key.upper()} — Test", os.path.join(out_dir, "cm_test.png"))
    _plot_roc(yl, pr, f"{model_key.upper()} — Test", os.path.join(out_dir, "roc_test.png"))
    _per_subject_table(yl, yp, pr, ste,
                       os.path.join(out_dir, "per_subject_test.csv"),
                       os.path.join(out_dir, "per_subject_test.png"),
                       f"{model_key.upper()} — Per-subject test accuracy")
    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return metrics


def loso(model_key, best_params, X, y, subjects, out_dir):
    print(f"\n{'='*70}\n  [{model_key.upper()}] STAGE 3 — LOSO\n{'='*70}")
    Xn   = normalize(X)
    uniq = np.unique(subjects)
    bs   = int(best_params["batch_size"])

    all_labels = np.empty(len(y), dtype=np.int64)
    all_preds  = np.empty(len(y), dtype=np.int64)
    all_probs  = np.empty(len(y), dtype=np.float32)

    for i, s in enumerate(uniq, 1):
        te = subjects == s
        tr = ~te
        print(f"  Fold {i}/{len(uniq)}  hold-out S{s}  "
              f"(train={tr.sum()}, test={te.sum()})")
        train_loader = make_loader(Xn[tr], y[tr], bs, True)
        test_loader  = make_loader(Xn[te], y[te], bs, False)
        criterion    = class_weighted_criterion(y[tr])
        model = build_model(model_key, float(best_params["dropout"]))
        model, _, _ = fit(
            model, train_loader, test_loader, criterion,
            lr=float(best_params["lr"]), epochs=int(best_params["epochs"]),
            weight_decay=float(best_params["weight_decay"]), patience=10,
        )
        yl, yp, pr = predict(model, test_loader)
        idx = np.where(te)[0]
        all_labels[idx], all_preds[idx], all_probs[idx] = yl, yp, pr
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    print("\n  LOSO aggregate metrics:")
    for k, v in metrics.items():
        print(f"    {k:<14}: {v:.4f}")

    with open(os.path.join(out_dir, "loso_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"subject": subjects, "label": all_labels,
                  "pred": all_preds, "prob": all_probs}).to_csv(
        os.path.join(out_dir, "loso_predictions.csv"), index=False)
    _plot_confusion(all_labels, all_preds, f"{model_key.upper()} — LOSO",
                    os.path.join(out_dir, "cm_loso.png"))
    _plot_roc(all_labels, all_probs, f"{model_key.upper()} — LOSO",
              os.path.join(out_dir, "roc_loso.png"))
    _per_subject_table(all_labels, all_preds, all_probs, subjects,
                       os.path.join(out_dir, "per_subject_loso.csv"),
                       os.path.join(out_dir, "per_subject_loso.png"),
                       f"{model_key.upper()} — Per-subject LOSO accuracy")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
def _plot_training_curves(history, model_name, out_dir):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    epochs = df["epoch"].values

    panels = [
        ("train_loss", "val_loss",        "Loss"),
        ("val_f1",      None,             "F1"),
        ("val_balanced_acc", None,        "Balanced Accuracy"),
        ("val_sensitivity",  None,        "Sensitivity"),
        ("val_specificity",  None,        "Specificity"),
        ("val_auroc",   None,             "AUROC"),
    ]
    for ax, (tk, vk, title) in zip(axes.flat, panels):
        if tk in df.columns:
            ax.plot(epochs, df[tk], label="train", color="steelblue", lw=2)
        if vk and vk in df.columns:
            ax.plot(epochs, df[vk], label="val", color="darkorange", lw=2)
        elif tk.startswith("val_") and tk in df.columns:
            ax.plot(epochs, df[tk], label="val", color="darkorange", lw=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} — Training Curves", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()


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
        pred_lbl = int(preds[m].mean() >= 0.5)
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
    ax.set_xticklabels([f"S{int(s)}" for s in d["subject"]], rotation=45,
                       ha="right", fontsize=7)
    ax.axhline(0.5, color="gray", linestyle="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Window-level accuracy")
    ax.set_title(title)
    ax.legend(handles=[
        Patch(facecolor="steelblue", label="Correct subject-level prediction"),
        Patch(facecolor="tomato",    label="Incorrect subject-level prediction"),
    ], loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wesad_dir", required=True, help="Path to WESAD/ directory")
    parser.add_argument("--ubfc_dir",  required=True, help="Path to ubfc_baselines.npz OR directory of raw bvp_s*.csv files")
    parser.add_argument("--out_root",  default="results/baselines")
    parser.add_argument("--models",    nargs="+", default=["lstm", "cnn1d"])
    parser.add_argument("--stages",    nargs="+", default=["cv", "test", "loso"])
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Models: {args.models}  |  Stages: {args.stages}")

    # Load raw signal data
    print("\nLoading WESAD...")
    Xw, yw, sw = load_wesad(args.wesad_dir)
    print(f"  WESAD: {len(yw)} windows, {len(np.unique(sw))} subjects")

    print("\nLoading UBFC-Phys...")
    Xu, yu, su = load_ubfc(args.ubfc_dir)
    print(f"  UBFC: {len(yu)} windows, {len(np.unique(su))} subjects")

    # Pool
    X = np.concatenate([Xw, Xu])
    y = np.concatenate([yw, yu])
    subjects = np.concatenate([sw, su])
    print(f"\nPooled: {len(y)} windows, {len(np.unique(subjects))} subjects, "
          f"class dist {np.bincount(y)}")

    # Split
    splits = subject_split(X, y, subjects)

    # Pool all for LOSO
    Xall = np.concatenate([splits["train"][0], splits["val"][0], splits["test"][0]])
    yall = np.concatenate([splits["train"][1], splits["val"][1], splits["test"][1]])
    sall = np.concatenate([splits["train"][2], splits["val"][2], splits["test"][2]])

    run_cv   = "cv"   in args.stages
    run_test = "test" in args.stages
    run_loso = "loso" in args.stages

    overall = {}
    for model_key in args.models:
        out_dir = os.path.join(args.out_root, model_key)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n\n########## {model_key.upper()} ##########")

        bp_path = os.path.join(out_dir, "best_params.json")
        if run_cv:
            Xtv = np.concatenate([splits["train"][0], splits["val"][0]])
            ytv = np.concatenate([splits["train"][1], splits["val"][1]])
            stv = np.concatenate([splits["train"][2], splits["val"][2]])
            best_params = cv_sweep(model_key, Xtv, ytv, stv, out_dir)
        elif os.path.exists(bp_path):
            best_params = json.load(open(bp_path))
            print(f"  Loaded best params: {best_params}")
        else:
            best_params = {k: v[0] for k, v in GRID[model_key].items()}
            print(f"  Using grid defaults: {best_params}")

        model_summary = {"best_params": best_params}
        if run_test:
            model_summary["test"] = final_test(model_key, best_params, splits, out_dir)
        if run_loso:
            model_summary["loso"] = loso(model_key, best_params, Xall, yall, sall, out_dir)

        overall[model_key] = model_summary
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(model_summary, f, indent=2)

    with open(os.path.join(args.out_root, "baselines_summary.json"), "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nDone. Results in {args.out_root}")


if __name__ == "__main__":
    main()
