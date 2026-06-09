"""
moment_baseline.py
==================
MOMENT (AutonLab/MOMENT-1-base) baseline for binary stress classification
from raw BVP signal windows.

MOMENT expects input of shape (B, n_channels, seq_len) where seq_len=512.
Our windows are 1920 samples (30s at 64 Hz). We take the first 512 samples
(8 seconds) as input — this captures the most stable part of the window and
is within MOMENT's native context length.

Alternatively, we downsample 1920 -> 512 (factor ~3.75) to preserve the
full 30s window structure at lower resolution.

We use downsampling (scipy.signal.resample) to preserve full-window context.

Usage:
  python moment_baseline.py --wesad_dir /path/to/WESAD --ubfc_dir /path/to/ubfc_baselines.npz
  python moment_baseline.py --wesad_dir /data/WESAD --ubfc_dir /data/ubfc_baselines.npz --out_root /data/results/baselines

Reference: Goswami et al. (2024) MOMENT: A Family of Open Time-series Foundation Models. ICML 2024.
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report,
)
from scipy.signal import resample, resample_poly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
FS           = 64.0
LABEL_FS     = 700
WINDOW_SAMP  = 1920   # 30s at 64 Hz
MOMENT_SEQ   = 512    # MOMENT native context length
MAJORITY_THRESHOLD = 0.80
STRESS_LABELS      = {2}
NON_STRESS_LABELS  = {1, 3}
VALID_WESAD        = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
UBFC_OFFSET        = 100

CV_FOLDS  = 5
SEED      = 91
CLASS_NAMES = ["No Stress", "Stress"]

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

torch.manual_seed(SEED)
np.random.seed(SEED)

GRID = {
    "lr":           [1e-4, 3e-4],
    "batch_size":   [16, 32],
    "weight_decay": [1e-5, 1e-4],
    "epochs":       [20],
}

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING (reuse from baselines.py)
# ══════════════════════════════════════════════════════════════════════════════
def _majority_label(label_window):
    unique, counts = np.unique(label_window, return_counts=True)
    best = np.argmax(counts)
    if counts[best] / len(label_window) >= MAJORITY_THRESHOLD:
        return int(unique[best])
    return -1


def load_wesad(wesad_dir):
    all_X, all_y, all_s = [], [], []
    for sid in VALID_WESAD:
        pkl = os.path.join(wesad_dir, f"S{sid}", f"S{sid}.pkl")
        if not os.path.exists(pkl):
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
        for i in range(n // WINDOW_SAMP):
            s, e = i * WINDOW_SAMP, (i + 1) * WINDOW_SAMP
            maj = _majority_label(labels64[s:e])
            if maj in STRESS_LABELS:
                all_X.append(bvp[s:e]); all_y.append(1); all_s.append(sid)
            elif maj in NON_STRESS_LABELS:
                all_X.append(bvp[s:e]); all_y.append(0); all_s.append(sid)
        print(f"  WESAD S{sid}: done")
    return (np.array(all_X, dtype=np.float32),
            np.array(all_y, dtype=np.int64),
            np.array(all_s, dtype=np.int64))


def load_ubfc(ubfc_path):
    if ubfc_path.endswith(".npz") and os.path.isfile(ubfc_path):
        data = np.load(ubfc_path)
        X, y, subjects = data["X"], data["y"], data["subjects"]
        print(f"  UBFC (npz): {len(y)} windows")
        return X.astype(np.float32), y.astype(np.int64), subjects.astype(np.int64)
    import glob, re
    all_X, all_y, all_s = [], [], []
    for fpath in sorted(glob.glob(os.path.join(ubfc_path, "bvp_s*.csv"))):
        match = re.search(r"s(\d+)_T(\d+)", os.path.basename(fpath))
        if not match: continue
        sid, trial = int(match.group(1)), int(match.group(2))
        label  = 0 if trial == 1 else 1
        signal = np.loadtxt(fpath).astype(np.float32)
        for i in range(len(signal) // WINDOW_SAMP):
            s, e = i * WINDOW_SAMP, (i + 1) * WINDOW_SAMP
            all_X.append(signal[s:e]); all_y.append(label); all_s.append(sid + UBFC_OFFSET)
    return (np.array(all_X, dtype=np.float32),
            np.array(all_y, dtype=np.int64),
            np.array(all_s, dtype=np.int64))


def downsample_to_moment(X):
    """Downsample (N, 1920) -> (N, 512) preserving full-window context."""
    return resample(X, MOMENT_SEQ, axis=1).astype(np.float32)


def normalize(X):
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sigma


def subject_split(X, y, subjects, test_size=0.15, val_size=0.1765, seed=SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tv, te = next(gss.split(X, y, groups=subjects))
    X_tv, y_tv, s_tv = X[tv], y[tv], subjects[tv]
    X_te, y_te, s_te = X[te], y[te], subjects[te]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr, va = next(gss2.split(X_tv, y_tv, groups=s_tv))
    splits = {
        "train": (X_tv[tr], y_tv[tr], s_tv[tr]),
        "val":   (X_tv[va], y_tv[va], s_tv[va]),
        "test":  (X_te, y_te, s_te),
    }
    print("\n===== SPLIT SUMMARY =====")
    for name, (Xs, ys, ss) in splits.items():
        print(f"  {name}: {len(ys)} windows, {len(np.unique(ss))} subjects, "
              f"class dist {np.bincount(ys)}")
    return splits


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════
def build_moment(dropout=0.1):
    """Load MOMENT-1-base in classification mode (frozen encoder + linear head)."""
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-base",
        model_kwargs={
            "task_name": "classification",
            "n_channels": 1,
            "num_class": 2,
            "dropout": dropout,
        },
    )
    model.init()
    # Encoder is frozen by default — only classification head is trained
    return model.to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def make_loader(X, y, batch_size, shuffle):
    # MOMENT expects (B, n_channels, seq_len) — add channel dim
    Xt = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32)  # (N, 1, 512)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)


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
        out = model(x_enc=xb)
        loss = criterion(out.logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, probs, labels = [], [], []
    for xb, yb in loader:
        out = model(x_enc=xb.to(DEVICE))
        p = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
        probs.extend(p)
        preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
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
    # Only train classification head (encoder frozen by default)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    best_f1, best_state, no_improve = -1.0, None, 0
    history = []

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, criterion)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(x_enc=xb)
                val_loss += criterion(out.logits, yb).item() * xb.size(0)
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
                if verbose: print(f"    early stop @ epoch {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1, history


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STAGES
# ══════════════════════════════════════════════════════════════════════════════
def cv_sweep(X, y, subjects, out_dir):
    print(f"\n{'='*70}\n  [MOMENT] STAGE 1 — CV sweep\n{'='*70}")
    Xd = downsample_to_moment(normalize(X))
    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    k = min(CV_FOLDS, len(np.unique(subjects)))
    gkf = GroupKFold(n_splits=k)

    rows = []
    for ci, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"\n  Combo {ci}/{len(combos)}: {params}")
        for fold, (tr, va) in enumerate(gkf.split(Xd, y, subjects), 1):
            train_loader = make_loader(Xd[tr], y[tr], int(params["batch_size"]), True)
            val_loader   = make_loader(Xd[va], y[va], int(params["batch_size"]), False)
            criterion    = class_weighted_criterion(y[tr])
            model = build_moment()
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
    return best_params


def final_test(best_params, splits, out_dir):
    print(f"\n{'='*70}\n  [MOMENT] STAGE 2 — final test eval\n{'='*70}")
    Xtr, ytr, _ = splits["train"]
    Xva, yva, _ = splits["val"]
    Xte, yte, ste = splits["test"]

    Xdev = downsample_to_moment(normalize(np.concatenate([Xtr, Xva])))
    ydev = np.concatenate([ytr, yva])
    Xva_d = downsample_to_moment(normalize(Xva))
    Xte_d = downsample_to_moment(normalize(Xte))

    bs = int(best_params["batch_size"])
    train_loader = make_loader(Xdev, ydev, bs, True)
    val_loader   = make_loader(Xva_d, yva, bs, False)
    test_loader  = make_loader(Xte_d, yte, bs, False)
    criterion    = class_weighted_criterion(ydev)

    model = build_moment()
    model, _, history = fit(
        model, train_loader, val_loader, criterion,
        lr=float(best_params["lr"]), epochs=int(best_params["epochs"]),
        weight_decay=float(best_params["weight_decay"]),
        patience=10, verbose=True,
    )
    torch.save(model.state_dict(), os.path.join(out_dir, "final_model.pt"))

    if history:
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
        _plot_training_curves(history, "MOMENT", out_dir)

    yl, yp, pr = predict(model, test_loader)
    metrics = compute_metrics(yl, yp, pr)
    print("\n  TEST metrics:")
    for k, v in metrics.items():
        print(f"    {k:<14}: {v:.4f}")
    print(classification_report(yl, yp, target_names=CLASS_NAMES, zero_division=0))

    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _plot_confusion(yl, yp, "MOMENT — Test", os.path.join(out_dir, "cm_test.png"))
    _plot_roc(yl, pr, "MOMENT — Test", os.path.join(out_dir, "roc_test.png"))
    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return metrics


def loso(best_params, X, y, subjects, out_dir):
    print(f"\n{'='*70}\n  [MOMENT] STAGE 3 — LOSO\n{'='*70}")
    Xd   = downsample_to_moment(normalize(X))
    uniq = np.unique(subjects)
    bs   = int(best_params["batch_size"])

    all_labels = np.empty(len(y), dtype=np.int64)
    all_preds  = np.empty(len(y), dtype=np.int64)
    all_probs  = np.empty(len(y), dtype=np.float32)

    for i, s in enumerate(uniq, 1):
        te = subjects == s
        tr = ~te
        print(f"  Fold {i}/{len(uniq)}  hold-out S{s}")
        train_loader = make_loader(Xd[tr], y[tr], bs, True)
        test_loader  = make_loader(Xd[te], y[te], bs, False)
        criterion    = class_weighted_criterion(y[tr])
        model = build_moment()
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
    _plot_confusion(all_labels, all_preds, "MOMENT — LOSO",
                    os.path.join(out_dir, "cm_loso.png"))
    _plot_roc(all_labels, all_probs, "MOMENT — LOSO",
              os.path.join(out_dir, "roc_loso.png"))
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
def _plot_training_curves(history, model_name, out_dir):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    epochs = df["epoch"].values
    panels = [
        ("train_loss", "val_loss",       "Loss"),
        ("val_f1",      None,            "F1"),
        ("val_balanced_acc", None,       "Balanced Accuracy"),
        ("val_sensitivity",  None,       "Sensitivity"),
        ("val_specificity",  None,       "Specificity"),
        ("val_auroc",   None,            "AUROC"),
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
    if len(np.unique(labels)) < 2: return
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, lw=2, label=f"AUROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wesad_dir", required=True)
    parser.add_argument("--ubfc_dir",  required=True,
                        help="Path to ubfc_baselines.npz OR raw bvp_s*.csv directory")
    parser.add_argument("--out_root",  default="results/baselines/moment")
    parser.add_argument("--stages",    nargs="+", default=["cv", "test", "loso"])
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"MOMENT input: 1920 samples downsampled to {MOMENT_SEQ}")

    print("\nLoading WESAD...")
    Xw, yw, sw = load_wesad(args.wesad_dir)
    print("\nLoading UBFC-Phys...")
    Xu, yu, su = load_ubfc(args.ubfc_dir)

    X = np.concatenate([Xw, Xu])
    y = np.concatenate([yw, yu])
    subjects = np.concatenate([sw, su])
    print(f"\nPooled: {len(y)} windows, {len(np.unique(subjects))} subjects, "
          f"class dist {np.bincount(y)}")

    splits = subject_split(X, y, subjects)
    Xall = np.concatenate([splits["train"][0], splits["val"][0], splits["test"][0]])
    yall = np.concatenate([splits["train"][1], splits["val"][1], splits["test"][1]])
    sall = np.concatenate([splits["train"][2], splits["val"][2], splits["test"][2]])

    bp_path = os.path.join(args.out_root, "best_params.json")
    if "cv" in args.stages:
        Xtv = np.concatenate([splits["train"][0], splits["val"][0]])
        ytv = np.concatenate([splits["train"][1], splits["val"][1]])
        stv = np.concatenate([splits["train"][2], splits["val"][2]])
        best_params = cv_sweep(Xtv, ytv, stv, args.out_root)
    elif os.path.exists(bp_path):
        best_params = json.load(open(bp_path))
        print(f"  Loaded best params: {best_params}")
    else:
        best_params = {k: v[0] for k, v in GRID.items()}

    summary = {"best_params": best_params}
    if "test" in args.stages:
        summary["test"] = final_test(best_params, splits, args.out_root)
    if "loso" in args.stages:
        summary["loso"] = loso(best_params, Xall, yall, sall, args.out_root)

    with open(os.path.join(args.out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Results in {args.out_root}")


if __name__ == "__main__":
    main()
