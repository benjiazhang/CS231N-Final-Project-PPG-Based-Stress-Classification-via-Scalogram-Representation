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


TRAIN_PATH = "data/ubfc_train.npz"
VAL_PATH = "data/ubfc_val.npz"
OUT_DIR = "results/efficientnet_crossval"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_FRACTION = 0.8

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


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


def load_split(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    filenames = data["filenames"].astype(str)
    subjects = data["subjects"].astype(np.int64)
    trial_ids = data["trial_ids"].astype(np.int64)

    if X.shape[-1] == 3:
        X = np.transpose(X, (0, 3, 1, 2))

    return X, y, filenames, subjects, trial_ids


def load_trainval():
    X_train, y_train, f_train, s_train, t_train = load_split(TRAIN_PATH)
    X_val, y_val, f_val, s_val, t_val = load_split(VAL_PATH)

    X = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)
    filenames = np.concatenate([f_train, f_val], axis=0)
    subjects = np.concatenate([s_train, s_val], axis=0)
    trial_ids = np.concatenate([t_train, t_val], axis=0)

    return X, y, filenames, subjects, trial_ids


def apply_subject_fraction(X, y, filenames, subjects, trial_ids, fraction):
    if fraction >= 1.0:
        return X, y, filenames, subjects, trial_ids

    rng = np.random.default_rng(91)
    unique_subjects = np.unique(subjects)

    n_subjects = max(2, int(len(unique_subjects) * fraction))
    chosen_subjects = rng.choice(unique_subjects, size=n_subjects, replace=False)

    mask = np.isin(subjects, chosen_subjects)

    print(f"Using {mask.sum()} samples from {n_subjects}/{len(unique_subjects)} subjects")

    return X[mask], y[mask], filenames[mask], subjects[mask], trial_ids[mask]


def normalize_imagenet(X):
    return (X - IMAGENET_MEAN) / IMAGENET_STD


def make_loaders(X, y, train_idx, val_idx, batch_size):
    X_train = normalize_imagenet(X[train_idx])
    y_train = y[train_idx]

    X_val = normalize_imagenet(X[val_idx])
    y_val = y[val_idx]

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    preds = []
    labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)

            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(yb.cpu().numpy())

    return {
        "loss": total_loss / len(loader.dataset),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "balanced_acc": balanced_accuracy_score(labels, preds),
    }


def main():
    print(f"Using device: {DEVICE}")

    X, y, filenames, subjects, trial_ids = load_trainval()

    X, y, filenames, subjects, trial_ids = apply_subject_fraction(
        X, y, filenames, subjects, trial_ids, DATA_FRACTION
    )

    param_grid = {
        "lr": [1e-4, 3e-4],
        "batch_size": [8, 16],
        "weight_decay": [1e-4],
        "dropout": [0.2, 0.3],
        "epochs": [10],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    n_splits = min(5, len(np.unique(subjects)))
    group_kfold = GroupKFold(n_splits=n_splits)

    results = []

    for combo in combos:
        params = dict(zip(keys, combo))
        print(f"\nTesting params: {params}")

        for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, subjects)):
            print(f"\nFold {fold + 1}/{n_splits}")

            train_loader, val_loader = make_loaders(
                X,
                y,
                train_idx,
                val_idx,
                batch_size=int(params["batch_size"]),
            )

            model = EfficientNetClassifier(
                num_classes=2,
                dropout=float(params["dropout"]),
                pretrained=True,
            ).to(DEVICE)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(params["lr"]),
                weight_decay=float(params["weight_decay"]),
            )

            class_counts = np.bincount(y[train_idx], minlength=2)
            class_weights = len(y[train_idx]) / (2 * np.maximum(class_counts, 1))

            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
            )

            best_f1 = -1.0
            best_metrics = None

            for epoch in range(int(params["epochs"])):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
                val_metrics = evaluate(model, val_loader, criterion)

                print(
                    f"Epoch {epoch + 1}/{params['epochs']} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_acc={val_metrics['acc']:.4f} | "
                    f"val_f1={val_metrics['f1']:.4f}"
                )

                if val_metrics["f1"] > best_f1:
                    best_f1 = val_metrics["f1"]
                    best_metrics = val_metrics

            results.append({
                **params,
                "fold": fold,
                **best_metrics,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "ubfc_crossval_results.csv"), index=False)

    summary = (
        results_df
        .groupby(keys)
        [["loss", "acc", "f1", "balanced_acc"]]
        .mean()
        .reset_index()
        .sort_values("f1", ascending=False)
    )

    summary.to_csv(os.path.join(OUT_DIR, "cv_summary.csv"), index=False)

    best_params = summary.iloc[0][keys].to_dict()

    with open(os.path.join(OUT_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    print("\nBest params:")
    print(best_params)


if __name__ == "__main__":
    main()