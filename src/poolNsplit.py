"""
pool_then_split.py

Combines full WESAD + UBFC datasets first, then performs one
group-aware train/val/test split across all subjects.

Outputs:
    data_combined/train.npz
    data_combined/val.npz
    data_combined/test.npz

Split:
    train ≈ 70%
    val   ≈ 15%
    test  ≈ 15%

Subject ID namespacing:
    WESAD subjects unchanged
    UBFC subjects offset by +100
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ── CONFIG ────────────────────────────────────────────────────────────────
WESAD_NPZ = "wesad_cwtSINGLE/wesad_datasetSINGLE.npz"
UBFC_NPZ  = "ubfc_cwtSINGLE/ubfc_datasetSINGLE.npz"

OUT_DIR = "data_combined"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 75

# ── LOAD WESAD ────────────────────────────────────────────────────────────
print("Loading WESAD...")
w = np.load(WESAD_NPZ, allow_pickle=True)

X_w = w["X"]
y_w = w["y"]
subjects_w = w["subjects"]
filenames_w = w["filenames"]

# Optional metadata
window_ids_w = w["window_ids"] if "window_ids" in w.files else np.arange(len(y_w))

dataset_w = np.array(["WESAD"] * len(y_w))

# ── LOAD UBFC ─────────────────────────────────────────────────────────────
print("Loading UBFC...")
u = np.load(UBFC_NPZ, allow_pickle=True)

X_u = u["X"]
y_u = u["y"]
subjects_u = u["subjects"] + 100
filenames_u = u["filenames"]

# Optional metadata
trial_ids_u = u["trial_ids"] if "trial_ids" in u.files else np.arange(len(y_u))

dataset_u = np.array(["UBFC"] * len(y_u))

# ── MAKE SHARED ID FIELD ──────────────────────────────────────────────────
# WESAD has window_ids, UBFC has trial_ids, so store both as one generic sample_ids field.
sample_ids_w = np.array([f"wesad_window_{x}" for x in window_ids_w])
sample_ids_u = np.array([f"ubfc_trial_{x}" for x in trial_ids_u])

# ── COMBINE FULL DATASETS ─────────────────────────────────────────────────
X = np.concatenate([X_w, X_u], axis=0)
y = np.concatenate([y_w, y_u], axis=0)
subjects = np.concatenate([subjects_w, subjects_u], axis=0)
filenames = np.concatenate([filenames_w, filenames_u], axis=0)
dataset = np.concatenate([dataset_w, dataset_u], axis=0)
sample_ids = np.concatenate([sample_ids_w, sample_ids_u], axis=0)

print("\nCombined dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Total samples:", len(y))
print("Total subjects:", len(np.unique(subjects)))
print("Class counts:", np.bincount(y.astype(int)))
print("Datasets:", pd.Series(dataset).value_counts().to_dict())

# ── TRAIN+VAL / TEST SPLIT ────────────────────────────────────────────────
gss_test = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=RANDOM_STATE
)

trainval_idx, test_idx = next(
    gss_test.split(X, y, groups=subjects)
)

# ── TRAIN / VAL SPLIT ─────────────────────────────────────────────────────
# 0.1765 of 85% ≈ 15% total validation
gss_val = GroupShuffleSplit(
    n_splits=1,
    test_size=0.1765,
    random_state=RANDOM_STATE
)

train_idx_rel, val_idx_rel = next(
    gss_val.split(
        X[trainval_idx],
        y[trainval_idx],
        groups=subjects[trainval_idx]
    )
)

train_idx = trainval_idx[train_idx_rel]
val_idx = trainval_idx[val_idx_rel]

# ── SAVE FUNCTION ─────────────────────────────────────────────────────────
def save_split(name, idx):
    out_npz = os.path.join(OUT_DIR, f"{name}SINGLE.npz")
    out_csv = os.path.join(OUT_DIR, f"{name}_labelsSINGLE.csv")

    np.savez_compressed(
        out_npz,
        X=X[idx],
        y=y[idx],
        subjects=subjects[idx],
        filenames=filenames[idx],
        dataset=dataset[idx],
        sample_ids=sample_ids[idx],
    )

    pd.DataFrame({
        "filename": filenames[idx],
        "label": y[idx],
        "subject_id": subjects[idx],
        "dataset": dataset[idx],
        "sample_id": sample_ids[idx],
    }).to_csv(out_csv, index=False)
    
    wesad_subjects = len(
        np.unique(subjects[idx][dataset[idx] == "WESAD"])
    )

    ubfc_subjects = len(
        np.unique(subjects[idx][dataset[idx] == "UBFC"])
    )

    print(f"\n{name.upper()}")
    print("samples :", len(idx))
    print("subjects:", len(np.unique(subjects[idx])))
    print("class counts:", np.bincount(y[idx].astype(int)))
    print("datasets:", pd.Series(dataset[idx]).value_counts().to_dict())
    print("subjects by dataset:")
    print(f"  WESAD: {wesad_subjects}")
    print(f"  UBFC : {ubfc_subjects}")
    print("saved:", out_npz)
    print("saved:", out_csv)

# ── LEAKAGE CHECKS ────────────────────────────────────────────────────────
train_subjects = set(subjects[train_idx])
val_subjects = set(subjects[val_idx])
test_subjects = set(subjects[test_idx])

print("\nLeakage checks:")
print("train ∩ val  =", train_subjects & val_subjects)
print("train ∩ test =", train_subjects & test_subjects)
print("val ∩ test   =", val_subjects & test_subjects)

# ── SAVE SPLITS ───────────────────────────────────────────────────────────
save_split("train", train_idx)
save_split("val", val_idx)
save_split("test", test_idx)

print("\nDone.")