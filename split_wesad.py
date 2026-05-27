import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ── LOAD DATASET ──────────────────────────────────────────────────────────────
data = np.load("wesad_cwt/wesad_dataset.npz", allow_pickle=True)

X          = data["X"]
y          = data["y"]
subjects   = data["subjects"]
window_ids = data["window_ids"]
filenames  = data["filenames"]

print("Dataset loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Subjects:", sorted(set(subjects)))

# ── TRAIN+VAL / TEST SPLIT ────────────────────────────────────────────────────
# 15% of subjects for final test set (mirrors split_ubfc2.py)

gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)

trainval_idx, test_idx = next(gss_test.split(X, y, groups=subjects))

X_trainval        = X[trainval_idx]
y_trainval        = y[trainval_idx]
subjects_trainval = subjects[trainval_idx]
window_ids_trainval = window_ids[trainval_idx]
filenames_trainval  = filenames[trainval_idx]

X_test        = X[test_idx]
y_test        = y[test_idx]
subjects_test = subjects[test_idx]
window_ids_test = window_ids[test_idx]
filenames_test  = filenames[test_idx]

# ── TRAIN / VAL SPLIT ─────────────────────────────────────────────────────────
# 0.1765 of remaining 85% ≈ 15% total validation (mirrors split_ubfc2.py)

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)

train_idx, val_idx = next(gss_val.split(X_trainval, y_trainval, groups=subjects_trainval))

X_train        = X_trainval[train_idx]
y_train        = y_trainval[train_idx]
subjects_train = subjects_trainval[train_idx]
window_ids_train = window_ids_trainval[train_idx]
filenames_train  = filenames_trainval[train_idx]

X_val        = X_trainval[val_idx]
y_val        = y_trainval[val_idx]
subjects_val = subjects_trainval[val_idx]
window_ids_val = window_ids_trainval[val_idx]
filenames_val  = filenames_trainval[val_idx]

# ── VERIFY NO SUBJECT LEAKAGE ─────────────────────────────────────────────────
train_subjects = set(subjects_train)
val_subjects   = set(subjects_val)
test_subjects  = set(subjects_test)

print("\nLeakage checks:")
print("train ∩ val  =", train_subjects & val_subjects)
print("train ∩ test =", train_subjects & test_subjects)
print("val ∩ test   =", val_subjects & test_subjects)

# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────
print("\n===== SPLIT SUMMARY =====")

print("\nTRAIN")
print("samples :", len(X_train))
print("subjects:", len(np.unique(subjects_train)), "->", sorted(set(subjects_train)))
print("class counts:", np.bincount(y_train))

print("\nVAL")
print("samples :", len(X_val))
print("subjects:", len(np.unique(subjects_val)), "->", sorted(set(subjects_val)))
print("class counts:", np.bincount(y_val))

print("\nTEST")
print("samples :", len(X_test))
print("subjects:", len(np.unique(subjects_test)), "->", sorted(set(subjects_test)))
print("class counts:", np.bincount(y_test))

# ── SAVE SPLITS ───────────────────────────────────────────────────────────────
np.savez_compressed(
    "wesad_cwt/train.npz",
    X=X_train, y=y_train,
    subjects=subjects_train,
    window_ids=window_ids_train,
    filenames=filenames_train,
)

np.savez_compressed(
    "wesad_cwt/val.npz",
    X=X_val, y=y_val,
    subjects=subjects_val,
    window_ids=window_ids_val,
    filenames=filenames_val,
)

np.savez_compressed(
    "wesad_cwt/test.npz",
    X=X_test, y=y_test,
    subjects=subjects_test,
    window_ids=window_ids_test,
    filenames=filenames_test,
)

print("\nSaved:")
print("  wesad_cwt/train.npz")
print("  wesad_cwt/val.npz")
print("  wesad_cwt/test.npz")
