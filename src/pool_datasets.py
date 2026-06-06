"""
pool_splits.py

Pools WESAD and UBFC train/val/test splits separately.

Outputs:
    data/trainSINGLE.npz
    data/valSINGLE.npz
    data/testSINGLE.npz

Subject ID namespacing:
    WESAD: unchanged
    UBFC: +100 offset
"""

import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)

SPLITS = ["trainSINGLE", "valSINGLE", "testSINGLE"]

WESAD_DIR = "wesad_cwtSINGLE"
UBFC_DIR = "ubfc_cwtSINGLE"


def pool_split(split_name):
    wesad_file = os.path.join(WESAD_DIR, f"wesad_{split_name}.npz")
    ubfc_file = os.path.join(UBFC_DIR, f"ubfc_{split_name}.npz")

    print(f"\n=== {split_name} ===")

    print(f"Loading {wesad_file}")
    w = np.load(wesad_file, allow_pickle=True)

    print(f"Loading {ubfc_file}")
    u = np.load(ubfc_file, allow_pickle=True)

    X_w = w["X"]
    y_w = w["y"]
    subj_w = w["subjects"]
    files_w = w["filenames"]
    #dataset_w = w["dataset"]

    X_u = u["X"]
    y_u = u["y"]
    subj_u = u["subjects"] + 100
    files_u = u["filenames"]
    #dataset_u = u["dataset"]

    X = np.concatenate([X_w, X_u], axis=0)
    y = np.concatenate([y_w, y_u], axis=0)
    subjects = np.concatenate([subj_w, subj_u], axis=0)
    filenames = np.concatenate([files_w, files_u], axis=0)
    #dataset = np.concatenate([dataset_w, dataset_u], axis=0)

    print(f"Samples         : {len(y)}")
    print(f"Subjects        : {len(np.unique(subjects))}")
    print(f"Stress (1)      : {y.sum()}")
    print(f"Non-stress (0)  : {(y == 0).sum()}")
    print(f"X shape         : {X.shape}")

    out_npz = os.path.join("data", f"{split_name}.npz")
    out_csv = os.path.join("data", f"{split_name}_labels.csv")

    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        subjects=subjects,
        filenames=filenames,
        #dataset=dataset,
    )

    pd.DataFrame({
        "filename": filenames,
        "label": y,
        "subject_id": subjects,
        #"dataset": dataset,
    }).to_csv(out_csv, index=False)

    print(f"Saved {out_npz}")
    print(f"Saved {out_csv}")



if __name__ == "__main__":

  for split in SPLITS:
    pool_split(split)

  print("\nDone.")


# """
# pool_datasets.py

# Merges wesad_dataset.npz and ubfc_dataset.npz into a single pooled_dataset.npz
# for LOSO cross-validation across all 71 subjects.

# Subject ID namespacing:
#   WESAD subjects:  S2-S17  -> kept as-is (2-17)
#   UBFC subjects:   s1-s56  -> offset by 100 (101-156)
#   This avoids collisions in the pooled subject ID column used for LOSO grouping.

# Run after both processing scripts have completed.
# """

# import numpy as np
# import pandas as pd
# import os

# WESAD_NPZ = "wesad_cwt/wesad_dataset.npz"
# UBFC_NPZ  = "ubfc_cwt/ubfc_dataset.npz"
# OUT_NPZ   = "data/pooled_dataset.npz"
# OUT_CSV   = "data/pooled_labels.csv"

# os.makedirs("data", exist_ok=True)

# print("Loading WESAD...")
# w = np.load(WESAD_NPZ, allow_pickle=True)
# X_w       = w["X"]
# y_w       = w["y"]
# subj_w    = w["subjects"]
# files_w   = w["filenames"]
# dataset_w = w["dataset"]

# print("Loading UBFC...")
# u = np.load(UBFC_NPZ, allow_pickle=True)
# X_u       = u["X"]
# y_u       = u["y"]
# subj_u    = u["subjects"] + 100   # offset to avoid collision with WESAD IDs
# files_u   = u["filenames"]
# dataset_u = u["dataset"]

# X        = np.concatenate([X_w, X_u],       axis=0)
# y        = np.concatenate([y_w, y_u],       axis=0)
# subjects = np.concatenate([subj_w, subj_u], axis=0)
# files    = np.concatenate([files_w, files_u], axis=0)
# dataset  = np.concatenate([dataset_w, dataset_u], axis=0)

# print(f"\nPooled dataset:")
# print(f"  Total samples : {len(y)}")
# print(f"  Total subjects: {len(np.unique(subjects))}  "
#       f"(WESAD: {len(np.unique(subj_w))}, UBFC: {len(np.unique(u['subjects']))})")
# print(f"  Stress (1)    : {y.sum()}")
# print(f"  Non-stress (0): {(y==0).sum()}")
# print(f"  X shape       : {X.shape}")

# np.savez_compressed(
#     OUT_NPZ,
#     X=X, y=y, subjects=subjects, filenames=files, dataset=dataset,
# )
# print(f"\nSaved {OUT_NPZ}")

# pd.DataFrame({
#     "filename":   files,
#     "label":      y,
#     "subject_id": subjects,
#     "dataset":    dataset,
# }).to_csv(OUT_CSV, index=False)
# print(f"Saved {OUT_CSV}")
