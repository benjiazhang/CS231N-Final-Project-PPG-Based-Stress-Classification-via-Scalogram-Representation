"""
UBFC-Phys BVP Processing Pipeline
-----------------------------------
Reads BVP CSVs, applies CWT, saves 224x224x3 numpy arrays + labels.

Output format:
  - ubfc_cwt/
      bvp_s1_T1.npy   # shape: (224, 224, 3)
      bvp_s1_T2.npy
      ...
  - ubfc_labels.csv
      filename, label, subject_id, trial_id
  - ubfc_dataset.npz
      X: (N, 224, 224, 3)
      y: (N,)
      filenames: (N,)
      subjects: (N,)
      trial_ids: (N,)

Requirements:
    pip install numpy pandas scipy pywavelets scikit-image
"""

import os
import glob
import re  # NEW
import numpy as np
import pandas as pd
import pywt
from skimage.transform import resize
import matplotlib.pyplot as plt

# ── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

BVP_DIR    = os.path.join(BASE_DIR, "bvp_ubfc")
OUTPUT_DIR = os.path.join(BASE_DIR, "ubfc_cwt")
FS         = 64.0
IMG_SIZE   = 224
WAVELET    = "morl"
NUM_SCALES = 224
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_bvp(filepath):
    """Just raw float values, one per line."""
    signal = np.loadtxt(filepath)
    return signal


def compute_cwt(signal, fs=FS, num_scales=NUM_SCALES, img_size=IMG_SIZE):
    scales = np.geomspace(1, num_scales, num=num_scales)
    coeffs, _ = pywt.cwt(signal, scales, WAVELET, sampling_period=1.0 / fs)

    power = np.abs(coeffs) ** 2

    power = resize(power, (img_size, img_size), anti_aliasing=True)

    power = power - power.min()
    power = power / (power.max() + 1e-8)

    colored = plt.cm.viridis(power)[:, :, :3]

    return colored.astype(np.float32)


def parse_subject_trial(filename):  # NEW
    """
    Extract subject_id and trial_id from filenames like:
      bvp_s01_T1.csv
      bvp_s1_T2.csv
      bvp_s23_T3.csv
    """
    basename = os.path.basename(filename)

    match = re.search(r"s(\d+)_T(\d+)", basename)
    if match is None:
        raise ValueError(f"Cannot parse subject/trial from filename: {basename}")

    subject_id = int(match.group(1))
    trial_id = int(match.group(2))

    return subject_id, trial_id


def get_label_from_trial(trial_id):  # CHANGED
    """
    T1 = 0 (no stress)
    T2 or T3 = 1 (stress)
    """
    if trial_id == 1:
        return 0
    elif trial_id in [2, 3]:
        return 1
    else:
        raise ValueError(f"Cannot determine label from trial_id: T{trial_id}")


def main():
    csv_files = sorted(glob.glob(os.path.join(BVP_DIR, "bvp_s*.csv")))
    if not csv_files:
        print(f"No BVP CSV files found in {BVP_DIR}")
        return

    print(f"Found {len(csv_files)} BVP files\n")

    all_arrays = []
    all_labels = []
    all_names  = []
    all_subjects = []   # NEW
    all_trial_ids = []  # NEW
    label_rows = []

    for i, filepath in enumerate(csv_files):
        basename = os.path.splitext(os.path.basename(filepath))[0]

        subject_id, trial_id = parse_subject_trial(filepath)  # NEW
        label = get_label_from_trial(trial_id)                # CHANGED

        print(
            f"[{i+1}/{len(csv_files)}] {basename}  "
            f"subject={subject_id}  trial=T{trial_id}  label={label}",
            end="  "
        )

        try:
            signal  = load_bvp(filepath)
            cwt_img = compute_cwt(signal)

            npy_path = os.path.join(OUTPUT_DIR, f"cwtFiles/{basename}.npy")
            np.save(npy_path, cwt_img)

            all_arrays.append(cwt_img)
            all_labels.append(label)
            all_names.append(basename)
            all_subjects.append(subject_id)    # NEW
            all_trial_ids.append(trial_id)     # NEW

            label_rows.append({                # CHANGED
                "filename": basename,
                "label": label,
                "subject_id": subject_id,
                "trial_id": trial_id,
            })

            print(f"saved  shape={cwt_img.shape}  samples={len(signal)}")

        except Exception as e:
            print(f"ERROR: {e}")

    # ── Save combined dataset.npz ─────────────────────────────────────────────
    X = np.stack(all_arrays)
    y = np.array(all_labels)
    subjects = np.array(all_subjects)      # NEW
    trial_ids = np.array(all_trial_ids)    # NEW
    filenames = np.array(all_names)        # NEW

    npz_path = os.path.join(OUTPUT_DIR, "ubfc_dataset.npz")

    np.savez_compressed(                   # CHANGED
        npz_path,
        X=X,
        y=y,
        filenames=filenames,
        subjects=subjects,
        trial_ids=trial_ids,
    )

    print(f"\nSaved dataset.npz  →  X: {X.shape}, y: {y.shape}")
    print(f"subjects: {subjects.shape}, trial_ids: {trial_ids.shape}")
    print(f"Stress (1): {y.sum()}  |  No-stress (0): {(y == 0).sum()}")

    # ── Save labels.csv ───────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "ubfc_labels.csv")
    pd.DataFrame(label_rows).to_csv(csv_path, index=False)

    print(f"Saved labels.csv  →  {csv_path}")
    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()