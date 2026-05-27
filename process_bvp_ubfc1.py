"""
process_bvp_ubfc1.py

UBFC-Phys BVP Processing Pipeline — updated to match WESAD pipeline.
See PROCESSING_README.md for full rationale.

Key change from original: output is (224, 224, 1) float32 grayscale,
NOT (224, 224, 3) viridis-colored. Reason: AudioMAE requires single-channel
input. EfficientNet/CvT loaders expand to 3 channels at training time.

No windowing: each trial CSV is one continuous recording of a single condition
(T1=rest, T2/T3=stress). The entire signal is processed as one scalogram.
This is appropriate because the label is constant for the whole file.

Output format:
  ubfc_cwt/
    cwtFiles/
      bvp_s1_T1.npy    # shape: (224, 224, 1) float32
      ...
    ubfc_labels.csv
    ubfc_dataset.npz   # X, y, filenames, subjects, trial_ids, dataset

Requirements:
    pip install numpy pandas scipy pywavelets scikit-image
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import pywt
from skimage.transform import resize

# CONFIG
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
BVP_DIR   = os.path.join(BASE_DIR, "bvp_ubfc")
OUTPUT_DIR = os.path.join(BASE_DIR, "ubfc_cwt")

FS         = 64.0
IMG_SIZE   = 224
WAVELET    = "morl"
NUM_SCALES = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_bvp(filepath):
    return np.loadtxt(filepath)


def compute_cwt(signal):
    """
    Returns (224, 224, 1) float32, values in [0, 1].
    Single channel: AudioMAE needs 1-channel. EfficientNet/CvT loaders
    expand to 3 channels via np.repeat at training time.
    """
    scales = np.geomspace(1, NUM_SCALES, num=NUM_SCALES)
    coeffs, _ = pywt.cwt(signal, scales, WAVELET, sampling_period=1.0 / FS)

    power = np.abs(coeffs) ** 2
    power = resize(power, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)

    power = power - power.min()
    power = power / (power.max() + 1e-8)

    return power[:, :, np.newaxis].astype(np.float32)   # (224, 224, 1)


def parse_subject_trial(filename):
    basename = os.path.basename(filename)
    match = re.search(r"s(\d+)_T(\d+)", basename)
    if match is None:
        raise ValueError(f"Cannot parse subject/trial from filename: {basename}")
    return int(match.group(1)), int(match.group(2))


def get_label_from_trial(trial_id):
    if trial_id == 1:
        return 0   # rest -> non-stress
    elif trial_id in [2, 3]:
        return 1   # social stress tasks -> stress
    else:
        raise ValueError(f"Unexpected trial_id: T{trial_id}")


def main():
    npy_dir = os.path.join(OUTPUT_DIR, "cwtFiles")
    os.makedirs(npy_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(BVP_DIR, "bvp_s*.csv")))
    if not csv_files:
        print(f"No BVP CSV files found in {BVP_DIR}")
        return

    print(f"Found {len(csv_files)} BVP files\n")

    all_arrays, all_labels, all_names = [], [], []
    all_subjects, all_trial_ids = [], []
    label_rows = []

    for i, filepath in enumerate(csv_files):
        basename   = os.path.splitext(os.path.basename(filepath))[0]
        subject_id, trial_id = parse_subject_trial(filepath)
        label      = get_label_from_trial(trial_id)

        print(f"[{i+1}/{len(csv_files)}] {basename}  subject={subject_id}  "
              f"trial=T{trial_id}  label={label}", end="  ")

        try:
            signal  = load_bvp(filepath)
            cwt_img = compute_cwt(signal)   # (224, 224, 1)

            npy_path = os.path.join(npy_dir, f"{basename}.npy")
            np.save(npy_path, cwt_img)

            all_arrays.append(cwt_img)
            all_labels.append(label)
            all_names.append(basename)
            all_subjects.append(subject_id)
            all_trial_ids.append(trial_id)

            label_rows.append({
                "filename":   basename,
                "label":      label,
                "subject_id": subject_id,
                "trial_id":   trial_id,
                "dataset":    "ubfc",
            })

            print(f"saved  shape={cwt_img.shape}  samples={len(signal)}")

        except Exception as e:
            print(f"ERROR: {e}")

    X          = np.stack(all_arrays)
    y          = np.array(all_labels,    dtype=np.int64)
    filenames  = np.array(all_names,     dtype=str)
    subjects   = np.array(all_subjects,  dtype=np.int64)
    trial_ids  = np.array(all_trial_ids, dtype=np.int64)
    dataset    = np.array(["ubfc"] * len(y), dtype=str)

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "ubfc_dataset.npz"),
        X=X, y=y, filenames=filenames,
        subjects=subjects, trial_ids=trial_ids, dataset=dataset,
    )
    print(f"\nSaved ubfc_dataset.npz -> X: {X.shape}  stress={y.sum()}  non_stress={(y==0).sum()}")

    pd.DataFrame(label_rows).to_csv(
        os.path.join(OUTPUT_DIR, "ubfc_labels.csv"), index=False
    )
    print(f"Saved ubfc_labels.csv")


if __name__ == "__main__":
    main()
