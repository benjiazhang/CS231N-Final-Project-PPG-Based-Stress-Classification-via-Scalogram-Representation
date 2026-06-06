"""
process_bvp_wesad.py

Processes WESAD wrist BVP data into CWT scalogram images for binary stress
classification. See PROCESSING_README.md for full rationale of every decision.

Key design choices (see README for justification):
  - Output: (224, 224, 1) float32, normalized to [0, 1]
  - Grayscale (1 channel): AudioMAE requires single-channel input. EfficientNet
    and CvT loaders expand to 3 channels via np.repeat at training time.
  - 224x224: highest model requirement; AudioMAE loader downsamples to 128x128.
  - 30s non-overlapping windows (1920 samples at 64 Hz).
  - Window accepted only if >=80% of labels agree (drops transition windows).
  - Morlet wavelet, geomspace(1, 224, 224) scales — matches UBFC pipeline.
  - Per-window min-max normalization to [0, 1].

Label mapping (binary, per Schmidt et al. 2018):
  stress (1)     <- WESAD label 2
  non-stress (0) <- WESAD labels 1 (baseline) + 3 (amusement)
  discarded      <- 0, 4, 5, 6, 7

Usage:
  python process_bvp_wesad.py --wesad_dir /path/to/WESAD --out_dir ./wesad_cwt

Output:
  wesad_cwt/
    cwtFiles/
      bvp_S2_stress_0000.npy       # shape: (224, 224, 1) float32
      bvp_S2_non_stress_0000.npy
      ...
    wesad_labels.csv
    wesad_dataset.npz              # X, y, filenames, subjects, window_ids, dataset
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import pywt
from skimage.transform import resize
from scipy.signal import resample_poly
from math import gcd

# CONFIG
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

FS          = 64.0
LABEL_FS    = 700
WINDOW_SEC  = 30
WINDOW_SAMP = int(FS * WINDOW_SEC)   # 1920

IMG_SIZE    = 224
WAVELET     = "morl"
NUM_SCALES  = 224

MAJORITY_THRESHOLD = 0.80

STRESS_LABELS     = {2}
NON_STRESS_LABELS = {1, 3}

# S1 and S12 excluded due to sensor malfunction (per WESAD readme)
VALID_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]


def load_subject(wesad_dir, sid):
    pkl_path = os.path.join(wesad_dir, f"S{sid}", f"S{sid}.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    bvp       = data["signal"]["wrist"]["BVP"].flatten().astype(np.float32)
    labels700 = data["label"].flatten().astype(np.int32)

    # Resample labels 700 Hz -> 64 Hz
    g = gcd(int(LABEL_FS), int(FS))
    up, down = int(FS) // g, int(LABEL_FS) // g
    labels64 = resample_poly(labels700.astype(np.float64), up, down)
    labels64  = np.round(labels64).astype(np.int32)

    min_len = min(len(bvp), len(labels64))
    return bvp[:min_len], labels64[:min_len]


def majority_label(label_window):
    unique, counts = np.unique(label_window, return_counts=True)
    best_idx = np.argmax(counts)
    if counts[best_idx] / len(label_window) >= MAJORITY_THRESHOLD:
        return int(unique[best_idx])
    return -1


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


def process_subject(sid, wesad_dir, npy_dir):
    pkl_path = os.path.join(wesad_dir, f"S{sid}", f"S{sid}.pkl")
    if not os.path.exists(pkl_path):
        print(f"  [skip] S{sid}: not found")
        return [], [], [], [], []

    bvp, labels = load_subject(wesad_dir, sid)
    n_windows   = len(bvp) // WINDOW_SAMP

    arrays, ys, names, subjects, window_ids = [], [], [], [], []
    stress_ct = nonstress_ct = skipped_ct = 0

    for i in range(n_windows):
        start = i * WINDOW_SAMP
        end   = start + WINDOW_SAMP
        maj   = majority_label(labels[start:end])

        if maj in STRESS_LABELS:
            y_val = 1; win_name = f"bvp_S{sid}_stress_{stress_ct:04d}"; stress_ct += 1
        elif maj in NON_STRESS_LABELS:
            y_val = 0; win_name = f"bvp_S{sid}_non_stress_{nonstress_ct:04d}"; nonstress_ct += 1
        else:
            skipped_ct += 1; continue

        cwt_img  = compute_cwt(bvp[start:end])
        np.save(os.path.join(npy_dir, f"{win_name}.npy"), cwt_img)

        arrays.append(cwt_img); ys.append(y_val); names.append(win_name)
        subjects.append(sid); window_ids.append(i)

    print(f"  S{sid}: stress={stress_ct}  non_stress={nonstress_ct}  skipped={skipped_ct}")
    return arrays, ys, names, subjects, window_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wesad_dir", default=os.path.join(BASE_DIR, "WESAD"))
    parser.add_argument("--out_dir",   default=os.path.join(BASE_DIR, "wesad_cwt"))
    parser.add_argument("--subjects",  nargs="+", type=int, default=VALID_SUBJECTS)
    args = parser.parse_args()

    npy_dir = os.path.join(args.out_dir, "cwtFiles")
    os.makedirs(npy_dir, exist_ok=True)

    all_arrays, all_y, all_names, all_subjects, all_wids = [], [], [], [], []

    for sid in args.subjects:
        print(f"Processing S{sid}...")
        a, y, n, s, w = process_subject(sid, args.wesad_dir, npy_dir)
        all_arrays.extend(a); all_y.extend(y); all_names.extend(n)
        all_subjects.extend(s); all_wids.extend(w)

    X          = np.stack(all_arrays)
    y          = np.array(all_y,        dtype=np.int64)
    filenames  = np.array(all_names,    dtype=str)
    subjects   = np.array(all_subjects, dtype=np.int64)
    window_ids = np.array(all_wids,     dtype=np.int64)
    dataset    = np.array(["wesad"] * len(y), dtype=str)

    np.savez_compressed(
        os.path.join(args.out_dir, "wesad_dataset.npz"),
        X=X, y=y, filenames=filenames,
        subjects=subjects, window_ids=window_ids, dataset=dataset,
    )
    print(f"\nSaved wesad_dataset.npz -> X: {X.shape}  stress={y.sum()}  non_stress={(y==0).sum()}")

    pd.DataFrame({
        "filename": filenames, "label": y,
        "subject_id": subjects, "window_id": window_ids, "dataset": dataset,
    }).to_csv(os.path.join(args.out_dir, "wesad_labels.csv"), index=False)
    print(f"Saved wesad_labels.csv")


if __name__ == "__main__":
    main()
