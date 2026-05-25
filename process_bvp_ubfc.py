"""
UBFC-Phys BVP Processing Pipeline
-----------------------------------
Reads BVP CSVs, applies CWT, saves 224x224x3 numpy arrays + labels.

Output format:
  - ubfc_cwt/
      bvp_s1_T1.npy   # shape: (224, 224, 3)
      bvp_s1_T2.npy
      ...
  - ubfc_labels.csv         # filename, label (0=no stress, 1=stress)
  - ubfc_dataset.npz        # X: (N, 224, 224, 3), y: (N,), filenames: (N,)

Requirements:
    pip install numpy pandas scipy pywavelets scikit-image
"""

import os
import glob
import numpy as np
import pandas as pd
import pywt
from skimage.transform import resize
import matplotlib.pyplot as plt

# ── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

BVP_DIR    = os.path.join(BASE_DIR, "bvp_ubfc")
OUTPUT_DIR = os.path.join(BASE_DIR, "ubfc_cwt")
FS         = 64.0                                # Empatica E4 BVP sample rate
IMG_SIZE   = 224                                 # CWT image size
WAVELET    = "morl"                              # Morlet wavelet
NUM_SCALES = 224                                 # number of CWT scales
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

    # resize first
    power = resize(power, (img_size, img_size), anti_aliasing=True)

    # normalize to [0, 1]
    power = power - power.min()
    power = power / (power.max() + 1e-8)

    # apply viridis colormap: RGBA -> RGB
    colored = plt.cm.viridis(power)[:, :, :3]

    return colored.astype(np.float32)


def get_label(filename):
    """T1 = 0 (no stress), T2 or T3 = 1 (stress)"""
    basename = os.path.basename(filename)
    if "T1" in basename:
        return 0
    elif "T2" in basename or "T3" in basename:
        return 1
    else:
        raise ValueError(f"Cannot determine label from filename: {basename}")


def main():
    csv_files = sorted(glob.glob(os.path.join(BVP_DIR, "bvp_s*.csv")))
    if not csv_files:
        print(f"No BVP CSV files found in {BVP_DIR}")
        return

    print(f"Found {len(csv_files)} BVP files\n")

    all_arrays = []
    all_labels = []
    all_names  = []
    label_rows = []

    for i, filepath in enumerate(csv_files):
        # if i == 10:
        #     break
        basename = os.path.splitext(os.path.basename(filepath))[0]
        label    = get_label(filepath)

        print(f"[{i+1}/{len(csv_files)}] {basename}  →  label={label}", end="  ")

        try:
            signal  = load_bvp(filepath)
            cwt_img = compute_cwt(signal)        # (224, 224, 3)

            # save individual .npy
            npy_path = os.path.join(OUTPUT_DIR, f"{basename}.npy")
            np.save(npy_path, cwt_img)

            all_arrays.append(cwt_img)
            all_labels.append(label)
            all_names.append(basename)
            label_rows.append({"filename": basename, "label": label})

            print(f"saved  shape={cwt_img.shape}  (samples={len(signal)})")

        except Exception as e:
            print(f"ERROR: {e}")
        


    # ── Save combined dataset.npz ─────────────────────────────────────────────
    X = np.stack(all_arrays)   # shape: (N, 224, 224, 3)
    y = np.array(all_labels)   # shape: (N,)

    npz_path = os.path.join(OUTPUT_DIR, "dataset.npz")
    np.savez_compressed(npz_path, X=X, y=y, filenames=np.array(all_names))
    print(f"\nSaved dataset.npz  →  X: {X.shape}, y: {y.shape}")
    print(f"  Stress (1): {y.sum()}  |  No-stress (0): {(y==0).sum()}")

    # ── Save labels.csv ───────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "labels.csv")
    pd.DataFrame(label_rows).to_csv(csv_path, index=False)
    print(f"Saved labels.csv  →  {csv_path}")
    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()