"""
process_ubfc_baselines.py
=========================
Processes UBFC-Phys raw BVP CSVs into raw signal windows for 1D baseline models
(LSTM, 1D CNN). Run this on the machine that has the raw UBFC BVP CSVs.

Output: ubfc_baselines.npz
  X:        (N, 1920) float32  — raw BVP windows (30s at 64 Hz)
  y:        (N,)      int64    — 0=non-stress, 1=stress
  subjects: (N,)      int64    — subject IDs offset by +100 (101-156)

Usage:
  python process_ubfc_baselines.py --ubfc_dir /path/to/bvp_ubfc
  python process_ubfc_baselines.py --ubfc_dir ./bvp_ubfc --out ./ubfc_baselines.npz

Label mapping:
  T1 = non-stress (0)
  T2, T3 = stress (1)
"""

import os
import glob
import re
import argparse
import numpy as np

FS          = 64.0
WINDOW_SEC  = 30
WINDOW_SAMP = int(FS * WINDOW_SEC)   # 1920
UBFC_OFFSET = 100   # subject IDs: 101-156


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ubfc_dir", required=True,
                        help="Directory containing bvp_s*.csv files")
    parser.add_argument("--out", default="ubfc_baselines.npz",
                        help="Output .npz path")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.ubfc_dir, "bvp_s*.csv")))
    if not csv_files:
        print(f"No bvp_s*.csv files found in {args.ubfc_dir}")
        return

    print(f"Found {len(csv_files)} CSV files")

    all_X, all_y, all_s = [], [], []

    for fpath in csv_files:
        basename = os.path.basename(fpath)
        match = re.search(r"s(\d+)_T(\d+)", basename)
        if not match:
            print(f"  [skip] Cannot parse: {basename}")
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
            all_s.append(sid + UBFC_OFFSET)

        print(f"  {basename}: {n_win} windows  label={label}  subject={sid + UBFC_OFFSET}")

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    subjects = np.array(all_s, dtype=np.int64)

    np.savez_compressed(args.out, X=X, y=y, subjects=subjects)

    print(f"\nSaved {args.out}")
    print(f"  X shape:    {X.shape}")
    print(f"  y shape:    {y.shape}")
    print(f"  Subjects:   {len(np.unique(subjects))} ({np.unique(subjects).min()}-{np.unique(subjects).max()})")
    print(f"  Stress:     {(y==1).sum()}")
    print(f"  Non-stress: {(y==0).sum()}")


if __name__ == "__main__":
    main()
