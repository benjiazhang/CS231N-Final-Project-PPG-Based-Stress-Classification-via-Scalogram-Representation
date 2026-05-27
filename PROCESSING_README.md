# PPG Scalogram Processing — Design Decisions & Rationale

This document explains every choice made in `process_bvp_wesad.py` and
`process_bvp_ubfc1.py`, and how they interact with the three downstream models:
**EfficientNet-B0**, **CvT**, and **AudioMAE**.

---

## 1. Model Input Requirements (the constraints that drive everything)

| Model | Expected input size | Channels | Pretrained on |
|---|---|---|---|
| EfficientNet-B0 | 224×224 | 3 (RGB) | ImageNet |
| CvT | 224×224 | 3 (RGB) | ImageNet |
| AudioMAE | 128×128 | **1** (grayscale) | AudioSet mel spectrograms |

**The core tension:** EfficientNet and CvT want 224×224×3. AudioMAE wants
128×128×1 — single channel because audio is inherently single-channel and
that is what its patch embedder was trained on.

**Resolution:** We produce a single canonical scalogram per window —
`(224, 224, 1)` float32, normalized to [0, 1] — and handle the difference
at the model's input layer:

- **EfficientNet / CvT loaders**: expand to 3 channels via `np.repeat(X, 3, axis=1)`
  after transposing to `(N, C, H, W)`, then apply ImageNet normalization.
  Three identical channels is standard practice for grayscale inputs to
  ImageNet-pretrained models.
- **AudioMAE loader**: use the single channel directly, resize to 128×128
  in the loader.

**Why not save viridis-colored 3-channel images (as in the original UBFC
script):** Viridis is a display choice that adds no information. More
importantly, feeding a 3-channel RGB image to AudioMAE's single-channel
patch embedder is out-of-distribution for its first layer. Saving raw
grayscale power values gives every model the correct input.

---

## 2. Image Size: 224×224

We produce at 224×224 for all outputs. AudioMAE's loader downsamples to
128×128 on the fly.

Always produce at the highest required resolution and downsample at load
time. The reverse (produce at 128, upsample to 224) is lossy.

---

## 3. Windowing Strategy

### WESAD
WESAD provides continuous labeled signals in `SX.pkl` — one file per subject
covering the entire session (~2 hours). Labels change over time as the
protocol moves between conditions. We must window.

**Window size: 30 seconds (1920 samples at 64 Hz)**
- Standard in PPG/HRV literature: captures LF band (0.04–0.15 Hz, requires
  ~25s minimum) and HF band (0.15–0.4 Hz)
- Consistent with Milestone 2

**Window labeling:** A window is assigned the majority label across its 1920
samples. Windows where the majority covers less than 80% of samples are
discarded — these are transition windows between conditions and are genuinely
ambiguous.

**Label mapping (binary, per Schmidt et al. 2018):**
- Stress (1) = WESAD label 2
- Non-stress (0) = WESAD labels 1 (baseline) + 3 (amusement)
- Discarded = 0 (transient), 4 (meditation), 5/6/7 (protocol artifacts)

Combining baseline + amusement into non-stress is the standard binary
protocol for WESAD. Schmidt et al. 2018 Table 4 reports the 85.83% wrist
BVP baseline using exactly this split.

**Label resampling (700 Hz → 64 Hz):** Labels are at 700 Hz (RespiBAN
rate). We use `scipy.resample_poly` + round-to-int to resample to 64 Hz.
This handles the non-integer ratio (700/64 ≈ 10.9375) correctly and
preserves integer label IDs without interpolation artifacts.

### UBFC-Phys
UBFC provides one BVP CSV per trial per subject (T1, T2, T3). Each file is
a single continuous recording of one experimental condition — the label is
constant for the entire file. The current pipeline processes each trial as
one scalogram (no windowing).

**Known limitation of the no-windowing approach for UBFC:** A ~5 minute
trial at 64 Hz = 19,200 samples. After CWT and resize, the time axis is
compressed ~85× into 224 pixels. Short-duration stress features may be
invisible after this compression. WESAD windows (30s = 1920 samples) do
not have this problem.

**An alternative approach worth discussing as a team:** Window UBFC trials
the same way as WESAD (30s windows, label every window with the trial's
constant label). This would give many more training samples from UBFC and
make the time-frequency resolution consistent across both datasets. The
label assignment is unambiguous since T1 is always non-stress and T2/T3 are
always stress. This would be a meaningful improvement but requires updating
`process_bvp_ubfc1.py`.

---

## 4. CWT Parameters

```
wavelet  = 'morl'   (Morlet)
scales   = np.geomspace(1, 224, num=224)
```

**Morlet wavelet:** Standard for PPG/HRV analysis — good time-frequency
localization, oscillatory structure matches quasi-periodic BVP signal.
Same choice as in Mathunjwa et al. 2021 (ECG scalogram paper cited in
Milestone 1).

**Geomspace scales:** Geometric spacing in scale = logarithmic spacing in
frequency, which matches how HRV frequency bands are conventionally analyzed.
Both datasets use the same fs = 64 Hz, same wavelet, same scales — so their
scalograms are directly comparable when pooled.

---

## 5. Normalization

```python
power = power - power.min()
power = power / (power.max() + 1e-8)   # [0, 1]
```

Per-window min-max normalization. Makes images comparable across subjects
with different PPG amplitudes (skin tone, perfusion, sensor fit). ImageNet
normalization (mean/std) is applied in the dataloader, not here, so the
`.npz` stays in clean [0, 1] float32 that any loader can use.

---

## 6. Output Format

Each window: `(224, 224, 1)` float32 `.npy` file.

Combined `.npz` keys:
- `X`: `(N, 224, 224, 1)` float32
- `y`: `(N,)` int64 — 0=non-stress, 1=stress
- `filenames`: `(N,)` str
- `subjects`: `(N,)` int64 — used for LOSO grouping
- `window_ids`: `(N,)` int64 — window index within subject
- `dataset`: `(N,)` str — `'wesad'` or `'ubfc'`

**What is committed to the repo:** `.npz` + `.csv` only. Individual `.npy`
files are in `cwtFiles/` (gitignored) — they are redundant once the `.npz`
exists.

---

## 7. What Loaders Need to Do

Processing outputs `(N, H, W, 1)`. Loaders handle the rest:

**EfficientNet / CvT:**
```python
X = np.transpose(X, (0, 3, 1, 2))   # (N, 1, H, W)
X = np.repeat(X, 3, axis=1)          # (N, 3, H, W)
X = (X - IMAGENET_MEAN) / IMAGENET_STD
```

**AudioMAE:**
```python
X = np.transpose(X, (0, 3, 1, 2))   # (N, 1, H, W)
# resize to 128x128 in the loader if needed
# no ImageNet normalization
```

`crossval_base.py` already handles this — see `load_split()`.

---

## 8. Summary Table

| Decision | Choice | Reason |
|---|---|---|
| Image size | 224×224 | Highest model requirement; downsample for AudioMAE in loader |
| Channels | 1 (grayscale) | AudioMAE requires 1-channel; expand to 3 for EfficientNet/CvT in loader |
| Colormap | None (raw power) | Viridis adds no information; incompatible with AudioMAE |
| Wavelet | Morlet (`morl`) | Standard for PPG/HRV; consistent across both datasets |
| Scales | geomspace(1, 224, 224) | Log-spaced in frequency; matches UBFC pipeline |
| Normalization | Per-window min-max [0,1] | Cross-subject amplitude invariance |
| Window size | 30s / 1920 samples | HRV LF band minimum; literature standard |
| Window threshold | ≥80% majority label | Discard transition-contaminated windows |
| WESAD label map | stress=2, non-stress={1,3} | Schmidt et al. 2018 binary protocol |
| UBFC label map | T1=0, T2/T3=1 | Trial protocol; label constant per file |
| UBFC windowing | Whole-trial per scalogram | Current approach — see Section 3 for known limitation and proposed improvement |
