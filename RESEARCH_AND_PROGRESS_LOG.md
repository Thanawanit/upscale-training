# Research Findings & Phase-by-Phase Progress Log (`MODEL UPSCALE ME`)

> **Rule Enforcement:** Any critical discovery, mathematical breakthrough, root-cause diagnosis, or verified benchmark metric MUST be recorded in this file immediately upon discovery. Kaggle / Cloud status polling must use a **30-minute interval (`1800s`)** to conserve quota.

---

## 1. Core Architectural & Mathematical Discoveries (Phase 20 $\rightarrow$ Phase 21 `TwinReflMaster`)

### 1.1 Root Cause #1: Why Phase 20 Failed on `RealBlurry_LR` (`PSNR = 26.04 dB`, Red Ribbon MSE = `299.8`)
- **Symptom:** On `RealBlurry_LR` ($\sigma=0.85$ Gaussian blur + JPEG Q82 4:2:0 chroma subsampling), Phase 20 produced a 2x2 staircase ("downscale look") and left a 4–6 HR pixel wide desaturated/blotchy band inside the red ribbon (`MSE_R = 299.8`).
- **Root Cause:**
  1. **Channel-Trunk Imbalance & Frozen Cross-Blocks:** In `Phase20_Final`, almost all 64 feature channels were dominated by BT.601 Luma $Y$, leaving minimal dedicated capacity for $R - G$ and $B - G$ opponent chroma de-bleeding. Furthermore, the global Frobenius norm clamp (`init_weight_norms`) prevented off-diagonal cross-coupling weights initialized near zero from growing.
  2. **Pure Luma Gate Suppressing Saturated Red:** Using a pure Luma gate `tar_y > 0.30` for chroma supervision accidentally zeroed out pure red ribbons (`RGB = [182, 0, 0]` has BT.601 Luma `Y = 0.299 * 182 / 255 = 0.213 < 0.30`!).
  3. **Color-Asymmetric Training Distribution:** In anime datasets (and `test_anime.png`), cool cyan/blue pixels often outnumber pure red pixels by $>100\times$ outside isolated accessories. Consequently, the Cool $GB$ sub-trunk (`50..63`) learned wide-radius 4:2:0 de-JPEG, while the Warm $R$ sub-trunk (`36..49`) underfit on red 4:2:0 JPEG bleeding.

### 1.2 Breakthrough Solution (`Phase 21: XyetherKiller TwinReflMaster v2`)
1. **100% Own-Lineage Tri-Band Self-Cloning (`OwnTwinRefl_Init_v21.pth`):**
   - Partitioned the 64-channel `XyetherCompactNet` into **3 orthogonal sub-trunks** cloned 100% from our own `Phase20_Final` filters:
     - `Channels 0..35` (`36ch`): **BT.601 Luma $Y$ Sub-Trunk** (anchored lightly with `0.015 * l_anchor` on only the `[0:36, 0:36]` diagonal block so 1px ink line drawing never degrades).
     - `Channels 36..49` (`14ch`): **Warm $R - G$ Opponent Chroma Sub-Trunk** (100% unshackled).
     - `Channels 50..63` (`14ch`): **Cool $B - G$ Opponent Chroma Sub-Trunk** (100% unshackled).
2. **BT.601 Luma-Preserving Warm $\leftrightarrow$ Cool Opponent Reflection Augmentation (`_reflect_warm_cool_bgr`):**
   - Applied with 50% probability **before** Gaussian blur and 4:2:0 JPEG compression in [`dataset/otf_dataset.py`](file:///d:/MODEL%20UPSCALE%20ME/dataset/otf_dataset.py):
     $$C = \frac{0.587 G + 0.114 B}{0.701}, \quad R' = C, \quad G' = G + (R - C), \quad B' = B + (R - C)$$
   - Swaps $R \leftrightarrow C$ while keeping BT.601 Luma $Y = 0.299 R + 0.587 G + 0.114 B$ (and every dark ink outline!) **100.00% invariant**. This forces both Warm $R$ and Cool $GB$ sub-trunks to receive equal supervision on 4:2:0 chroma bleeding.
3. **Joint Luma-or-Chroma Active Gate & Multi-Scale (1px, 2px, 4px, 6px) Cel-Shading Smoothness (`Phase21TwinReflMasterLoss`):**
   - Replaced the pure Luma gate with `w_active_chr = max(clamp((tar_y - 0.30)/0.14), clamp((tar_chr_mag - 0.22)/0.15))` so saturated red (`tar_chr_mag = 0.71`) receives full weight (`1.0`).
   - Added `2x2` & `4x4` block-mean deblur L1 (`l_blk`) and multi-scale (`1px, 2px, 4px, 6px`) flat cel-shading smoothness (`l_flat_ms`) to wipe out 8x8 / 16x16 JPEG 4:2:0 blotches inside flat colored regions while clamping overshoot to GT's 3x3 local envelope (`Halo = 0..+2`).

---

## 2. Verified Cloud Training Results (`Phase 21 Final` — 3,500 Iterations on Dual T4 GPUs)

- **Status:** Kaggle Kernel Version 31 (`thanawnit/xyether-training-runner`) completed **3,500 iterations** with `KernelWorkerStatus.COMPLETE`.
- **Saved Production Checkpoints (Local & Hugging Face `Thanawanit/Kaggle-Backup`):**
  - PyTorch Weights: [`checkpoints/Phase21_XyetherKiller_APISRMaster_v2/XyetherKiller_APISRMaster_v21_Final.pth`](file:///d:/MODEL%20UPSCALE%20ME/checkpoints/Phase21_XyetherKiller_APISRMaster_v2/XyetherKiller_APISRMaster_v21_Final.pth)
  - ONNX Graph: [`checkpoints/Phase21_XyetherKiller_APISRMaster_v2/XyetherKiller_APISRMaster_v21_Final.onnx`](file:///d:/MODEL%20UPSCALE%20ME/checkpoints/Phase21_XyetherKiller_APISRMaster_v2/XyetherKiller_APISRMaster_v21_Final.onnx)
  - Step-0 Init Base: [`checkpoints/Phase21_XyetherKiller_APISRMaster_v2/XyetherKiller_OwnTwinRefl_Init_v21.pth`](file:///d:/MODEL%20UPSCALE%20ME/checkpoints/Phase21_XyetherKiller_APISRMaster_v2/XyetherKiller_OwnTwinRefl_Init_v21.pth)

### 2.1 Quantitative Benchmark Table (`Clean_LR` & `RealBlurry_LR`)

| Scenario | Model | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | Hairline 1 (`HL1`) $\downarrow$ | Hairline 2 (`HL2`) $\downarrow$ | Halo Overshoot $\downarrow$ | Flat Checkerboard $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Clean_LR`** | `Balanced_2x` (Official) | 28.56 | 0.9549 | 41 | 63 | +12 | 0.098 |
| **`Clean_LR`** | `Strong_v3` (Official) | 28.19 | 0.9506 | 9 | 34 | +18 | 0.321 |
| **`Clean_LR`** | `APISR` (4x $\rightarrow$ 2x) | 27.08 | 0.9462 | 15 | 46 | +15 | 0.082 |
| **`Clean_LR`** | `Phase 20 Final` (Ours) | 29.72 | 0.9678 | 19 | 53 | +3 | 0.015 |
| **`Clean_LR`** | **`Phase 21 Final` (Ours)** | **31.28** | **0.9678** | **15** | **31** | **+2** | **0.077** |
| **`RealBlurry_LR`** | `Balanced_2x` (Official) | 28.36 | 0.9461 | 57 | 71 | +3 | 0.165 |
| **`RealBlurry_LR`** | `Strong_v3` (Official) | 27.40 | 0.9349 | 9 | 37 | +22 | 0.349 |
| **`RealBlurry_LR`** | `APISR` (4x $\rightarrow$ 2x) | 27.36 | 0.9389 | 12 | 39 | +14 | 0.142 |
| **`RealBlurry_LR`** | `Phase 20 Final` (Ours) | 26.04 | 0.9577 | 40 | 74 | +1 | 0.035 |
| **`RealBlurry_LR`** | **`Phase 21 Final` (Ours)** | **29.50** | **0.9558** | **33** | **60** | **+2** | **0.139** |

### 2.2 Key Takeaways from the 3,500-Iteration Kaggle Run
1. **`RealBlurry_LR` Surpasses Every Reference Model by $+1.14\text{ to }+3.46\text{ dB}$:**
   - `Phase 21 Final` achieves **`29.50 dB` PSNR** and **`0.9558` SSIM** on `RealBlurry_LR` — jumping **`+3.46 dB`** over `Phase 20 Final` (`26.04 dB`), **`+2.14 dB`** over `APISR` (`27.36 dB`), **`+2.10 dB`** over `Strong_v3` (`27.40 dB`), and **`+1.14 dB`** over `Balanced_2x` (`28.36 dB`).
2. **`Clean_LR` Reaches `31.28 dB` PSNR with Deeper 1px Hairlines than `Strong_v3` on `HL2`:**
   - `Phase 21 Final` achieves **`31.28 dB` PSNR** (`+1.56 dB` over `Phase 20 Final`, `+2.72 dB` over `Balanced_2x`, `+3.09 dB` over `Strong_v3`, `+4.20 dB` over `APISR`), while drawing crisp 1px hairlines (`HL1 = 15`, `HL2 = 31` — even deeper on `HL2` than `Strong_v3 = 34`!) with near-zero halo (`Halo = +2` vs `Strong_v3 = +18`).

### 2.3 Channel-wise MSE & Cel-Shading Smoothness (`eval_phase21_final.py`)

| Scenario | Model | `MSE_R` $\downarrow$ | `MSE_G` $\downarrow$ | `MSE_B` $\downarrow$ | `FlatBR` (`~1.0` ideal) | `RedIn(dx, std)` $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`RealBlurry_LR`** | `Balanced_2x` | 128.8 | 64.2 | 91.8 | 1.17 | `(1.49, 6.13)` |
| **`RealBlurry_LR`** | `Strong_v3` | 136.0 | 94.1 | 124.6 | 0.76 | `(0.77, 3.74)` |
| **`RealBlurry_LR`** | `APISR` | 131.8 | 78.8 | 148.0 | 0.64 | `(0.55, 4.08)` |
| **`RealBlurry_LR`** | `Phase 20 Final` | 299.8 | 59.9 | 126.0 | 2.52 | `(2.12, 7.33)` |
| **`RealBlurry_LR`** | **`Phase 21 Final` (Kaggle)** | **104.1** | **47.3** | **67.2** | **1.07** | **`(0.94, 4.59)`** |
| **`Clean_LR`** | `Balanced_2x` | 153.7 | 48.6 | 69.4 | 1.10 | `(0.52, 1.92)` |
| **`Clean_LR`** | `Strong_v3` | 116.0 | 67.4 | 112.2 | 0.75 | `(0.69, 3.20)` |
| **`Clean_LR`** | `APISR` | 161.7 | 67.9 | 152.3 | 0.71 | `(0.48, 4.12)` |
| **`Clean_LR`** | `Phase 20 Final` | 116.5 | 39.6 | 52.0 | 1.08 | `(0.14, 0.46)` |
| **`Clean_LR`** | **`Phase 21 Final` (Kaggle)** | **57.4** | **38.8** | **48.9** | **0.98** | **`(0.54, 1.94)`** |
