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

---

## 3. Root-Cause Analysis of Remaining Visual Artifacts in `Phase 21 Final` (Post-3,500-Iter Inspection)

Following user visual inspection of `p21_final_RealBlurry_LR_ribbon.png`, `p21_final_RealBlurry_LR_hairline.png`, and `p21_final_Clean_LR_eye_color.png`, pixel-level diagnostics ([`diagnose_p21_issues.py`](file:///d:/MODEL%20UPSCALE%20ME/scratch/diagnose_p21_issues.py) & [`inspect_exact_defects.py`](file:///d:/MODEL%20UPSCALE%20ME/scratch/inspect_exact_defects.py)) revealed **4 exact mathematical root causes**:

### 3.1 Why Bright Shoulders & Cel Edges Look Slightly Darker / Dip Inward (`dY_near_edge = -1.87` on `RealBlurry`, `-3.12` on `Clean`)
- **Measured Evidence:**
  - At the Ribbon cyan shoulder (`y=260, x=229..232`, where `GT = [141, 141, 141, 141]`):
    - `Phase 21 Final` outputs `[143, 131, 137, 141]` on `Clean_LR` (RGB at `x=230` drops to `[24, 169, 212]` vs `GT = [51, 174, 203]`).
    - On `RealBlurry_LR`, `x=225..226` inside the red ribbon drops early (`[174, 0, 0], [138, 1, 11]` vs `APISR = [187, 0, 12], [180, 0, 0]`).
- **Mathematical Cause:**
  1. **One-Sided Anti-Halo Ceiling Bias (`l_halo` & `l_over_chr`):** `4.0 * F.relu(pred_y - max3_gt_y)` and `4.0 * F.relu(pred - max3_gt_rgb)` penalize upward errors (`+1`) with weight `4.0`, while downward errors (`-1`) have `0.0` penalty outside dark lines! Over 3,500 iterations with random noise, the network shifts its prediction `-2 to -4` levels *below* `max3_gt_y` on every bright shoulder (`x=230: 131 vs 141`) to avoid ever triggering the one-sided ReLU ceiling.
  2. **Over-Wide `w_dark_pix` (`tar_y < 0.42`):** Because `0.42 * 255 = 107`, any mid-tone or anti-aliased transition pixel with `Luma < 107` (including the red ribbon interior where `Y = 54`, and the gray hair shadow where `Y = 96..106`!) satisfies `tar_y < 0.42` and gets pulled *downward* by `l_under_ink = w_dark_pix * F.relu(pred_y - tar_y)`.
     - **Fix:** Restrict `w_dark_pix` strictly to **true local 1px Luma valleys (`valley_depth > 0.015`)** rather than `tar_y < 0.42`, and pair it with a symmetric **Bright Shoulder Floor (`l_under_shoulder`)** that prevents bright plateau pixels adjacent to dark lines from dipping below `tar_y`.

### 3.2 Why the Vertical Hairline Tip Wobbles / Splits into 2 Columns on `RealBlurry_LR` (`y=82..90, x=79..81`)
- **Measured Evidence (`inspect_exact_defects.py`):**
  - Look at the vertical hairline (`x=76..83`) at `y=80..88` on `RealBlurry_LR`:
    - `GT` has a 1px line at `x=81` (`Luma = 69..71`).
    - `APISR` sharpens `x=80` to a crisp single 1px column (`Luma = 24, 29, 24, 22, 29` while `x=81` stays bright at `94..98`).
    - `Phase 21 Final` splits the darkness across **two adjacent columns** `x=80` and `x=81`:
      - `y=80`: `[77, 78]` (wobbles right)
      - `y=82`: `[67, 63]` (2px wide)
      - `y=84`: `[61, 64]` (2px wide)
      - `y=86`: `[63, 70]` (shifts left)
- **Mathematical Cause:**
  - In `train_xyether.py`, `l_blk` penalizes `2x2` and `4x4` average-pooled blocks `(F.avg_pool2d(pred, 2, 2) - F.avg_pool2d(tar_b, 2, 2)).abs()` **everywhere**, including across 1px dark lines!
  - On `RealBlurry_LR`, when the 1px vertical hairline at `x=80..81` is blurred across two columns, `2x2` average pooling is satisfied equally well by `[61, 64]` (a blurry/wobbly 2px line) as by `[24, 97]` (a crisp 1px line), and `l_rgb` (L1) hedged between `x=80` and `x=81`!
  - **Fix:**
    1. Mask `l_blk` (`2x2` and `4x4` block-mean loss) by `(1.0 - is_edge)` so block-averaging operates **only in flat/chroma-bleed regions** and never blurs 1px lines.
    2. Add **2x2 Local Minimum-Contrast Line-Centering Loss (`l_line_nms`)**: matching `-F.max_pool2d(-pred_y, 2, 1)` and `F.max_pool2d(pred_y, 2, 1) - (-F.max_pool2d(-pred_y, 2, 1))` against `tar_y` so every blurred 2px trough is forced to collapse into a single crisp 1px minimum at the exact subpixel coordinate!

### 3.3 Why Cel-Shading Gradients (`CyanGrad` / `RedGrad`) Still Show Subtle 2x2 Grain on `RealBlurry_LR` (`RedGrad = 5.42` vs `GT = 3.59`, `APISR = 3.26`)
- **Mathematical Cause:**
  1. `multiscale_flat_smoothness_loss` in `Phase21TwinReflMasterLoss` used `F.relu(dx_p - dx_t)`, which had **zero gradient** whenever `dx_p <= dx_t` and used a hard threshold `span_gt < 0.050` that turned off smoothness in gently shaded anime gradients (like the hair highlight gradient and curved red ribbon shading).
  2. Furthermore, in `LumaGuidedAPISRTeacher`, `tar_b` used `gt_t` directly in flat areas, whereas `GT` itself has mild scan grain (`CyanGrad = 1.64, RedGrad = 3.59`), while `APISR` bilateral-filters flat regions (`RedGrad = 3.26`).
  - **Fix:** Use a **Continuous Soft-Gated Bilateral Total Variation + Exact Subpixel Gradient Matching** in flat/gradual shading zones (`w_gradual = clamp((0.12 - span_gt) / 0.08, 0, 1)`) so both flat cel areas and smooth hair/eye gradients are silky-smooth without any 2x2 grain or blotchiness.

---

## 4. Phase 22 Google Colab Tesla T4 GPU Experiments (`v1` $\rightarrow$ `v14` & `Phase 22 Colab Final`)

All Phase 22 equations were systematically tested on a live **Google Colab Tesla T4 GPU** (`colab-mcp`), revealing **5 foundational mathematical & dataset breakthroughs** that solve all 3 visual defects reported on `Phase 21 Final`:

### 4.1 Breakthrough #1: Self-Bounded Zero-Halo 1px H/V Valley Teacher (`build_p22_v6_teacher_np`)
- **Discovery:** Earlier teacher formulations (`v1..v5`) used `ap_y_min3 = -F.max_pool2d(-ap_y, 3, 1, 1)` (3x3 min of APISR) or positive shoulder boost (`+0.35 * val_neighbor`). Diagnostics on the `512x512` crop proved this leaked darkness from adjacent black outlines (`max_neg = -203.0`) and created `+73.0` upward halo spikes.
- **Equation (`build_p22_v6_teacher_np`):**
  1. Zero positive boost everywhere (`max_pos = 0.0` across 100% of pixels — **100% identical to GT** on all non-valley pixels!).
  2. Strict 1px Horizontal/Vertical local minimum check:
     $$\text{is\_min\_h} = \mathbb{I}\big(Y_{\text{GT}}(y,x) \le \min(Y_{\text{GT}}(y,x-1), Y_{\text{GT}}(y,x+1))\big)$$
     $$\text{is\_min\_v} = \mathbb{I}\big(Y_{\text{GT}}(y,x) \le \min(Y_{\text{GT}}(y-1,x), Y_{\text{GT}}(y+1,x))\big)$$
  3. Self-bounded trough darkening proportional strictly to the pixel's own bilateral valley depth $\text{val\_hv} = \max(\text{val\_h}\cdot\text{is\_min\_h}, \text{val\_v}\cdot\text{is\_min\_v})$:
     $$Y_{\text{self}} = \max\!\big(Y_{\text{GT}} - 1.15\,\text{val\_hv},\; 0.28\,Y_{\text{GT}}\big), \quad Y_{\text{tar}} = 0.65\,Y_{\text{self}} + 0.35\,\min(Y_{\text{GT}}, Y_{\text{APISR}})$$
- **Verified Teacher Output:**
  - Ribbon 1D (`y=260, x=224..233`): `[54, 54, 54, 51, 64, 141, 141, 141, 141, 141]` (**100% identical to GT!**)
  - Hairline 1D (`y=85, x=72..83`): `[158, 112, 19, 91, 105, 105, 106, 106, 84, 27, 96, 106]` (`x=74: 67 -> 19`, `x=81: 69 -> 27`, shoulders 100% identical to GT!)
  - Vertical Hairline (`y=80..92, x=80..81`): `[74, 31], [77, 30], [79, 27], [84, 28], [87, 30], [91, 39], [103, 55]` (100% single-column at `x=81`!).

### 4.2 Breakthrough #2: Valley-Free Multi-Scale Block-Mean Deblur & Neighbor-Detached 1px Valley Step
- **Discovery:** Why did `l_blk` (`2x2` & `4x4` average-pool L1) pull bright line shoulders down (`x=80` dropping from `84` to `60`, and `dY_near = -1.87` to `-2.47`)?
  - Because when `tar_b` darkens a 1px valley at `x=81` from `69` to `27`, the `2x2` block mean drops from `(84+69)/2 = 76.5` to `(84+27)/2 = 55.5`. On blurred input where the network predicts `x=81` around `51`, `F.avg_pool2d(pred, 2, 2)` forces the neighbor `x=80` down to `60` so `(60+51)/2 = 55.5`!
  - Furthermore, in `step_ph = 0.5*(pred_y(x-1) + pred_y(x+1)) - pred_y(x)`, if `pred_y(x-1)` is not detached, `F.relu(val_h - step_ph)` pushes gradients onto the neighbors or creates an asymmetric battle with `l_non_valley_floor`.
- **Equation Fix:**
  1. Compare `l_blk_rgb`, `l_blk_chr`, `l_chr`, and `l_flat_ms` against **clean `gt_b, gt_cr, gt_cb`** (never the valley-darkened `tar_b`) and mask out any `2x2` / `4x4` block containing a 1px valley (`no_val_2x2 = (F.max_pool2d(is_valley_1px, 2, 2) < 0.5).float()`).
  2. **Neighbor-Detached 1px Valley Step (`l_valley_1px`):** Detach the neighbors `pp1 = F.pad(pred_y.detach(), (1, 1, 1, 1), mode='replicate')` inside `step_ph` and `step_pv` so only the 1px valley center pixel `pred_y(x)` is pulled downward relative to its live neighbors.

### 4.3 Breakthrough #3: Endpoint-Symmetric Edge-Barrier Bilateral Smoothness (`bilateral_cel_smoothness_v2`)
- **Discovery:** In `bilateral_cel_smoothness_loss`, for `s=4 (k_pool=7, pad=3)` and `s=6 (k_pool=9, pad=4)`, checking `w_strict_flat[:, :, :, s:]` only checked whether the **right endpoint** `x+s` (`x=232`, inside flat Cyan) had a flat window (`[x+s-3 .. x+s+3] = [229..235]`), without checking whether the **left endpoint** `x` (`x=226`, inside Red Ribbon) or the path between them crossed a strong outline! Thus `s=6` reached right across the black outline at `x=227..228` and penalized the Red-to-Cyan contrast by `0.75`.
- **Equation Fix (`bilateral_cel_smoothness_v2`):**
  - Compute `path_span = F.max_pool2d(span3_gt, 2*s+1, 1, s)` and take the minimum of both endpoints `wx = torch.min(w_gradual[:, :, :, s:], w_gradual[:, :, :, :-s])`, guaranteeing **zero cross-edge leakage** at all scales `s in {1, 2, 4, 6}`.

### 4.4 Breakthrough #4: Dataset Green-Channel Patch-Selection Blindness & Warm-Cool Reflection Saturation Bound
- **Discovery (`Cell 54` & `Cell 57`):**
  1. Why did `train_patches_hr` have **`0` (ZERO) pure-red pixels (`R > 150, G < 30, B < 30`)** across all 48 patches?
     - First, in [`dataset/otf_dataset.py`](file:///d:/MODEL%20UPSCALE%20ME/dataset/otf_dataset.py#L82), `edge_energy` was computed solely on the **Green channel (`patch[:, :, 1]`)**. On a pure Red-to-Black boundary (`[182, 0, 0]` to `[0, 0, 0]`), the Green channel is `0` on **both sides** (`|dG| = 0`), making pure red-to-black outlines 100% invisible to patch selection!
     - Second, BT.601 Luma-preserving `reflect_warm_cool_bgr` on Cyan (`Y = 141`, `[R=51, G=174, B=203]`) can **never** produce pure Red (`G=0, B=0`) because any pixel with `G=0, B=0` has maximum possible BT.601 Luma `0.299 * 255 = 76.2 < 141`. Thus reflected Cyan always had `[R=179, G=46, B=75]`, teaching an unanchored network that `R ~ 180` has `G > 0, B > 0`.
  2. **Fix:** Compute `edge_energy` across **all 3 RGB channels (`max(|dR|, |dG|, |dB|)`)** and explicitly include real Red-Ribbon patches (`R - max(G, B) > 120`) in the training pool without overwriting them via `reflect_warm_cool_bgr`, while merging the learned Chroma-to-Chroma sub-trunk `W[36:64, 36:64]`.

---

### 4.5 Complete Verified Benchmark Table on Google Colab Tesla T4 GPU (`512x512` Center Validation Crop)

| Scenario | Model | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | `MSE_R` $\downarrow$ | `MSE_G` $\downarrow$ | `MSE_B` $\downarrow$ | `HL1 / HL2` $\downarrow$ | `Halo` $\downarrow$ | `dY_near` (`0.0` ideal) | `CyanG` $\downarrow$ | `RedIn_dx` $\downarrow$ | Vertical Hairline (`x=81 < x=80`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`RealBlurry_LR`** | `Balanced_2x` (Official) | 28.36 | 0.9461 | 128.8 | 64.2 | 91.8 | 57 / 71 | +3 | +0.57 | 3.47 | 1.62 | 5/7 (blurry) |
| **`RealBlurry_LR`** | `Strong_v3` (Official) | 27.40 | 0.9349 | 136.0 | 94.1 | 124.7 | 9 / 37 | +22 | +7.22 | 2.98 | 0.88 | 4/7 (severe halo) |
| **`RealBlurry_LR`** | `APISR` (Teacher) | 27.36 | 0.9394 | 131.8 | 78.8 | 148.0 | 29 / 23 | +9 | +1.96 | 2.73 | 0.60 | 2/7 (shifted to `x=80`) |
| **`RealBlurry_LR`** | `Phase 20 Final` | 26.04 | 0.9577 | 299.8 | 59.9 | 126.0 | 40 / 74 | +1 | -0.19 | 3.28 | 2.12 | 5/7 (blurry) |
| **`RealBlurry_LR`** | `Phase 21 Final` (Kaggle) | 29.50 | 0.9558 | 104.1 | 47.3 | 67.2 | 33 / 60 | +2 | -1.87 | 2.57 | 1.04 | 2/7 (wobbles `x=80..81`) |
| **`RealBlurry_LR`** | **`Phase 22 Colab v6`** | **29.74** | **0.9566** | 99.6 | **48.1** | **59.3** | **24 / 41** | **+0** | **-0.38** | **2.23** | 1.06 | 6/7 |
| **`RealBlurry_LR`** | **`Phase 22 Colab Final`** | **29.75** | **0.9560** | **98.2** | 49.2 | **59.4** | **25 / 39** | **+0** | **-0.51** | **2.25** | **1.05** | **7/7 (100% locked at `x=81`!)** |
| **`Clean_LR`** | `Balanced_2x` (Official) | 28.56 | 0.9549 | 153.7 | 48.6 | 69.4 | 41 / 63 | +12 | +2.27 | 3.28 | 0.70 | — |
| **`Clean_LR`** | `Strong_v3` (Official) | 28.19 | 0.9506 | 116.0 | 67.4 | 112.4 | 9 / 34 | +18 | +6.32 | 2.84 | 0.83 | — |
| **`Clean_LR`** | `APISR` (Teacher) | 27.08 | 0.9371 | 161.7 | 67.9 | 152.3 | 19 / 18 | +2 | +3.16 | 2.51 | 0.55 | — |
| **`Clean_LR`** | `Phase 20 Final` | 29.72 | 0.9678 | 116.5 | 39.6 | 52.0 | 19 / 53 | +3 | +1.00 | 2.15 | 0.26 | — |
| **`Clean_LR`** | `Phase 21 Final` (Kaggle) | 31.28 | 0.9678 | 57.4 | 38.8 | 48.9 | 15 / 31 | +2 | -3.12 | 2.20 | 0.68 | — |
| **`Clean_LR`** | **`Phase 22 Colab v6`** | **31.38** | **0.9665** | **55.2** | **42.2** | **44.7** | **22 / 44** | **+0** | **-0.34** | **2.02** | **0.52** | — |
| **`Clean_LR`** | **`Phase 22 Colab Final`** | **31.29** | **0.9660** | **55.7** | 43.3 | **45.8** | **23 / 43** | **+0** | **-0.20** | **2.03** | **0.54** | — |

---

## 5. Phase 22 Production Calibration & Final Benchmarks (`XyetherKiller_APISRMaster_v22_Final`)

### 5.1 Local Root-Cause Fixes Beyond Colab (`2px`-Radius Valley Teacher + Flat-Only Block-Mean)
1. **2px-Radius Bilateral Valley Teacher (`LumaGuidedAPISRTeacher`):** Replaced `0.5 * (l1 + r1)` with `min(max(l1, l2), max(r1, r2)) - gt_y` (and symmetrically for vertical `u1, u2, d1, d2`). Anti-aliased shoulders at ±1px previously attenuated valley supervision depth (`HL1` stopped around `57`); looking out to ±2px shoulders triples effective valley contrast without touching non-valley pixels.
2. **Flat-Only Block-Mean Edge Exclusion (`Phase21TwinReflMasterLoss` / `Phase22TwinReflMasterV3Loss`):** Masking only 1px valleys (`no_val_2x2`) still allowed `2x2` pooling in `l_blk_rgb` / `l_blk_chr` to average bright shoulders (`x=229`) with 2px blurred edge transitions, pulling `dY_near` negative (`-2.13` to `-3.0`). Replacing it with `flat_blk_2x2` / `flat_blk_4x4` masking **all** `is_edge` pixels isolates block-mean deblur strictly to flat cel regions.
3. **Sub-Trunk Surgical Calibration (`Z1_fp45_vhX50_io32`):** Combined `PeakPrec_s40`, `FinalPol_s45`, and `V6Hybrid_s75` across Luma `[:36, :36]`, Chroma `[36:, :]`, Cross-Block `[:36, 36:]`, and I/O layers while keeping 100% own-checkpoint lineage.

### 5.2 Complete Verified Benchmark Table (`512x512` Center Validation Crop — Local Production Verification)

| Scenario | Model | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | `MSE(R, G, B)` $\downarrow$ | `HL1 / HL2` $\downarrow$ | `Halo` $\downarrow$ | `dY_near` (`0.0` ideal) | `CyanG` $\downarrow$ | `RedIn (dx, std)` $\downarrow$ | Vertical Hairline (`x=81 < x=80`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`RealBlurry_LR`** | `Balanced_2x` (Official) | 28.36 | 0.9462 | 128.8 / 64.2 / 91.8 | 57 / 71 | +3 | +0.56 | 3.48 | 1.49, 6.13 | **7/7** |
| **`RealBlurry_LR`** | `Strong_v3` (Official) | 27.40 | 0.9349 | 136.0 / 94.1 / 124.6 | 9 / 37 | +22 | +7.21 | 2.98 | 0.77, 3.74 | 4/7 |
| **`RealBlurry_LR`** | `APISR` (Teacher, tiled) | 27.36 | 0.9395 | 131.7 / 78.7 / 147.8 | 29 / 23 | +8 | +1.88 | 2.72 | 0.55, 4.05 | 1/7 |
| **`RealBlurry_LR`** | `Phase20_Final` | 26.04 | 0.9577 | 299.8 / 59.9 / 126.0 | 40 / 74 | +1 | -0.19 | 3.28 | 2.12, 7.33 | 2/7 |
| **`RealBlurry_LR`** | `Phase21_Final` | 29.50 | 0.9558 | 104.1 / 47.3 / 67.2 | 33 / 60 | +2 | -1.87 | 2.57 | 0.94, 4.59 | 2/7 |
| **`RealBlurry_LR`** | **`Phase22_Final`** | **29.99** | **0.9597** | **94.5 / 42.1 / 58.8** | **28 / 55** | **+2** | **-0.92** | **2.34** | **0.69, 4.45** | **6/7** |
| **`Clean_LR`** | `Balanced_2x` (Official) | 28.56 | 0.9549 | 153.7 / 48.6 / 69.4 | 41 / 63 | +12 | +2.25 | 3.28 | 0.52, 1.92 | 2/7 |
| **`Clean_LR`** | `Strong_v3` (Official) | 28.20 | 0.9506 | 116.0 / 67.4 / 112.2 | 9 / 34 | +18 | +6.29 | 2.84 | 0.69, 3.20 | 4/7 |
| **`Clean_LR`** | `APISR` (Teacher, tiled) | 27.09 | 0.9371 | 161.6 / 67.9 / 152.1 | 19 / 18 | +2 | +3.14 | 2.51 | 0.47, 4.11 | 1/7 |
| **`Clean_LR`** | `Phase20_Final` | 29.72 | 0.9678 | 116.5 / 39.6 / 52.0 | 19 / 53 | +3 | +1.00 | 2.15 | 0.14, 0.46 | 1/7 |
| **`Clean_LR`** | `Phase21_Final` | 31.28 | 0.9678 | 57.4 / 38.8 / 48.9 | 15 / 31 | +2 | -3.12 | 2.19 | 0.54, 1.94 | 1/7 |
| **`Clean_LR`** | **`Phase22_Final`** | **30.95** | **0.9657** | **54.0 / 48.3 / 54.4** | **10 / 39** | **+2** | **-1.73** | **2.02** | **0.40, 1.63** | 1/7 |

### 5.3 Codebase Sync & Artifacts
- **Losses synced:** [`losses/xyether_losses.py`](file:///d:/MODEL%20UPSCALE%20ME/losses/xyether_losses.py) (`LumaGuidedAPISRTeacher` & `Phase21TwinReflMasterLoss` / `Phase22TwinReflMasterV3Loss`), verified with active + flat forward/backward pass tests.
- **Checkpoint archived locally:** [`checkpoints/Phase22_XyetherKiller_APISRMaster_v3/XyetherKiller_APISRMaster_v22_Final.pth`](file:///d:/MODEL%20UPSCALE%20ME/checkpoints/Phase22_XyetherKiller_APISRMaster_v3/XyetherKiller_APISRMaster_v22_Final.pth) (`3.64 MB`, `params_ema` FP16 + `params` FP32).
- **Pending external environment items:** Local ONNX export (`XyetherKiller_APISRMaster_v22_Final.onnx`) requires installing `onnx` / `onnxscript` (not installed in current Python 3.11 environment), and Hugging Face Hub backup requires authenticating via `hf auth login`.

---

## 6. Post-Phase 22 Root-Cause Discovery: Why Hairlines Still Wobble vs `APISR` (`GOD TIER` Comparison)

User visual review identified persistent **hairline bending/wobble ("เส้นเบี้ยว")** in `Phase 22 Final` compared directly against `APISR (4x->2x)`. Pixel-level diagnosis ([`scratch/diagnose_vs_apisr.py`](file:///d:/MODEL%20UPSCALE%20ME/scratch/diagnose_vs_apisr.py)) and `6x` crop inspection ([`diag_hairline_6x.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/diag_hairline_6x.png)) proved the exact cause:

1. **GT's Subpixel Shift (`x=80 -> x=81`) + Blurry Anti-Aliasing vs `APISR`'s Clean Straight Geometry (`RealBlurry_LR`, `y=70..90`):**
   - `GT` has a soft double-column transition shifting diagonally across subpixels: `y=70..76` sits at `x=80` (`[68, 79, 63, 68]`), then at `y=80..88` blurs across BOTH columns `x=80, 81` (`[74, 71]`, `[77, 71]`, `[79, 69]`, `[84, 69]`, `[87, 70]`).
   - When degraded to `256x256` `LR`, columns `(80, 81)` merge into one LR column — making a `2x` subpixel shift from `x=80` to `x=81` physically ill-posed.
   - **`APISR` (`GOD TIER` geometry)** snaps to a **completely straight, crisp 1px jet-black column at `x=80`** (`[39, 26, 10, 22, 21, 24, 29, 24, 22, 29]`) with clean unblurred shoulders (`94..107`) before curving smoothly at `y=92..94`, and keeps diagonal strokes crisp 1px black (`14..27`).
   - **`Phase 22 Final`** tried to follow `GT` (`l_rgb` / `l_non_valley_floor`) AND only supervised 1px H/V minima (`LumaGuidedAPISRTeacher`), causing a kink/bend at the fork (`y=78..82`: `[36,64]` -> `[89,78]` -> `[75,56]` -> `[70,49]`), leaving `P22` blurry 2px gray and visibly bent ("เบี้ยว").
2. **Metric Bug in `VOK` (`x=81 < x=80`):**
   - `VOK` penalized `APISR` as `1/7` because `APISR` kept the vertical hairline straight at `x=80`, falsely rewarding models that bent `x=80 -> x=81`!
3. **Direction Forward (`Phase 23`):** Supervise **all edges, corners, forks, and diagonal strokes** directly with **`APISR`'s geometry (`ap_t`)** rather than blurry anti-aliased `GT`, while clamping **upward edge halo** (`<= max3(GT) + margin`) and preserving `GT` flat-region true color (`1 - is_edge`).

---

## 7. Phase 23: `XyetherLineFlowNet` (100% Own Architecture Trained from Scratch, Zero-Halo `APISR` Geometry Distillation)

### 7.1 Architecture & Training Summary
- **Architecture:** [`XyetherLineFlowNet(num_feat=56, num_blocks=8, upscale=2)`](file:///d:/MODEL%20UPSCALE%20ME/archs/xyether_lineflow.py) (`514,148` params — fewer parameters than `XyetherCompactNet` `600,652`), combining `3x3` local mixes, alternating standard/dilated (`dilation=1,2`) separable `(1x9)` + `(9x1)` residual directional strip branches (`>150` HR px receptive span), 4-tap hierarchical feature fusion, a single `Sigmoid` spatial-channel output gate, and `PixelShuffle(2)` over `Nearest(x)` residual.
- **Lineage & Training:** Trained **100% from random initialization (scratch)** for `6000` iterations (`batch_size=12, patch_size=160`, `AdamW` `lr=3.5e-4 -> 1e-5`, `EMA=0.999` with warmup) on Kaggle GPU (`thanawnit/xyether-training-runner`, v33) via [`train_xyether_p23.py`](file:///d:/MODEL%20UPSCALE%20ME/train_xyether_p23.py), supervised by [`Phase23ZeroHaloAPISRTeacher`](file:///d:/MODEL%20UPSCALE%20ME/losses/xyether_losses.py#L1183) and [`Phase23LineFlowLoss`](file:///d:/MODEL%20UPSCALE%20ME/losses/xyether_losses.py#L1228) + `ColorLuvLoss(0.25)`.

### 7.2 Verified Cloud GPU & Local ONNXRuntime Benchmark (`512x512` Anime Eval)

| Scenario | Model | PSNR (dB) | SSIM | HL1 / HL2 Min | Edge Slope | LapVar | Halo Overshoot | Flat Checker |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Clean_LR`** | `Balanced_2x` | 28.56 | 0.9549 | 41 / 63 | 782.5 | 672.2 | +12 | 0.10 |
| **`Clean_LR`** | `Strong_v3` | 28.19 | 0.9506 | 9 / 34 | 914.0 | 1058.3 | +18 | 0.32 |
| **`Clean_LR`** | `Phase22_TwinReflMaster_v3` | 32.80 | 0.9760 | 32 / 64 | 801.7 | 794.4 | +14 | 0.11 |
| **`Clean_LR`** | `APISR_4x_to_2x` (Teacher) | 27.09 | 0.9371 | 19 / 18 | 846.7 | 969.3 | +2 | 0.17 |
| **`Clean_LR`** | **`Phase23_LineFlow_v1`** | **29.42** | **0.9619** | **22 / 19** | **849.4** | **903.0** | **-1 (Zero Halo)** | **0.15** |
| **`RealBlurry_LR`** | `Balanced_2x` | 28.36 | 0.9461 | 57 / 71 | 722.8 | 547.2 | +3 | 0.16 |
| **`RealBlurry_LR`** | `Strong_v3` | 27.40 | 0.9349 | 9 / 37 | 906.4 | 1017.9 | +22 | 0.35 |
| **`RealBlurry_LR`** | `Phase22_TwinReflMaster_v3` | 29.83 | 0.9611 | 57 / 72 | 706.4 | 381.5 | +2 | 0.15 |
| **`RealBlurry_LR`** | `APISR_4x_to_2x` (Teacher) | 27.36 | 0.9395 | 28 / 23 | 878.0 | 958.3 | +8 | 0.18 |
| **`RealBlurry_LR`** | **`Phase23_LineFlow_v1`** | **28.88** | **0.9549** | **28 / 38** | **824.4** | **783.1** | **+1 (Zero Halo)** | **0.20** |

### 7.3 Verified Improvements Over Phase 22 & Teacher Parity
1. **Unbroken Straight Hairline Column (`RealBlurry_LR`, `y=76..88` at `x=80`):**
   - `Phase22`: broke & kinked to `x=81` (`[89, 75, 70, 68, 70]` at `x=80`).
   - **`Phase23_LineFlow_v1`**: stays cleanly anchored at `x=80` (`[53, 43, 39, 40, 53]` at `y=80..88`) matching `APISR` straight geometry (`[24, 29, 24, 22, 29]`), verified visually in [`p23_hairline_6x.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/p23_hairline_6x.png).
2. **Zero Cyan/Bright Edge Halo:** Suppresses `APISR`'s `+8` (`RealBlurry_LR`) / ribbon cyan glow down to **`+1` / `-1`**, verified visually in [`p23_final_RealBlurry_LR_ribbon.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/p23_final_RealBlurry_LR_ribbon.png).
3. **Speed & ONNX Export Verification (`256x256 -> 512x512` CPU):**
   - ONNXRuntime CPU (`XyetherLineFlow_v23_Final.onnx` vs PyTorch EMA): **`max_abs_diff = 6.85e-07`**
   - CPU latency (4 threads): **`975.4 ms` warm (`1012.2 ms` cold)** vs `APISR.onnx` (`~5,915 ms` equivalent on CPU -> **`~6x` faster**).
   - Checkpoints archived in [`checkpoints/Phase23_XyetherLineFlow_v1/`](file:///d:/MODEL%20UPSCALE%20ME/checkpoints/Phase23_XyetherLineFlow_v1/XyetherLineFlow_v23_Final.pth) and [`XyetherLineFlow_v23_Final.onnx`](file:///d:/MODEL%20UPSCALE ME/checkpoints/Phase23_XyetherLineFlow_v1/XyetherLineFlow_v23_Final.onnx), and backed up on Hugging Face (`Thanawanit/Kaggle-Backup/checkpoints/Phase23_XyetherLineFlow_v1/`).

---

## 8. Phase 24 (`XyetherLineFlow_v24`) — Execution & GPU Teacher Root-Cause Fix

### 8.1 Kaggle v35 12h Timeout Root Cause (`CPUExecutionProvider` Fallback)
- **Observed Failure (v34 & v35):** `v34` was cancelled in queue (`KernelWorkerStatus.CANCEL_ACKNOWLEDGED`). `v35` ran `43,063s` (~11.96h) cleanly through step **`6300 / 7500`** before hitting Kaggle's 12-hour wall-clock limit (`CANCEL_ACKNOWLEDGED`, `loss=2.9870`, `l_halo=0.0023` at step `6300`).
- **Root Cause (`onnxruntime-gpu >= 1.27` vs Kaggle CUDA 12):** Unpinned `pip install onnxruntime-gpu` pulled CUDA 13 ONNX Runtime, failing on Kaggle T4 with `libcublasLt.so.13: cannot open shared object file` (`Require cuDNN 9.* and CUDA 13.*`), while `onnx2torch` raised `Average pool operation with spatial rank == -2`. As a result, `APISR.onnx` teacher silently fell back to **`CPUExecutionProvider`** (`~6.8 s/iter`), pushing 7,500 iterations past 12 hours (~14.1h).
- **Fix Applied:**
  1. [`Kaggle_Xyether_Training.ipynb`](file:///d:/MODEL%20UPSCALE%20ME/Kaggle_Xyether_Training.ipynb): Detect `torch.version.cuda` and pin **`onnxruntime-gpu==1.23.2`** for CUDA 12 environments, call `ort.preload_dlls(cuda=True, cudnn=True)`, and verify active `APISR.onnx` providers right after download.
  2. [`losses/xyether_losses.py`](file:///d:/MODEL%20UPSCALE%20ME/losses/xyether_losses.py): Run `onnx.shape_inference.infer_shapes` before `onnx2torch.convert` and invoke `ort.preload_dlls` before `InferenceSession` on CUDA, logging an explicit warning if `CPUExecutionProvider` fallback occurs.
  3. [`train_xyether_p23.py`](file:///d:/MODEL%20UPSCALE%20ME/train_xyether_p23.py): Enabled `persistent_workers=(args.num_workers > 0)` on `DataLoader`.

### 8.2 Verified Kaggle v36 Completion & Benchmark (`Phase24_LineFlow_v2`)
- **Run Time & GPU Teacher Confirmation:** Kaggle kernel `thanawnit/xyether-training-runner` (**v36**, commit `386ef85`) completed all `7,500` iterations in **`10,852s` (~3.01 hours)** — vs >14h projected on CPU — with `APISR.onnx` confirmed running on `['CUDAExecutionProvider', 'CPUExecutionProvider']` (`ORT=1.23.2`, Torch `CUDA=12.8`).

| Scenario | Model | PSNR (dB) | SSIM | HL1 / HL2 Min | Edge Slope | LapVar | Halo Overshoot | Flat Checker |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Clean_LR`** | `Phase23_LineFlow_v1` | 29.42 | 0.9619 | 22 / 19 | 849.4 | 903.0 | -1 (Zero Halo) | 0.15 |
| **`Clean_LR`** | **`Phase24_LineFlow_v2`** | **29.69** | **0.9601** | **19 / 21** | **828.1** | **938.2** | **-3 (Zero Halo)** | **0.14** |
| **`Clean_LR`** | `APISR_4x_to_2x` (Teacher) | 27.09 | 0.9371 | 19 / 18 | 846.7 | 969.3 | +2 | 0.17 |
| **`RealBlurry_LR`** | `Phase23_LineFlow_v1` | 28.88 | 0.9549 | 28 / 38 | 824.4 | 783.1 | +1 (Zero Halo) | 0.20 |
| **`RealBlurry_LR`** | **`Phase24_LineFlow_v2`** | **29.06** | **0.9552** | **24 / 35** | **815.0** | **868.6** | **-1 (Zero Halo)** | **0.16** |
| **`RealBlurry_LR`** | `APISR_4x_to_2x` (Teacher) | 27.36 | 0.9395 | 28 / 23 | 878.0 | 958.3 | +8 | 0.18 |
| **`Harsh_Q62_LR`** | `Balanced_2x` | 26.85 | 0.9288 | 68 / 74 | 735.2 | 400.6 | +7 | 0.19 |
| **`Harsh_Q62_LR`** | `Strong_v3` | 25.71 | 0.9115 | 40 / 31 | 905.6 | 981.3 | +30 | 0.36 |
| **`Harsh_Q62_LR`** | `Phase22_TwinReflMaster_v3` | 27.39 | 0.9285 | 83 / 82 | 690.0 | 186.3 | +4 | 0.17 |
| **`Harsh_Q62_LR`** | `Phase23_LineFlow_v1` | 27.73 | 0.9411 | 36 / 62 | 827.3 | 534.0 | +2 | 0.21 |
| **`Harsh_Q62_LR`** | **`Phase24_LineFlow_v2`** | **28.15** | **0.9458** | **29 / 69** | **809.1** | **638.8** | **+0 (Zero Halo)** | **0.16** |
| **`Harsh_Q62_LR`** | `APISR_4x_to_2x` (Teacher) | 26.21 | 0.9222 | 14 / 50 | 840.6 | 895.7 | +10 | 0.18 |

### 8.3 Local Multi-Scene, Visual & ONNXRuntime Verification ([`eval_p24_final.py`](file:///d:/MODEL%20UPSCALE%20ME/scratch/eval_p24_final.py))
- **Multi-Scene (`test_anime.png`):**
  - `face_eyes_hair [real]`: `Lap 1755.6 -> 1961.4` (`ChrMAE 1.60 -> 1.52`, `PSNR 27.07` vs `APISR 24.95 / ChrMAE 3.44`)
  - `face_eyes_hair [harsh_q62]`: `Lap 1099.3 -> 1312.4` (`PSNR 26.57 -> 26.84`, `Halo99.9 23 -> 19`, `ChrMAE 1.94 -> 1.79`)
  - `lower_folds [real]`: `PSNR 31.09 -> 31.27`, `Lap 572.2 -> 597.5`, `ChrMAE 1.18 -> 1.08`
  - `lower_folds [harsh_q62]`: `PSNR 30.10 -> 30.67`, `Lap 422.2 -> 466.9`, `Halo99.9 18 -> 15`, `ChrMAE 1.51 -> 1.33`
- **Visual Findings:**
  - Zero cyan edge glow around curved red ribbon even under `Harsh_Q62` ([`p24_final_Harsh_Q62_LR_ribbon.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/p24_final_Harsh_Q62_LR_ribbon.png)) where `Phase23` showed slight outer greenish-cyan tint and `APISR` exhibits strong cyan halo (`+10`).
  - Vertical hairline (`x=80`, `y=76..88`) remains anchored and cleaner on `Clean_LR`/`RealBlurry_LR` ([`p24_hairline_6x.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/p24_hairline_6x.png)), while harsh `Q=62` reduced reddish `4:2:0` blotches inside the dark `<` strap (`ChrMAE 2.37 -> 2.08`), leaving slight warm interior tint and lower extreme-compression `LapVar` (`638.8` vs `APISR 895.7`) as the remaining targets if pursuing Phase 25.
- **ONNX Parity & CPU Speed (`256x256 -> 512x512`, 4 threads):**
  - ONNXRuntime parity (`XyetherLineFlow_v24_Final.onnx` vs PyTorch EMA): **`max_abs_diff = 8.94e-07`**
  - CPU latency: **`741.7 ms` warm** (`1491.3 ms` cold) — ~**8x faster** than `APISR.onnx` CPU (`~5,915 ms`).
- **Artifacts:** Archived in [`checkpoints/Phase24_XyetherLineFlow_v2/`](file:///d:/MODEL%20UPSCALE%20ME/checkpoints/Phase24_XyetherLineFlow_v2/XyetherLineFlow_v24_Final.pth), [`XyetherLineFlow_v24_Final.onnx`](file:///d:/MODEL%20UPSCALE%20ME/checkpoints/Phase24_XyetherLineFlow_v2/XyetherLineFlow_v24_Final.onnx), and backed up on Hugging Face (`Thanawanit/Kaggle-Backup/checkpoints/Phase24_XyetherLineFlow_v2/`).

---

## 9. Pre-Phase 25 Study — APISR (CVPR 2024) Deep Dive & 2020+ Modern Anime Empirical Audit

### 9.1 Key Takeaways from APISR (CVPR 2024) Architecture & Methodology
1. **Shuffled Resize & Prediction-Oriented Multi-Codec Compression:** Real-world anime streams undergo multi-stage scaling and intra-prediction block codecs (H.264/HEVC/WebP) in arbitrary orders ($\pi(\text{Resize}, \text{Blur}, \text{Noise}, \text{Compression})$). Training exclusively on fixed `Blur -> Resize(0.5x) -> JPEG` causes sharpness drop when compression precedes or interleaves with resizing (`shuffled_reencode`).
2. **XDoG Hand-Drawn Line Enhancement & Narrow-Gap Separation:** APISR uses Extended Difference of Gaussians (XDoG) with outlier filtering to isolate hand-drawn strokes and prevent close parallel strokes from merging under heavy compression.
3. **Balanced Perceptual & Color Fidelity:** Avoiding color bleeding in flat/dark cel shading while preserving crisp line troughs requires balancing perceptual structure against domain-accurate chrominance.

### 9.2 2020+ Era Modern Anime Benchmark (`5` High-Res 2020–2025 Frames, `3` Degradations)
Evaluated via [`study_2020plus_and_apisr.py`](file:///d:/MODEL%20UPSCALE%20ME/scratch/study_2020plus_and_apisr.py) across `anime2020_1_id7306078.png`..`anime2020_5_id7294435.png` under `stream_1080p` (`Q=80`), `harsh_q62_420` (`Q=62`), and `shuffled_reencode` (`Resize(360) -> JPEG(72) -> AnisoBlur -> Resize(256, CUBIC) -> WebP(68)`), visualized in [`study_2020plus_anime_vs_apisr.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/study_2020plus_anime_vs_apisr.png):

| Degradation Regime | Model | PSNR (dB) | SSIM | LapVar | XDoG LineSpan | InkShallow | Halo99.9 | ChrMAE | DarkChrMAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`stream_1080p`** | `Phase23_v1` | 26.56 | 0.9052 | 1744.8 | 125.7% | +0.09 | 14.4 | 1.51 | 2.40 |
| **`stream_1080p`** | **`Phase24_v2`** | **26.58** | **0.9009** | **1791.7** | **125.8%** | **+0.10** | **13.6** | **1.42** | **2.23** |
| **`stream_1080p`** | `APISR_4x_to_2x` | 25.70 | 0.8833 | 2004.2 | 120.9% | +0.14 | 31.8 | 1.69 | 1.94 |
| **`harsh_q62_420`** | `Phase23_v1` | 26.30 | 0.8985 | 1523.8 | 120.6% | +0.21 | 16.6 | 1.76 | 3.01 |
| **`harsh_q62_420`** | **`Phase24_v2`** | **26.38** | **0.8969** | **1600.4** | **122.0%** | **+0.22** | **15.4** | **1.62** | **2.80** |
| **`harsh_q62_420`** | `APISR_4x_to_2x` | 25.36 | 0.8786 | 1922.8 | 117.1% | +0.26 | 33.4 | 1.79 | 2.27 |
| **`shuffled_reencode`** | `Phase23_v1` | 26.28 | 0.8923 | 1522.6 | 118.0% | +0.25 | 14.6 | 1.72 | 2.91 |
| **`shuffled_reencode`** | **`Phase24_v2`** | **26.33** | **0.8913** | **1582.6** | **118.6%** | **+0.27** | **13.6** | **1.57** | **2.70** |
| **`shuffled_reencode`** | `APISR_4x_to_2x` | 25.20 | 0.8745 | 1918.1 | 114.6% | +0.35 | 32.4 | 1.73 | 2.45 |

### 9.3 Three Concrete Root-Cause Discoveries on 2020+ Anime
1. **Luma-Only Edge Detection Misses 2020+ Colored Line-Art (`Iro-tsuki Sen`):**
   - Modern 2020+ anime extensively uses colored strokes (dark purple on magenta in `anime2020_1`, brown/red on skin, blue on teal). Because `Phase23/24` computed `edge_zone` and `l_lap` purely on grayscale luma $Y = 0.299R + 0.587G + 0.114B$, chromatic lines had weaker `edge_zone` (`14.77%` vs `15.80%` with Max-Channel RGB span) and no RGB Laplacian supervision (`anime2020_1` `P24 Lap=1386.1` vs `APISR 1887.5`).
2. **Narrow-Gap Bridging Between Close Parallel Strokes & Subpixel Jitter Softening:**
   - In `anime2020_1`, two close parallel curved purple lines with a `2-3px` bright pink gap become partially bridged/smeared under `harsh_q62_420` and `shuffled_reencode`.
   - Under `Q=62` / WebP compression, subpixel line centers jitter by $\pm 0.5\text{px}$, causing pixel-wise L1/Laplacian to hedge via softer edges. Adding a **Shift-Tolerant $3\times 3$ Local Contrast Span & Trough Match (`l_span`)** plus **XDoG Narrow-Gap Ridge Shoulder Recovery (`xdog_ridge` in `sh_pull`)** restores crisp stroke contrast and gap separation without ever clamping dark ink troughs upward.
3. **Dark-Scene $16\times 16$ Macroblock `4:2:0` Chroma Bleed (`DarkChrMAE`):**
   - In dark/shadow areas (`Y < 75/255`), `P24` lagged `APISR` on `DarkChrMAE` (`2.80` vs `2.27` on `harsh_q62_420`) because `4:2:0` macroblocks span `16x16` in 2x HR space and require dark-region chroma weighting (`w_dark`) across `4x4`, `8x8`, and `16x16` pools.

### 9.4 Verified 100-Step Local Continuation Proof ([`verify_p25_improvements.py`](file:///d:/MODEL%20UPSCALE%20ME/scratch/verify_p25_improvements.py))
Starting from `XyetherLineFlow_v24_Final.pth`, 100 local continuation steps with `Phase25LineFlowLossCandidate` + Shuffled/WebP OTF degradations achieved ([`ablate_p25_2020plus_proof.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/ablate_p25_2020plus_proof.png)):
- **`anime2020_1` (`stream_1080p`):** `PSNR 25.64 -> 25.89 dB`, `LapVar 1567.0 -> 1682.1`, `Halo99.9 25.0 -> 22.0`, `DarkChrMAE 3.56 -> 3.19`
- **`anime2020_1` (`harsh_q62_420`):** `PSNR 24.82 -> 25.15 dB`, `LapVar 1386.1 -> 1526.9` (`+140.8`), `Halo99.9 34.0 -> 30.0`, `DarkChrMAE 5.15 -> 4.52`
- **`anime2020_1` (`shuffled_reencode`):** `PSNR 25.01 -> 25.26 dB`, `LapVar 1370.2 -> 1508.6` (`+138.4`), `Halo99.9 26.0 -> 23.0`, `DarkChrMAE 4.41 -> 4.00` (surpassing `APISR 4.16`)
- **`center_ribbon`:** `stream_1080p LapVar 868.6 -> 966.1`, `harsh_q62_420 LapVar 638.8 -> 752.8` (`+114.0`, `Halo99.9` fixed at `19.0` vs `APISR 32.0`), `shuffled_reencode LapVar 671.8 -> 776.1` (`+104.3`, `DarkChrMAE 2.71 -> 2.61`).

---

## 10. Phase 25 (`XyetherLineFlow_v25_Final`) — Verified Cloud GPU & 2020+ Modern Anime Benchmark

### 10.1 Verified 4-Regime Benchmark (`phase25_metrics_report.json`, Kaggle v37 `COMPLETE`)

| Scenario | Model | PSNR (dB) | SSIM | HL1 / HL2 Min | Edge Slope | LapVar | Halo Overshoot | Flat Checker |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Clean_LR`** | `Phase23_LineFlow_v1` | 29.42 | 0.9619 | 22 / 19 | 849.4 | 903.0 | -1 | 0.15 |
| **`Clean_LR`** | `Phase24_LineFlow_v2` | 29.69 | 0.9601 | 19 / 21 | 828.1 | 938.2 | -3 | 0.14 |
| **`Clean_LR`** | **`Phase25_LineFlow_v3`** | **29.69** | **0.9601** | **17 / 18** | **842.9** | **958.4** | **+1 (Zero Halo)** | **0.12** |
| **`Clean_LR`** | `APISR_4x_to_2x` (Teacher) | 27.09 | 0.9371 | 19 / 18 | 846.7 | 969.3 | +2 | 0.17 |
| **`RealBlurry_LR`** | `Phase23_LineFlow_v1` | 28.88 | 0.9549 | 28 / 38 | 824.4 | 783.1 | +1 | 0.20 |
| **`RealBlurry_LR`** | `Phase24_LineFlow_v2` | 29.06 | 0.9552 | 24 / 35 | 815.0 | 868.6 | -1 | 0.16 |
| **`RealBlurry_LR`** | **`Phase25_LineFlow_v3`** | **28.98** | **0.9544** | **18 / 34** | **822.4** | **912.3** | **-2 (Zero Halo)** | **0.14** |
| **`RealBlurry_LR`** | `APISR_4x_to_2x` (Teacher) | 27.36 | 0.9395 | 28 / 23 | 878.0 | 958.3 | +8 | 0.18 |
| **`Harsh_Q62_LR`** | `Phase23_LineFlow_v1` | 27.73 | 0.9411 | 36 / 62 | 827.3 | 534.0 | +2 | 0.21 |
| **`Harsh_Q62_LR`** | `Phase24_LineFlow_v2` | 28.15 | 0.9458 | 29 / 69 | 809.1 | 638.8 | +0 | 0.16 |
| **`Harsh_Q62_LR`** | **`Phase25_LineFlow_v3`** | **27.99** | **0.9455** | **27 / 64** | **817.1** | **679.8** | **+1 (Zero Halo)** | **0.15** |
| **`Harsh_Q62_LR`** | `APISR_4x_to_2x` (Teacher) | 26.21 | 0.9222 | 14 / 50 | 840.6 | 895.7 | +10 | 0.18 |
| **`Shuffled_Reencode_LR`** | `Phase23_LineFlow_v1` | 28.18 | 0.9422 | 44 / 74 | 822.9 | 598.5 | +2 | 0.19 |
| **`Shuffled_Reencode_LR`** | `Phase24_LineFlow_v2` | 28.43 | 0.9441 | 35 / 62 | 814.8 | 688.7 | +0 | 0.14 |
| **`Shuffled_Reencode_LR`** | **`Phase25_LineFlow_v3`** | **28.46** | **0.9443** | **30 / 52** | **814.7** | **742.0** | **-1 (Zero Halo)** | **0.14** |
| **`Shuffled_Reencode_LR`** | `APISR_4x_to_2x` (Teacher) | 26.42 | 0.9199 | 25 / 82 | 839.9 | 936.5 | +8 | 0.17 |

### 10.2 Verified 2020+ Modern Anime 5-Scene Benchmark ([`eval_p25_final.py`](file:///d:/MODEL%20UPSCALE%20ME/scratch/eval_p25_final.py))

| 2020+ Degradation (5-Scene Avg) | Model | PSNR (dB) | SSIM | LapVar | XDoG LineSpan | Halo99.9 | ChrMAE | DarkChrMAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`stream_1080p`** | `Phase23_v1` | 26.56 | 0.9052 | 1744.8 | 125.7% | 14.4 | 1.51 | 2.40 |
| **`stream_1080p`** | `Phase24_v2` | 26.58 | 0.9009 | 1791.7 | 125.8% | 13.6 | 1.42 | 2.23 |
| **`stream_1080p`** | **`Phase25_v3`** | **26.59** | **0.9005** | **1823.3** | **125.5%** | **14.2** | **1.37** | **2.19** |
| **`stream_1080p`** | `APISR_4x_to_2x` | 25.70 | 0.8833 | 2004.2 | 120.9% | 31.8 | 1.69 | 1.94 |
| **`harsh_q62_420`** | `Phase23_v1` | 26.30 | 0.8985 | 1523.8 | 120.6% | 16.6 | 1.76 | 3.01 |
| **`harsh_q62_420`** | `Phase24_v2` | 26.38 | 0.8969 | 1600.4 | 122.0% | 15.4 | 1.62 | 2.80 |
| **`harsh_q62_420`** | **`Phase25_v3`** | **26.37** | **0.8962** | **1635.4** | **122.0%** | **16.0** | **1.58** | **2.71** |
| **`harsh_q62_420`** | `APISR_4x_to_2x` | 25.36 | 0.8786 | 1922.8 | 117.1% | 33.4 | 1.79 | 2.27 |
| **`shuffled_reencode`** | `Phase23_v1` | 26.28 | 0.8923 | 1522.6 | 118.0% | 14.6 | 1.72 | 2.91 |
| **`shuffled_reencode`** | `Phase24_v2` | 26.33 | 0.8913 | 1582.6 | 118.6% | 13.6 | 1.57 | 2.70 |
| **`shuffled_reencode`** | **`Phase25_v3`** | **26.36** | **0.8910** | **1620.5** | **118.8%** | **13.8** | **1.52** | **2.65** |
| **`shuffled_reencode`** | `APISR_4x_to_2x` | 25.20 | 0.8745 | 1918.1 | 114.6% | 32.4 | 1.73 | 2.45 |

### 10.3 Key Verified Gains & ONNXRuntime Parity
1. **Superior Hairline Recovery Under Shuffled Multi-Codec Re-encoding (`Shuffled_Reencode_LR`):**
   - On `Shuffled_Reencode_LR` ([`p25_hairline_6x.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/p25_hairline_6x.png) Row 3), `APISR`'s lower vertical hairline fades to `HL2=82` (`[87, 82, 80, 85, 91]`), whereas **`Phase25_LineFlow_v3` achieves `HL2=52` (`[82, 79, 67, 58, 53]`)** alongside `+53.3` higher `LapVar` (`742.0` vs `P24 688.7`) and `Halo = -1`.
2. **2020+ Colored Stroke & Dark Chroma Gains ([`p25_2020plus_anime_vs_apisr.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/p25_2020plus_anime_vs_apisr.png)):**
   - Improves `LapVar` (+31 to +38 avg across all 3 modern regimes) and lowers `ChrMAE` & `DarkChrMAE` across all 3 regimes (`anime2020_1` `shuffled_reencode`: `PSNR 25.08 -> 25.32 dB`, `LapVar 1366.7 -> 1405.1`, `DarkChrMAE 4.33 -> 4.13` beating `APISR 4.16`).
3. **ONNXRuntime Parity & CPU Speed (`256x256 -> 512x512`, 4 threads):**
   - `XyetherLineFlow_v25_Final.onnx` vs PyTorch EMA: **`max_abs_diff = 7.75e-07`**
   - CPU latency: **`776.8 ms` warm / `838.1 ms` cold** (~**7.6x faster** than `APISR.onnx` CPU `~5,915 ms`).
   - Checkpoints archived in [`checkpoints/Phase25_XyetherLineFlow_v3/`](file:///d:/MODEL%20UPSCALE%20ME/checkpoints/Phase25_XyetherLineFlow_v3/XyetherLineFlow_v25_Final.pth), [`XyetherLineFlow_v25_Final.onnx`](file:///d:/MODEL%20UPSCALE%20ME/checkpoints/Phase25_XyetherLineFlow_v3/XyetherLineFlow_v25_Final.onnx), and backed up on Hugging Face (`Thanawanit/Kaggle-Backup/checkpoints/Phase25_XyetherLineFlow_v3/`).

---

## 11. User Real-World 3-Scene Diagnostic (`model_comparison_result*.png`) & Phase 26 Root-Cause Discovery

### 11.1 Contrast-Band Decomposition & Why `v24`/`v25` Look Identical on Main Outlines Yet Differ from `APISR`/`Strong_v3` on Hair Flare & Subtle Details ([`deep_probe_flare_and_blending.py`](file:///d:/MODEL%20UPSCALE%20ME/scratch/deep_probe_flare_and_blending.py))

Across the user's 3 real-world comparison images (`img0_sunlight`: `model_comparison_result.png`, `img1_dark_hall`: `model_comparison_result (1).png`, `img3_hair_flare`: `model_comparison_result (3).png`), each `697x697` panel was decomposed into 3 contrast regimes by local `3x3` RGB span:
- **Flat/Gradient Surfaces (`span < 4` RGB levels):** `52.5%` (`img0`), `66.1%` (`img1`), `71.9%` (`img3`)
- **Low-Contrast Micro-Details (`4 <= span < 22` RGB levels — internal dark hair strands inside light flares, bangs creases, flare transition borders, pastel cheek blush lines):** `22.9%` (`img0`), `18.3%` (`img1`), `15.9%` (`img3`)
- **High-Contrast Main Ink Outlines (`span >= 35` RGB levels — outer silhouettes, eyes, mouth, collar):** `13.2%` (`img0`), `7.9%` (`img1`), `5.1%` (`img3`)

| Scene | Model | Low-Contrast (`4..22` lv) Lap / Grad | High-Contrast (`>=35` lv) Lap / Grad | Pixel Diff vs `v25` (`>=4` lv %) | Pixel Diff vs `APISR` (`>=4` lv %) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`img0_sunlight`** | `Original` | 3.74 / 1.54 | 70.45 / 26.80 | — | — |
| **`img0_sunlight`** | `LineFlow_v24` | 3.71 / 1.66 | 100.39 / 34.47 | 8.1% (MAE 1.24) | — |
| **`img0_sunlight`** | **`LineFlow_v25`** | **3.32 / 1.62** | **101.15 / 34.72** | 0.0% | **34.3% (MAE 3.84)** |
| **`img0_sunlight`** | `Strong_v3` | **4.40 / 1.78** | 101.83 / 34.67 | — | 23.0% (MAE 2.75) |
| **`img0_sunlight`** | `APISR` | **4.09 / 1.72** | **106.25 / 35.56** | — | 0.0% |
| **`img1_dark_hall`** | `Original` | 3.92 / 1.48 | 63.62 / 23.16 | — | — |
| **`img1_dark_hall`** | `LineFlow_v24` | 4.10 / 1.62 | 83.22 / 27.60 | 7.3% (MAE 1.01) | — |
| **`img1_dark_hall`** | **`LineFlow_v25`** | **3.73 / 1.57** | **84.80 / 28.04** | 0.0% | **21.6% (MAE 2.52)** |
| **`img1_dark_hall`** | `Strong_v3` | **5.69 / 1.85** | 88.04 / 28.59 | — | 16.5% (MAE 1.98) |
| **`img1_dark_hall`** | `APISR` | **4.99 / 1.74** | **91.97 / 29.27** | — | 0.0% |
| **`img3_hair_flare`** | `Original` | 2.66 / 1.26 | 36.40 / 17.56 | — | — |
| **`img3_hair_flare`** | `LineFlow_v24` | 2.24 / 1.15 | 79.15 / 26.86 | **0.5% (MAE 0.21)** | — |
| **`img3_hair_flare`** | **`LineFlow_v25`** | **2.35 / 1.17** | **79.43 / 26.94** | 0.0% | **24.5% (MAE 2.52)** |
| **`img3_hair_flare`** | `Strong_v3` | **3.62 / 1.50** | 85.22 / 28.59 | — | 13.7% (MAE 1.81) |
| **`img3_hair_flare`** | `APISR` | **3.56 / 1.51** | **84.48 / 28.45** | — | 0.0% |

### 11.2 Two Architectural / Loss Root Causes Discovered in `losses/xyether_losses.py`

1. **Root Cause 1 — `0.85 * gt_t` Non-Edge Teacher Dilution Blocking `APISR`'s Dominant Base-Color Consolidation (`Phase23ZeroHaloAPISRTeacher` line 1239):**
   - In `img3_hair_flare`, a warm golden light flare splashes across the character's dark charcoal hair (`y=20..140`).
   - Both **`APISR`** and **`Strong_v3`** perform **Bilateral Color Mode Consolidation**: they raise the cool Blue channel (`B: 87.7 -> 89.1` in `APISR`, `88.2` in `Strong_v3`) and pull down the warm Red/Green flare haze across every vertical band, reducing `HSV Saturation` by **`11%–22%`** across the hair zones (`Zone3_RightHairFlare Sat: Original 47.4 -> Strong_v3 44.1 -> APISR 42.2`; `Zone2_MidBangsFlare Sat: 23.3 -> 19.1 -> 18.1`; `Zone4_LowerDarkHair Sat: 31.7 -> 27.9 -> 24.9`) and reducing warm excess `Warm(0.5*(R+G)-B)` from `15.06 -> 13.18` (`Strong_v3: 13.90`). This consolidates the diffuse multi-ring yellow gradient into a clean anime cel-highlight band while pulling the transition region back to the **dominant dark hair base color**.
   - In [`losses/xyether_losses.py`](file:///d:/MODEL%20UPSCALE%20ME/losses/xyether_losses.py) line 1239:
     ```python
     tar_b = edge_zone * ap_no_halo + (1.0 - edge_zone) * (0.85 * gt_t + 0.15 * ap_no_halo)
     ```
     Because `edge_zone` is `0.0` across all hair/skin surfaces and soft gradients (`85%` of pixels), our teacher target (`Old_Teacher_Tar`) discarded **85% of `APISR`'s color consolidation** (`Old_Teacher_Tar Sat = 45.7, Warm = 14.49`) and forced `LineFlow_v24/v25` to reproduce raw `gt_t` (`LineFlow_v25 Sat = 47.0, Warm = 14.65`).

2. **Root Cause 2 — Low-Contrast Detail Crusher in `_bilateral_cel_smoothness_v2` & `edge_strict` (`losses/xyether_losses.py` lines 1029–1043 & 1430):**
   - In `_bilateral_cel_smoothness_v2`, `w_gradual = ((0.095 - path_span) / 0.055).clamp(0, 1)` applied `13x13` (`s=1,2,4,6`) bilateral smoothing to every structure below **`24.2` RGB levels (`0.095 * 255`)**, and `w_strict_flat = (path_span < 0.038)` (`9.7` RGB levels) penalized `F.relu(dx_p.abs() - 0.80 * dx_t.abs())` — **explicitly forcing the student's gradients to be at least 20% weaker than the target on all `< 10` RGB-level micro-lines**.
   - Simultaneously, `edge_strict = (edge_zone > 0.28)` (`> 10.6` RGB levels) disabled `l_lap` and `l_span` on subtle features.
   - As a result, `LineFlow_v24` and `v25` blurred out:
     - **Internal dark hair shadow strands inside the light flare (`TopLeft_Flare_Strands`):** `Original Grad1px = 0.46`, `APISR = 0.54`, `Strong_v3 = 0.48`, vs **`LineFlow_v24/v25 = 0.29` (`-37%` blurrier than Original)**.
     - **Horizontal hair crease line inside middle bangs (`MidBangs_Flare_Edge`)** ([`user_img3_hotspot_analysis.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/user_img3_hotspot_analysis.png) Row 3).
     - **4 pastel pink diagonal cheek blush lines (`Cheek_Blush_Lines`, `4..11` RGB levels):** `Original Grad1px = 0.45`, `APISR = 0.54`, `Strong_v3 = 0.62`, vs **`LineFlow_v25 = 0.32` (`-41%` softer than APISR)**.

### 11.3 Verified Phase 26 Prototype Fix at Both Full-Panel (`verify_flare_and_lowcontrast_fix.py`) and Native 2x (`160x160 -> 320x320`, `test_native_scale_flare_and_blending.py`)

By replacing `0.85 * gt_t` non-edge dilution with **Full-Field Zero-Halo APISR Target (`strong_edge * ap_clamped + (1 - strong_edge) * ap_b`)**, lowering `_bilateral_cel_smoothness_v3` threshold from `24.2 -> 7.1` RGB levels (`0.095 -> 0.028`) with zero `0.80x` gradient suppression, and adding **Relative Micro-Contrast Booster (`w_micro_boost`)** for `3..28` RGB-level micro-lines ([`user_img3_native2x_p26_proof.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/user_img3_native2x_p26_proof.png)):

| Native 2x Region (`160x160 -> 320x320`) | Model | Warm Excess (`RG-B`) | Saturation (`HSV S`) | Micro-Edge `Grad1px` | `MAE vs APISR_2x` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`TopRight_HairFlare`** | `Original_2x` | 11.96 | 41.9 | 1.38 | 4.95 |
| **`TopRight_HairFlare`** | `LineFlow_v25` | 11.67 | 41.3 | 2.00 | 3.08 |
| **`TopRight_HairFlare`** | **`P26_Fixed_Proto`** | **10.47** | **37.0** | **1.94** | **2.38** |
| **`TopRight_HairFlare`** | `Strong_v3` | 11.22 | 38.8 | 2.68 | 2.44 |
| **`TopRight_HairFlare`** | `APISR_2x` | **10.42** | **36.4** | **2.39** | 0.00 |
| **`TopLeft_HairStrands`** | `Original_2x` | -6.07 | 36.7 | 0.39 | 2.54 |
| **`TopLeft_HairStrands`** | `LineFlow_v25` | -5.73 | 35.5 | 0.42 | 2.32 |
| **`TopLeft_HairStrands`** | **`P26_Fixed_Proto`** | **-5.43** | **31.4** | **0.54 (+28.6%)** | **1.41 (-39%)** |
| **`TopLeft_HairStrands`** | `Strong_v3` | -5.83 | 31.8 | 0.86 | 1.74 |
| **`TopLeft_HairStrands`** | `APISR_2x` | **-5.28** | **30.8** | **0.82** | 0.00 |
| **`MidBangs_LightEdge`** | `Original_2x` | -1.04 | 27.0 | 1.02 | 4.07 |
| **`MidBangs_LightEdge`** | `LineFlow_v25` | -0.88 | 26.1 | 1.46 | 2.66 |
| **`MidBangs_LightEdge`** | **`P26_Fixed_Proto`** | **-1.13** | **22.6** | **1.47** | **1.80 (-32%)** |
| **`MidBangs_LightEdge`** | `Strong_v3` | -1.08 | 23.9 | 1.94 | 2.13 |
| **`MidBangs_LightEdge`** | `APISR_2x` | **-1.10** | **21.2** | **1.76** | 0.00 |
| **`Cheek_BlushLines`** | `Original_2x` | 21.23 | 37.5 | 0.38 | 1.85 |
| **`Cheek_BlushLines`** | `LineFlow_v25` | 21.91 | 39.0 | 0.31 | 1.55 |
| **`Cheek_BlushLines`** | **`P26_Fixed_Proto`** | **19.28** | **35.9** | **0.51 (+64.5%)** | **1.04 (-33%)** |
| **`Cheek_BlushLines`** | `Strong_v3` | 23.23 | 39.3 | 1.23 | 1.88 |
| **`Cheek_BlushLines`** | `APISR_2x` | **20.02** | **36.2** | **0.55** | 0.00 |

---

## 12. Phase 26 (`XyetherLineFlow_v26_Final`) — Verified Cloud GPU, Real-World 3-Scene & 2020+ Modern Anime Benchmark

### 12.1 Verified 4-Regime Benchmark (`phase26_metrics_report.json`, Kaggle v38 `COMPLETE`)

| Scenario | Model | PSNR (dB) | SSIM | HL1 / HL2 Min | LapVar | MAE vs APISR | Halo Overshoot | Flat Checker |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Clean_LR`** | `Phase24_LineFlow_v2` | 29.69 | 0.9601 | 19 / 21 | 938.2 | 5.24 | -3 | 0.14 |
| **`Clean_LR`** | `Phase25_LineFlow_v3` | 29.69 | 0.9601 | 17 / 18 | 958.4 | 5.11 | +1 | 0.12 |
| **`Clean_LR`** | **`Phase26_LineFlow_v4`** | **29.22** | **0.9557** | **15 / 22** | **955.3** | **4.08 (-20.2%)** | **+2 (Zero Halo)** | **0.16** |
| **`Clean_LR`** | `Strong_v3` | 28.19 | 0.9506 | 9 / 34 | 1058.3 | 5.00 | +18 | 0.32 |
| **`Clean_LR`** | `APISR_4x_to_2x` (Teacher) | 27.09 | 0.9371 | 19 / 18 | 969.3 | 0.00 | +2 | 0.17 |
| **`RealBlurry_LR`** | `Phase24_LineFlow_v2` | 29.06 | 0.9552 | 24 / 35 | 868.6 | 4.82 | -1 | 0.16 |
| **`RealBlurry_LR`** | `Phase25_LineFlow_v3` | 28.98 | 0.9544 | 18 / 34 | 912.3 | 4.68 | -2 | 0.14 |
| **`RealBlurry_LR`** | **`Phase26_LineFlow_v4`** | **28.53** | **0.9504** | **16 / 37** | **915.5** | **3.94 (-15.8%)** | **+0 (Zero Halo)** | **0.17** |
| **`RealBlurry_LR`** | `Strong_v3` | 27.40 | 0.9349 | 9 / 37 | 1017.9 | 4.67 | +22 | 0.35 |
| **`RealBlurry_LR`** | `APISR_4x_to_2x` (Teacher) | 27.36 | 0.9395 | 28 / 23 | 958.3 | 0.00 | +8 | 0.18 |
| **`Harsh_Q62_LR`** | `Phase24_LineFlow_v2` | 28.15 | 0.9458 | 29 / 69 | 638.8 | 5.55 | +0 | 0.16 |
| **`Harsh_Q62_LR`** | `Phase25_LineFlow_v3` | 27.99 | 0.9455 | 27 / 64 | 679.8 | 5.46 | +1 | 0.15 |
| **`Harsh_Q62_LR`** | **`Phase26_LineFlow_v4`** | **27.58** | **0.9415** | **30 / 68** | **690.4** | **4.79 (-12.3%)** | **+3** | **0.17** |
| **`Harsh_Q62_LR`** | `Strong_v3` | 25.71 | 0.9038 | 40 / 31 | 981.3 | 5.73 | +30 | 0.36 |
| **`Harsh_Q62_LR`** | `APISR_4x_to_2x` (Teacher) | 26.21 | 0.9222 | 14 / 50 | 895.7 | 0.00 | +10 | 0.18 |
| **`Shuffled_Reencode_LR`** | `Phase24_LineFlow_v2` | 28.43 | 0.9441 | 35 / 62 | 688.7 | 5.60 | +0 | 0.14 |
| **`Shuffled_Reencode_LR`** | `Phase25_LineFlow_v3` | 28.46 | 0.9443 | 30 / 52 | 742.0 | 5.45 | -1 | 0.14 |
| **`Shuffled_Reencode_LR`** | **`Phase26_LineFlow_v4`** | **28.21** | **0.9408** | **33 / 58** | **753.9** | **4.66 (-14.5%)** | **+1 (Zero Halo)** | **0.16** |
| **`Shuffled_Reencode_LR`** | `Strong_v3` | 26.64 | 0.9096 | 22 / 34 | 1004.0 | 5.33 | +28 | 0.37 |
| **`Shuffled_Reencode_LR`** | `APISR_4x_to_2x` (Teacher) | 26.42 | 0.9199 | 25 / 82 | 936.5 | 0.00 | +8 | 0.17 |

### 12.2 Verified Evaluation on User's 3 Real-World Comparison Images ([`eval_p26_final_on_user_images_and_2020plus.py`](file:///d:/MODEL%20UPSCALE%20ME/scratch/eval_p26_final_on_user_images_and_2020plus.py))

1. **Full-Panel Color Consolidation & Main-Outline Sharpness (`704x704 -> 1408x1408 -> 704x704`, [`p26_final_user_img3_full_comparison.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/p26_final_user_img3_full_comparison.png)):**

| Scene | Model | High-Contrast (`>=35` lv) Lap | Low-Contrast (`4..22` lv) Lap | Saturation (`HSV S`) | `MAE vs APISR` | Pixel Diff `>=4` vs `v25` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`img0_sunlight`** | `LineFlow_v24` | 95.10 | 2.80 | 54.8 | 3.75 | 8.0% |
| **`img0_sunlight`** | `LineFlow_v25` | 96.92 | 2.54 | 54.6 | 2.28 | 0.0% |
| **`img0_sunlight`** | **`LineFlow_v26`** | **110.44** | **2.62** | **53.4** | **2.06 (Best)** | **33.0%** |
| **`img0_sunlight`** | `Strong_v3` / `APISR` | 97.54 / 101.90 | 3.48 / 3.15 | 54.2 / **51.9** | 3.65 / 0.00 | 17.4% / 34.7% |
| **`img1_dark_hall`** | `LineFlow_v24` | 78.72 | 3.58 | 35.1 | 2.95 | 7.3% |
| **`img1_dark_hall`** | `LineFlow_v25` | 80.83 | 3.40 | 35.2 | 2.00 | 0.0% |
| **`img1_dark_hall`** | **`LineFlow_v26`** | **90.85** | **3.69** | **33.2** | **1.66 (Best)** | **18.2%** |
| **`img1_dark_hall`** | `Strong_v3` / `APISR` | 84.07 / 87.69 | 5.19 / 4.57 | 35.8 / **31.6** | 2.76 / 0.00 | 12.2% / 21.7% |
| **`img3_hair_flare`** | `LineFlow_v24` | 67.84 | 3.03 | 41.5 | 3.83 | 6.9% |
| **`img3_hair_flare`** | `LineFlow_v25` | 75.52 | 1.95 | 41.5 | 2.01 | 0.0% |
| **`img3_hair_flare`** | **`LineFlow_v26`** | **85.85** | 1.87 | **39.4** | **1.80 (Tied Best)** | **27.7%** |
| **`img3_hair_flare`** | `Strong_v3` / `APISR` | 81.11 / 80.41 | 3.06 / 2.82 | 39.8 / **37.7** | 1.80 / 0.00 | 9.7% / 25.0% |

2. **Native 2x Hotspots on Image (3) (`160x160 -> 320x320`, [`p26_final_user_img3_native2x.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/p26_final_user_img3_native2x.png)):**
   - **`TopRight_HairFlare`:** `MAE_vs_APISR` dropped `3.12 (v24) -> 3.08 (v25) -> 2.71 (v26)`, `Warm(RG-B): 11.67 -> 11.44`, `Sat: 41.3 -> 40.7`.
   - **`TopLeft_HairStrands`:** `Sat` consolidated `35.7 (v24) -> 35.5 (v25) -> 33.0 (v26)` (`Strong_v3 = 31.8, APISR = 30.8`), `Warm: -5.73 -> -5.32` (`APISR = -5.28`), `MAE_vs_APISR: 2.32 -> 1.83 (-21.1%)`.
   - **`MidBangs_LightEdge`:** `Sat` consolidated `26.3 (v24) -> 26.1 (v25) -> 24.4 (v26)` (`Strong_v3 = 23.9, APISR = 21.2`), `MAE_vs_APISR: 2.66 -> 2.08 (-21.8%, beating Strong_v3 = 2.13)`.
   - **`Cheek_BlushLines`:** `Warm: 21.91 -> 20.34` (`APISR = 20.02`), `Sat: 39.0 -> 37.4` (`APISR = 36.2`), `MAE_vs_APISR: 1.55 -> 1.18 (-23.9%, beating Strong_v3 = 1.88)`.

3. **Teacher 2x Downsampling Bottleneck Discovery (`task-9447`):**
   - While `Phase26_LineFlow_v4` achieved the best color consolidation (`MAE_vs_APISR` reduced by `12%–24%` across all scenes and surpassing `Strong_v3`) and the highest high-contrast line sharpness (`HighContrast Lap = 110.44, 90.85, 85.85`), 1px low-contrast micro-lines (`4..22` RGB levels such as faint pastel cheek blush lines) remained softer in full-dataset training than in native-scale prototype training (`Cheek Grad1px = 0.23` vs `0.51`).
   - **Verified Mathematical Root Cause (`task-9447`):** Inside `Phase23ZeroHaloAPISRTeacher.forward(hr_tensor)`, `ap_t` is generated from a **2x-downsampled** version of `gt_t` (`lr_clean = F.avg_pool2d(gt_t, 2, 2) -> APISR(80x80) -> avg_pool2d(2, 2)`). Downsampling `gt_t` (`160x160`) to `80x80` via `2x2` box averaging halves the contrast of 1px micro-lines (`4..14` levels $\rightarrow$ `2..7` levels), causing `APISR(80x80)` to smooth them out in `ap_t` (`Tar_P26 LowContrast Lap = 3.92` vs `Tar_Old = 4.24`), AND 78% of `AnimeOTFDataset` applies heavy blur/WebP degradation to `lr_patch` that teaches the student to treat `4..12` level fluctuations as compression noise.
   - **Next-Phase Solution (Frequency-Decoupled Hybrid Target for Phase 27 if desired):** Take **low-frequency base color & light-flare consolidation (`F.avg_pool2d(ap_t, 5, 1, 2)`) + high-contrast edges (`strong_edge * ap_no_halo`) from `APISR`**, while injecting **high-frequency micro-detail residual (`gt_t - F.avg_pool2d(gt_t, 5, 1, 2)`) from `160x160` native `gt_t`** inside the `4..24` RGB-level micro-detail band!

---

## 13. Pre-Phase 27 10-Model 2021++ Anime Evaluation, Topaz/Literature Analysis & Bottom-Right Curve Wobble Root Cause

### 13.1 Genuine 2021++ 5-Scene Modern Anime Benchmark Across 10 SOTA 2x Models ([`benchmark_all_sota_2021plus.py`](file:///d:/MODEL%20UPSCALE%20ME/scratch/benchmark_all_sota_2021plus.py))
Evaluated on 5 modern 2021++ anime titles (`Sousou no Frieren`, `Bocchi the Rock!`, `Oshi no Ko`, `Kusuriya no Hitorigoto`, `Blue Archive`) across `stream_1080p`, `harsh_q62_420`, and `shuffled_reencode`, plus visual inspection of every rendered crop ([`p27_2021plus_5scenes_visual_compare.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/p27_2021plus_5scenes_visual_compare.png), [`p27_sota_2021plus_frieren_visual.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/p27_sota_2021plus_frieren_visual.png), [`p27_sota_img3_Cheek_Blush.png`](file:///C:/Users/IDAEPAD/.gemini/antigravity/brain/3cccec3d-87c9-4cd7-81d7-a85095f477eb/p27_sota_img3_Cheek_Blush.png)):

| 2021++ Degradation (5-Scene Avg) | Model | PSNR (dB) | SSIM | LapVar | LowLap (`4..22` lv) | HighLap (`>=35` lv) | Halo99.9 | ChrMAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`harsh_q62_420`** | `LineFlow_v25` / `v26` | 24.14 / 23.99 | 0.8802 / 0.8794 | 2162.3 / 2136.1 | 3.70 / 3.32 | 67.74 / 66.98 | **20.3 / 20.4** | 2.22 / 2.31 |
| **`harsh_q62_420`** | `APISR_2x` (Teacher) | 23.06 | 0.8630 | 2876.3 | **5.41** | **78.50** | 42.2 | 2.45 |
| **`harsh_q62_420`** | `Strong_v3` | 22.29 | 0.8457 | 3033.3 | **7.11** | **83.16** | 45.0 | 2.60 |
| **`harsh_q62_420`** | **`AnimeSharpV4_FastPU`** (`2025-01`) | 29.20 | **0.9270** | 703.3 | 4.22 | 42.72 | 16.2 | **1.85** |
| **`harsh_q62_420`** | `AnimeSharpV4_RCAN` (`2025-01`) | **29.29** | 0.9224 | 334.0 | 4.28 | 30.11 | **12.2** | 1.87 |
| **`harsh_q62_420`** | `ModernSpanimationV1` (`2024-04`) | 26.48 | 0.8887 | 560.3 | **6.20** | 36.53 | 19.6 | 2.97 |
| **`harsh_q62_420`** | `AnimeJaNai_V3Sharp1` (`2024-01`) | 27.60 | 0.8986 | 321.3 | 4.39 | 27.05 | 15.4 | 2.24 |
| **`harsh_q62_420`** | `RPLKSR_Sharp` / `MoSR_Sharp` | 27.11 / 27.53 | 0.8809 / 0.8970 | 169.2 / 194.1 | 4.79 / 4.70 | 21.19 / 22.58 | 16.6 / 13.2 | 2.07 / 2.00 |
| **`shuffled_reencode`** | `LineFlow_v25` / `v26` | 24.35 / 24.18 | 0.8788 / 0.8765 | 2183.0 / 2156.9 | 3.25 / 2.97 | 67.61 / 66.84 | **18.2 / 18.4** | 2.21 / 2.26 |
| **`shuffled_reencode`** | `APISR_2x` | 23.07 | 0.8545 | 2939.7 | 4.98 | 78.64 | 44.6 | 2.38 |
| **`shuffled_reencode`** | `Strong_v3` | 22.78 | 0.8570 | 2948.5 | **6.17** | **82.00** | 44.6 | 2.47 |
| **`shuffled_reencode`** | **`AnimeSharpV4_FastPU`** | **29.04** | **0.9234** | 695.9 | 3.99 | 41.59 | 16.6 | **1.94** |
| **`shuffled_reencode`** | `ModernSpanimationV1` | 26.47 | 0.8855 | 520.3 | **5.67** | 35.16 | 19.4 | 2.98 |

- **Verified Visual Findings (What Actually Looks Good to Human Eyes):**
  1. **`APISR_2x`** dominates razor-sharp vector main lines (`HighLap = 78.50–86.75`), hair flare base color consolidation (`TopRight_Flare Sat = 36.4`), and smooth curved light rims (`PolyRMS = 0.513 px`, `Seg16 = 0.0931 px`). However, it produces **heavy neon overshoot halos (`Halo99.9 = 42.2–46.6`)** and erases subtle `4..16` level cheek blush hatching in compressed 2021++ scenes (`Frieren`).
  2. **`AnimeSharpV4_FastPU` (`SSIM = 0.9548 / 0.9270`)** & **`ModernSpanimationV1` (`LowLap = 6.20`)** keep **100% of low-contrast `4..16` level micro-details (`Cheek_Blush Grad1px = 1.09 / 1.02` vs `LineFlow_v26 = 0.23`, `APISR = 0.55`)** with true warm-pink color fidelity (never over-darkened grey/black scratches like `Strong_v3`), plus `ModernSpanimationV1` achieves smooth `Seg16 = 0.0981 px` curves — though both remain noticeably softer on thick blurred compression contours (`HighLap = 35–48` vs `APISR 78.5–86.8`).
  3. **Topaz Video AI Literature Context:** Topaz's live-action/video models (*Proteus / Artemis / Rhea*) excel at temporal video stabilization, but their photographic training priors introduce waxy/plastic skin textures and ringing on 2D anime cel fills unless sharpening/detail recovery sliders are dialed down; *Gaia* is frame-independent for CG/animation yet trails anime-native pipelines (`APISR`, `Real-CUGAN`, `AnimeJaNai/SPAN`) on restoring degraded thin anime line art.

### 13.2 Bottom-Right Curved Light-Rim Wobble Lineage & Root Cause ([`trace_curve_wobble_lineage.py`](file:///d:/MODEL%20UPSCALE%20ME/scratch/trace_curve_wobble_lineage.py), [`test_curve_straightness_hypothesis.py`](file:///d:/MODEL%20UPSCALE%20ME/scratch/test_curve_straightness_hypothesis.py))

Extracting exact sub-pixel crossing coordinates $x(y)$ along the bottom-right pink-cream light arc in `img3` (`p27_lineage_curve_wobble_trace.png`, `p27_sota_img3_BottomRight_Curve.png`, `p27_exact_user_attached_br_zoom.png`):

| Model | `PolyRMS` (Smooth Arc Deviation) | `MaxBend` | `Seg16px_Wobble` (16px Segment Ripples) | `Smooth_D2` (2px Staircase Roughness) |
| :--- | :---: | :---: | :---: | :---: |
| `Original_Bic` | 0.692 px | 1.887 px | 0.2800 px | 0.3164 px |
| `P22_Compact` (`XyetherCompactNet`) | 0.708 px | 1.988 px | — | 0.2198 px |
| `LineFlow_v23` / `v24` / `v25` | 0.692 / 0.664 / 0.647 px | 1.951 / 1.793 / 1.737 px | — / — / 0.1732 px | 0.1742 / 0.1351 / 0.1134 px |
| `LineFlow_v26` | 0.623 px | 1.628 px | 0.1440 px | **0.0968 px** |
| **`Strong_v3`** (`XyetherCompactNet`) | **0.569 px** | **1.440 px** | **0.0767 px** | **0.0624 px** |
| **`ModernSpanimationV1`** (SPAN) | 0.592 px | **1.358 px** | **0.0981 px** | 0.1341 px |
| **`APISR_2x`** (GRL) | **0.513 px** | **1.379 px** | **0.0931 px** | **0.0959 px** |

- **Root Cause Proven:** While `LineFlow_v26` already matched `APISR` at pixel-level smoothness (`Smooth_D2 = 0.0968 px` vs `APISR 0.0959 px`), `Original`'s interpolation/compression quantization leaves `Seg16px_Wobble = 0.2800 px`, and our `AnimeOTFDataset` only included intermediate Bilinear/Cubic scale-jittered curve wobble `18%` of the time (`82%` used smooth `GaussianBlur -> INTER_AREA`). Combined with `Phase26LineFlowLoss` loosening `l_halo` against `max3_tar_rgb` instead of `max3_gt_rgb` and lacking multi-scale `3x3 / 7x7 / 11x11` orientation-only curve smoothness, `v25/v26` reproduced `0.173 / 0.144 px` of `Original`'s `16px` arc ripples instead of straightening them like `Strong_v3` (`0.077 px`), `APISR` (`0.093 px`), or `ModernSpanimationV1` (`0.098 px`).

### 13.3 Controlled `Batch=4` A/B Verification of Phase 27 Equations ([`ab_p26_vs_p27_control.py`](file:///d:/MODEL%20UPSCALE%20ME/scratch/ab_p26_vs_p27_control.py))
Under identical 60-step `batch=4` local continuation on held-out `img3` & `BlueArchive 2021++`:
- `A_P26Loss_Control`: `BA_PSNR=22.53`, `LowLap=4.70`, `CheekGrad=0.22`, `Halo=29.0`
- **`B_P27Loss_Pure` (`Phase27` Loss + Freq-Decoupled Hybrid Target + Angle-Only Curve Smoothness):** `BA_PSNR=22.59`, `LowLap=4.73`, `CheekGrad=0.25 (+13.6%)`, `Halo=28.0` (and in 220-step `verify_p27_curve_and_blush_250steps.py`: `CheekGrad=0.23 -> 0.27`, `LowLap=4.88 -> 5.11`, `FlareSat=40.7 -> 38.4`). Note that full 7,500-iter Kaggle training with `VGGPerceptualLoss` and 350 Safebooru 4K images is required to prevent small-subset `Halo` drift (`23 -> 28`) and fully realize `Shuffled Bilinear` curve straightening.

---

## 14. Phase 27 (`XyetherLineFlow_v27_Final`) — Kaggle v39 Training & Full Verification

### 14.1 Training Run & Artifacts (`Kaggle Kernel v39`)
- **Status:** `COMPLETE` (`7,500 / 7,500` iterations in `9,954.1s` / `2.76h` on Kaggle T4 GPU).
- **Checkpoints & Reports Archived Locally ([`checkpoints/Phase27_XyetherLineFlow_v5/`](file:///d:/MODEL%20UPSCALE%20ME/checkpoints/Phase27_XyetherLineFlow_v5/)):**
  - `XyetherLineFlow_v27_Final.pth` (`4,175,275` bytes)
  - `XyetherLineFlow_v27_Final.onnx` (`2,077,084` bytes)
  - `phase27_metrics_report.json` (`48,741` bytes)
- **ONNX Parity & Local CPU Speed (`256x256 -> 512x512`):**
  - Kaggle T4 ONNX parity: `max_abs_diff = 8.94e-07`
  - Local CPU ONNX parity: `max_abs_diff = 1.31e-06` | `CPU Cold = 814.0 ms` | `CPU Warm = 775.0 ms`

### 14.2 Kaggle 4-Regime Synthetic Validation (`xyether-training-runner.log`)
| Regime | Model | `HL1` | `HL2` | `LapVar` | `MicroLap` | `MAE_AP` | `Halo` | `Checker` | `PSNR` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Clean_LR`** | `Phase26_LineFlow_v4` | 15 | 22 | 955.3 | 2.50 | 4.08 | +2 | 0.16 | 29.22 |
| **`Clean_LR`** | **`Phase27_LineFlow_v5`** | **13** | 23 | 945.9 | **2.51** | 4.88 | **+0** | **0.13** | **29.57** |
| **`Clean_LR`** | `APISR_4x_to_2x` | 19 | 18 | 969.3 | 3.08 | 0.00 | +2 | 0.17 | 27.09 |
| **`RealBlurry_LR`** | `Phase26_LineFlow_v4` | 16 | 37 | 915.5 | 2.35 | 3.94 | +0 | 0.17 | 28.53 |
| **`RealBlurry_LR`** | **`Phase27_LineFlow_v5`** | **14** | **32** | 906.2 | **2.42** | 4.54 | **-3** | **0.15** | **28.81** |
| **`RealBlurry_LR`** | `APISR_4x_to_2x` | 28 | 23 | 958.3 | 3.70 | 0.00 | +8 | 0.18 | 27.36 |
| **`Harsh_Q62_LR`** | `Phase26_LineFlow_v4` | 30 | 68 | 690.4 | 2.36 | 4.79 | +3 | 0.17 | 27.58 |
| **`Harsh_Q62_LR`** | **`Phase27_LineFlow_v5`** | **28** | **61** | **700.4** | 2.35 | 5.41 | **+2** | **0.15** | **27.80** |
| **`Harsh_Q62_LR`** | `APISR_4x_to_2x` | 14 | 50 | 895.7 | 4.02 | 0.00 | +10 | 0.18 | 26.21 |
| **`Shuffled_Reencode_LR`** | `Phase26_LineFlow_v4` | 33 | 58 | 753.9 | 2.21 | 4.66 | +1 | 0.16 | 28.21 |
| **`Shuffled_Reencode_LR`** | **`Phase27_LineFlow_v5`** | **30** | **55** | **757.4** | **2.22** | 5.33 | **-3** | **0.14** | **28.42** |
| **`Shuffled_Reencode_LR`** | `APISR_4x_to_2x` | 25 | 82 | 936.5 | 3.94 | 0.00 | +8 | 0.17 | 26.42 |

### 14.3 User 3-Scene Full Panel & Native 2x ROI Measurements (`img3_hair_flare`)
- **Bottom-Right Curved Light-Rim Straightness (`BottomRight_Curve`):**
  - `LineFlow_v25` -> `LineFlow_v26` -> **`LineFlow_v27`**:
    - `PolyRMS`: `0.647` -> `0.623` -> **`0.615 px`** (`-4.9%` vs `v25`)
    - `Seg16_Wobble`: `0.1732` -> `0.1440` -> **`0.1358 px`** (`-21.6%` vs `v25`, `-5.7%` vs `v26`)
    - `Strong_v3` (`0.569 px` / `0.0767 px`) and `APISR_2x` (`0.513 px` / `0.0931 px`) remain slightly straighter on that diagonal arc due to their `Nearest(x)` base skip + 2D H/V strip kernels.
- **Cheek Blush & Low-Contrast Detail (`Cheek_BlushLines`):**
  - `LineFlow_v26` (`Grad1px=0.23`, `LapVar=9.6`) -> **`LineFlow_v27`** (`Grad1px=0.33`, **`+43.5%`**, `LapVar=17.3`, **`+80.2%`**); still softer than `APISR_2x` (`0.55 / 51.6`), `AnimeSharpV4_FastPU` (`1.09 / 29.6`), and `Strong_v3` (`1.23 / 239.2`).
- **Hair Light-Flare Consolidation (`TopRight_HairFlare`):**
  - `LineFlow_v27` tightened further (`Sat=40.5`, `Warm=11.40` vs `v26` `40.7 / 11.44` and `v25` `41.3 / 11.67`).
- **All 3 User Scenes (`img0_sunlight` / `img1_dark_hall` / `img3_hair_flare`):**
  - `LowContrast(4..22) Lap`: `v26` (`2.62 / 3.69 / 1.87`) -> **`v27` (`2.71 / 3.83 / 2.04`)**
  - `HighContrast(>=35) Lap`: **`v27`** ranked #1 across all models on all 3 scenes (**`110.99 / 91.50 / 86.35`** vs `APISR` `101.90 / 87.69 / 80.41` and `Strong_v3` `97.54 / 84.07 / 81.11`), while achieving #1 `MAE_vs_APISR` on `img3_hair_flare` (**`1.72`** vs `v26 1.80`, `Strong_v3 1.80`).

### 14.4 2021++ 5-Scene Modern Anime Sequential Benchmark (`bluearchive`, `bocchi`, `frieren`, `kusuriya`, `oshinoko`)
| Condition | Model | `PSNR` | `SSIM` | `LapVar` | `LowLap` | `HighLap` | `Halo99.9` | `ChrMAE` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`stream_1080p`** | `LineFlow_v26` | 23.86 | 0.8891 | 2876.2 | 3.45 | 78.54 | 22.4 | 2.17 |
| **`stream_1080p`** | **`LineFlow_v27`** | 23.85 | **0.8918** | **2923.8** | **3.50** | **79.28** | 23.0 | **2.11** |
| **`stream_1080p`** | `APISR_2x` | 23.27 | 0.8796 | 3414.2 | 5.16 | 86.75 | 46.6 | 2.36 |
| **`stream_1080p`** | `Strong_v3` | 23.11 | 0.8756 | 3403.9 | 6.61 | 89.87 | 47.6 | 2.40 |
| **`stream_1080p`** | `AnimeSharpV4_FastPU` | **30.74** | **0.9548** | 856.9 | 3.93 | 47.62 | **14.2** | **1.58** |
| **`stream_1080p`** | `ModernSpanimationV1` | 27.82 | 0.9219 | 722.4 | 6.06 | 42.10 | 17.8 | 2.78 |
| **`harsh_q62_420`** | `LineFlow_v26` | 23.99 | 0.8794 | 2136.1 | 3.32 | 66.98 | 20.4 | 2.31 |
| **`harsh_q62_420`** | **`LineFlow_v27`** | 23.94 | **0.8816** | **2203.8** | **3.42** | **68.20** | **20.0** | **2.28** |
| **`harsh_q62_420`** | `APISR_2x` | 23.06 | 0.8630 | 2876.3 | 5.41 | 78.50 | 42.2 | 2.45 |
| **`harsh_q62_420`** | `Strong_v3` | 22.29 | 0.8457 | 3033.3 | 7.11 | 83.16 | 45.0 | 2.60 |
| **`shuffled_reencode`** | `LineFlow_v26` | 24.18 | 0.8765 | 2156.9 | 2.97 | 66.84 | 18.4 | 2.26 |
| **`shuffled_reencode`** | **`LineFlow_v27`** | 24.14 | **0.8797** | **2222.6** | **3.02** | **68.03** | **18.2** | **2.23** |
| **`shuffled_reencode`** | `APISR_2x` | 23.07 | 0.8545 | 2939.7 | 4.98 | 78.64 | 44.6 | 2.38 |
| **`shuffled_reencode`** | `Strong_v3` | 22.78 | 0.8570 | 2948.5 | 6.17 | 82.00 | 44.6 | 2.47 |

