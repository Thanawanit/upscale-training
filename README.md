# Xyether Anime Super-Resolution Reproduction & Training Suite

A mathematical 1:1 reproduction of the **Xyether Strong Series (v1.5 / v2.5 / v3)** and **Balanced 2x** anime super-resolution architecture.

---

## 1. Architectural Blueprint

| Property | Strong Series (v1.5 / v2.5 / v3) | Balanced 2x |
| :--- | :--- | :--- |
| **Topology** | Compact (`SRVGGNetCompact`) | UltraCompact |
| **PyTorch Trainable Params** | **600,652** | **304,716** |
| **ONNX Graph Initializers** | 600,658 (+6 bounds) | 304,722 (+6 bounds) |
| **Convolution Layers** | 18 Convs (1 In + 16 Body + 1 Out) | 10 Convs (1 In + 8 Body + 1 Out) |
| **Activation** | Learnable PReLU (64 ch/layer) | Learnable PReLU (64 ch/layer) |
| **Global Residual Mode** | **`Nearest Neighbor`** | **`Nearest Neighbor`** |
| **Upsampler** | `PixelShuffle(2x)` | `PixelShuffle(2x)` |
| **Output Clamping** | `[0.0, 1.0]` | `[0.0, 1.0]` |

### The Global Nearest Residual Mechanism
Standard Real-ESRGAN models attempt to generate the entire image from scratch using feedforward convolutions without a skip connection, or with Bicubic interpolation. Bicubic interpolation introduces negative sinc lobes, inherently causing **ringing artifacts and glowing white halos (Overshoot > 10.0)** around high-contrast 2D anime lines.

Xyether completely eliminates this mathematically:
$$\text{Output} = \text{clamp}(\text{Interpolate}(x, \text{scale}=2, \text{mode}=\text{'nearest'}) + \text{BodyCNN}(x), 0.0, 1.0)$$

Nearest neighbor interpolation has **zero negative lobes**, guaranteeing **Overshoot = 0.00**. The 18-layer CNN only computes the anti-aliasing residual (smoothing staircases) and flat-shading denoise delta.

---

## 2. Directory Structure

```text
d:\MODEL UPSCALE ME/
├── archs/
│   ├── __init__.py
│   ├── xyether_compact.py       # XyetherCompactNet (600,652 params) & XyetherBalancedNet (304,716 params)
│   └── discriminator.py         # UNetDiscriminatorSN with Spectral Normalization
├── losses/
│   ├── __init__.py
│   └── xyether_losses.py        # CharbonnierLoss, MSSSIMLoss, ColorLuvLoss, FocalFrequencyLoss, GANLoss
├── dataset/
│   ├── __init__.py
│   ├── otf_dataset.py           # On-The-Fly Degradation (Blur 0.4-3.2, Downsample 2x, Noise 2-20, JPEG 40-90)
│   └── download_dataset.py      # Automated dataset downloader with calibration frame generation
├── benchmark/
│   ├── __init__.py
│   └── xyether_evaluator.py     # Quantitative Test Rig (Overshoot, Noise Residual, Line Depth, Sharpness)
├── weights/
│   ├── __init__.py
│   └── transplant.py            # Weight surgery tool (net_g_89000 -> xyether_transplanted_base.pth)
├── train_phase1.py              # Phase 1: Fidelity & Denoise Base Training
├── train_phase2.py              # Phase 2: Sharp Polish Training (FFLoss + D-UNet)
├── Colab_Xyether_Training_Suite.ipynb # Ready-to-run Google Colab Pro notebook
└── README.md
```

---

## 3. Training Curriculum (Two-Phase Execution)

### Phase 1: Fidelity Base (Strong v2.5 behavior)
* **Objective:** Teach the network to eliminate nearest-neighbor staircasing, preserve pitch-black lines, and eliminate up to $\sigma = 20.0$ noise.
* **Loss Configuration:**
  * `CharbonnierLoss`: Weight **1.0**
  * `MSSSIMLoss`: Weight **0.55**
  * `ColorLuvLoss`: Weight **0.80**
  * `PerceptualLoss` & `GANLoss`: **OFF (0.0)**
* **Optimizer:** AdamW (`lr=1e-4`, `betas=[0.9, 0.999]`, `weight_decay=0.0`)
* **EMA Decay:** `0.9995`
* **Target Iterations:** 10,000 – 15,000 steps

### Phase 2: Sharp Polish (Strong v3 evolution)
* **Objective:** Polish Conv 17 and Conv 18 with high-frequency guidance without disturbing the body's 91.5% noise reduction.
* **Loss Configuration:**
  * `CharbonnierLoss`: Weight **0.50**
  * `MSSSIMLoss`: Weight **0.50**
  * `ColorLuvLoss`: Weight **0.70**
  * `FocalFrequencyLoss` (FFLoss): Weight **0.05**
  * `GANLoss` (D-UNet): Weight **0.10**
* **Optimizer:** AdamW (`lr=3e-5`)
* **Target Iterations:** 4,000 – 6,000 steps

---

## 4. Signal Verification Benchmarks

| Metric | Target Specification | Measurement Method |
| :--- | :--- | :--- |
| **Overshoot / Halo** | **`0.00`** | Edge transition peak intensity check |
| **Noise Residual (In=10)** | **`< 1.0` (91.5% eliminated)** | Output std on flat skin tone with input $\sigma=10$ |
| **Line Depth** | **`<= 8.0` (Pitch Black)** | Minimum luma along black contour lines |
| **Sharpness** | **`18,000 - 25,000`** | Laplacian Variance on test frames |

---

## 5. How to Run on Google Colab Pro

1. Upload the project folder or open `Colab_Xyether_Training_Suite.ipynb` in Google Colab Pro.
2. Select **GPU: Nvidia L4 or A100** (`Runtime -> Change runtime type -> Hardware accelerator -> GPU -> L4`).
3. Run the notebook cells sequentially:
   * **Cell 1:** Connects Google Drive for zero-loss checkpoint persistence.
   * **Cell 2:** Performs weight surgery on `net_g_89000.pth`.
   * **Cell 3:** Launches Phase 1 training with live benchmark metrics.
   * **Cell 4:** Automatically loads Phase 1 and runs Phase 2 Sharp Polish.
   * **Cell 5:** Runs the evaluation rig and exports final `.pth` and `.onnx` models.
