# 🔬 Xyether Series - Official Model Specifications & Deep Analysis

> Reference Source: [https://huggingface.co/Thanawanit/Upscale/resolve/main/Model.zip](https://huggingface.co/Thanawanit/Upscale/resolve/main/Model.zip)  
> PyTorch Converted Checkpoints: [https://huggingface.co/Thanawanit/Kaggle-Backup/tree/main/checkpoints/Official_Xyether_References](https://huggingface.co/Thanawanit/Kaggle-Backup/tree/main/checkpoints/Official_Xyether_References)

---

## 1. Overview of Extracted Models

The official `Model.zip` package contains 10 production upscaling models in FP16 ONNX format:

| Model Name | Format | File Size | Parameters | Topology | Scale |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strong_v3.onnx** | ONNX FP16 | 1.15 MB | 600,658 | SRVGGNetCompact + Nearest Residual | 2x |
| **Strong_v2.5.onnx** | ONNX FP16 | 1.15 MB | 600,658 | SRVGGNetCompact + Nearest Residual | 2x |
| **Strong_v1.5.onnx** | ONNX FP16 | 1.15 MB | 600,658 | SRVGGNetCompact + Nearest Residual | 2x |
| **Balanced_2x.onnx** | ONNX FP16 | 0.59 MB | 304,722 | SRVGGNetUltraCompact + Nearest Residual | 2x |
| **RealESRGAN_2x.onnx** | ONNX FP16 | 1.15 MB | 600,658 | Compact Architecture | 2x |
| **RealESRGAN_4x.onnx** | ONNX FP16 | 2.33 MB | 1,200,000 | Compact Architecture | 4x |
| **Ani4Kv2_2x.onnx** | ONNX FP16 | 2.30 MB | ~1,200,000 | Enhanced Anime SR | 2x |
| **LiveActionV1_2x.onnx** | ONNX FP16 | 1.58 MB | ~800,000 | Photorealistic SR | 2x |
| **Adore_2x.onnx** | ONNX FP16 | 2.73 MB | ~1,400,000 | High-contrast Anime SR | 2x |
| **4xNomos8kDAT.onnx** | ONNX FP32 | 85.81 MB | ~22,000,000 | Dual Aggregation Transformer (DAT) | 4x |

---

## 2. Deep Architecture Specifications (Strong Series)

All models in the **Strong Series (v1.5, v2.5, v3.0)** share an identical network graph:

```
Input Image (1, 3, H, W) [Range 0.0 - 1.0, FP16/FP32]
  │
  ├───► [Nearest Neighbor 2x Upsample] ──────────────────────┐ (Base Branch)
  │                                                          │
  └───► Conv2d(3, 64, kernel=3, stride=1, padding=1)        │
        PReLU(num_parameters=64)                             │
        ├── Conv2d(64, 64, kernel=3, stride=1, padding=1)    │
        │   PReLU(num_parameters=64)                         │
        │   [Repeated for 16 intermediate Body layers]       │
        Conv2d(64, 12, kernel=3, stride=1, padding=1)        │
        PixelShuffle(upscale_factor=2)                       │
        └── Residual Delta (1, 3, 2H, 2W) ───────────────────┤ (Residual Branch)
                                                             ▼
                                                    Add (Base + Delta)
                                                             │
                                                    Clip(min=0.0, max=1.0)
                                                             │
                                                             ▼
                                                    Output (1, 3, 2H, 2W)
```

- **Exact Layer Count:** 18 Convs, 17 PReLUs, 1 PixelShuffle, 1 Nearest Resize, 1 Add, 1 Clip.
- **Total Weights:** 53 trainable parameter tensors in PyTorch.
- **Parameters:** 600,652 in PyTorch (`params`) / 600,658 in ONNX graph (+6 scalar bound initializers).

---

## 3. Weight Evolution Analysis: Base 89k vs Official Strong v3

Comparison between the base training checkpoint (`net_g_89000.pth`) and official `Strong_v3.onnx`:

1. **Convolution Filter Weights:**
   - Layer weights (`body.X.weight`) are remarkably close (Mean Absolute Difference $\approx 0.035$).
   - This proves `Strong_v3` was initialized and derived from the exact same line of compact checkpoints as `net_g_89000.pth`.
2. **Biases & PReLU Threshold Shifts:**
   - `body.0.bias` to `body.14.bias` show intentional positive bias shifts ($+1.5$ to $+4.0$).
   - PReLU activation slopes in early and middle layers were optimized to suppress noise while amplifying dark line contrasts.
3. **Visual Output Comparison:**
   - **Base 89k:** Clean, but line edges remain soft; subtle residual noise in flat anime skin.
   - **Our Master Step 6k:** Preserves color accuracy (brightness 130.4, no cyan tint), but line sharpness is still closer to Base 89k.
   - **Official Strong v3:** Needle-sharp line art, pitch-black line ink, zero halo, rich contrast.

---

## 4. PyTorch Converted Checkpoints

The official ONNX models have been extracted and mapped into PyTorch `state_dict` matching `XyetherCompactNet`:

- `checkpoints/Official_Xyether_References/Xyether_Strong_v3_Official.pth` (2.31 MB, 53 layers, strict=True)
- `checkpoints/Official_Xyether_References/Xyether_Strong_v2.5_Official.pth` (2.31 MB, 53 layers, strict=True)
- `checkpoints/Official_Xyether_References/Xyether_Strong_v1.5_Official.pth` (2.31 MB, 53 layers, strict=True)
- `checkpoints/Official_Xyether_References/Xyether_Balanced_2x_Official.pth` (1.17 MB, 29 layers, strict=True)

All checkpoints are hosted publicly on Hugging Face repo: `Thanawanit/Kaggle-Backup`.

---

## 5. Continued Training Strategy

Since we now have the **Official Strong v3.0 PyTorch weights**:

1. **Starting Point:** We no longer need to train from the older/softer `net_g_89000.pth`. We can start directly from `Xyether_Strong_v3_Official.pth`!
2. **Teacher-Student Distillation:** We can use `Xyether_Strong_v3_Official.pth` as a frozen Teacher model to guarantee that our model matches its sharpness while tuning custom degradation envelopes.
3. **Targeted Fine-Tuning:** Fine-tuning with small learning rate ($2 \times 10^{-5}$) starting from `Xyether_Strong_v3_Official.pth` ensures we never lose the razor-sharp line quality.
