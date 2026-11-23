# Operational Handover Context (`MODEL UPSCALE ME` -> Phase 30)

Compact operational state for continuing in a new session. **Do not duplicate historical benchmarks or equations**; follow the section pointers in [`RESEARCH_AND_PROGRESS_LOG.md`](RESEARCH_AND_PROGRESS_LOG.md).

---

## 1. Benchmark & Discovery Pointers (`RESEARCH_AND_PROGRESS_LOG.md`)
- **Phases 20–22 (CNN `TwinReflMaster` -> `APISR` Self-Bounded Valley Teacher):** Sections 1–6
- **Phases 23–27 (`XyetherLineFlowNet` from-scratch -> curve-smooth & micro-rescue `v27`):** Sections 7–14 (`LineFlow_v27` = crisp baseline `line_depth = 0.0`, 10-model 2021++ SOTA benchmark in Sec. 13.1)
- **Phase 28 (`XyetherRepHybrid_v28` Kaggle v40 & regression diagnosis):** Sections 15–16.3 (why random/warmstart-decayed `RepHybrid` + Pure-GT blurred H.264 lines `line_depth=26.53`, and why broad `gt_y` masking collapses `v27` sharpness)
- **Colab Vector-GAN, `4x->2x` SSAA, `Keep8B` vs `Pruned6B_Leaky` speed/quality:** Sections 16.4–16.5 (`8B FP32 = 277.5 ms`, `8B FP16 = 242.1 ms`, `6B_Leaky FP16 = 187.9 ms` on T4 CUDA EP)
- **Phase 29 (`XyetherLineFlow_v29` Kaggle v41 final 4-regime metrics & artifacts):** Section 17
- **Official reference model specs (`Strong_v3`, `Balanced_2x`):** `XYETHER_OFFICIAL_SPECS.md`

---

## 2. Active Codebase & Exact Signatures

### Primary Architecture (`archs/xyether_lineflow.py`) — Active Production Backbone
- `LineFlowBlock(dim: int, strip_k: int = 9, dilation: int = 1, res_scale: float = 0.25)` (`L6-33`): `conv_in(3x3)` -> `PReLU(dim, init=0.2)` -> `h + 0.5*(dw_h(1,9) + dw_v(9,1))` (`groups=dim`) -> `PReLU` -> `conv_mix(3x3)` -> `x + res_scale * out`.
- `XyetherLineFlowNet(num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 56, num_blocks: int = 8, upscale: int = 2)` (`L35-96`, **`514,148` params**): alternating dilations `1,2`, `tap_set={1,3,5,7}` -> `fusion(224->56, 1x1 + PReLU + 3x3)` -> `fused * Sigmoid(out_gate) + f0` -> `PixelShuffle(2)` -> `clamp(nearest(x) + residual, 0, 1)`.
- **Experimental / Reference Architecture (`archs/xyether_rephybrid.py`):** `RepConvK(dim, k=3)` (`L8-67`, `switch_to_deploy()`) and `XyetherRepHybridNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_blocks=10, k5_indices=(2,5,8), upscale=2, init_res_gain=0.15)` (`L69-131`, `574,604` deployed params). Fast GPU throughput (`~91 ms` T4), but blurred when trained without `LineFlow` weight continuity (Sec. 16.3).

### Loss Modules (`losses/xyether_losses.py`)
- `Phase23ZeroHaloAPISRTeacher(onnx_path="./reference_models/Model/APISR.onnx", full_color_consolidation=False, freq_micro_rescue=False, unclipped=False)` (`L1194-1280`): computes `APISR(F.avg_pool2d(gt_t, 2, 2)) -> 4x->2x` via `onnx2torch` or `onnxruntime`. When `unclipped=True` (`loss_version >= 29`), skips `min(ap_t, max3_gt)` clamping.
- `PatchVectorDiscriminator(nf: int = 48)` (`L1995-2015`): 5-layer `spectral_norm` + `LeakyReLU(0.2)` PatchGAN discriminator.
- `LPIPSPerceptualLoss(loss_weight: float = 0.45)` (`L2018-2047`): AlexNet `lpips.LPIPS` in `[-1, 1]` with lazy `VGGPerceptualLoss` fallback.
- `Phase29VectorCleanLoss(Phase27LineFlowLoss)` (`L2050-2135`):
  - `forward(pred, tar_b, gt_b, lr_b, edge_zone=None, regime=None, warmstart_out=None, warmstart_weight=0.0)`
  - Gates cel bilateral denoise strictly to `flat_mask = (span_ap < 0.018)` to keep thin strokes sharp.
  - Loss sum: `l_pix` (`1.35*Charbonnier + 0.90*|py-ay|`) + `2.4*l_ink` + `2.6*l_grad` + `1.35*l_lap` (amplitude + directional Laplacian weighted by `w_edge = 0.50 + 2.50*is_edge*(1+0.5*is_deg)`) + `2.0*l_curve` (`_angle_only_curve_smoothness`) + `1.0*l_cel` + `2.4*l_halo` (capped at `max3_ap + 0.002` / `min3_ap - 0.003`).
- Also present: `Phase28DegradGatedSWTLoss` (`L1833-1992`), `StationaryHaarSWT2D` (`L1809-1830`), `ColorLuvLoss` (`L94-143`), `Phase23–27LineFlowLoss`.

### Training, Dataset & Evaluation
- **`train_xyether_p23.py` (`train_p23` at `L197`, CLI `L447-489`):**
  - Core helpers: `ModelEMA` (`L40`), `export_onnx` (`L84`, opset 17 dynamic `batch/height/width`), `export_fp16_onnx` (`L113`, `onnxconverter_common.float16.convert_float_to_float16(..., keep_io_types=False)`).
  - Current Phase 29 Kaggle v41 CLI config (`Kaggle_Xyether_Training.ipynb` Cell 4):
    `--arch lineflow --resume_path .../XyetherLineFlow_v27_Final.pth --loss_version 29 --num_feat 56 --num_blocks 8 --batch_size 12 --patch_size 160 --total_iters 6500 --warmup_iters 200 --lr_g 1.5e-4 --lr_d 1.2e-4 --min_lr 8e-6 --luv_weight 0.18 --lpips_weight 0.45 --vgg_weight 0.0 --gan_weight 0.08 --ema_decay 0.9992 --export_fp16 --eval_every 1500 --save_every 2500`.
  - Discriminator & `LPIPS` & `ColorLuvLoss` execute in FP32 outside `autocast` (`ColorLuvLoss` targets `0.70*tar_b + 0.30*gt_b` when `is_p29`).
- **`dataset/otf_dataset.py`:** `AnimeOTFDataset(data_dir: str, patch_size: int = 192, repeat: int = 20, return_regime: bool = False)` (`L10-263`; note `train_xyether_p23.py` passes `repeat=40, return_regime=(loss_version >= 28)`). Regimes: `0=Clean(20%), 1=Medium YUV_I420 Stream(30%), 2=Harsh Q52..74 YUV_I420(20%), 3=Shuffled-Resize Curve-Jitter WebP/JPG(30%)`, plus BT.601 Warm<->Cool opponent reflection. `dataset/download_dataset.py` (`download_safebooru_anime(target_path, max_pages=6)`).
- **`benchmark/xyether_evaluator.py`:** `XyetherEvaluator(device='cpu')` -> `evaluate_model(model, test_images=None)` (`L42-125`): synthetic `0.88/0.12` step edge (`overshoot_halo`), `In=10` noise (`noise_residual_std`), 2px line (`line_depth`), and `Laplacian` variance.
- **Kaggle Runner:** `Kaggle_Xyether_Training.ipynb` (6 cells; clones `Thanawanit/upscale-training.git`, runs `train_xyether_p23.py`, evaluates 4 regimes on `test_anime.png`, checks ONNX FP32 & FP16 parity, uploads via `HF_TOKEN`), `kernel-metadata.json` (`id: thanawnit/xyether-training-runner`, GPU + Internet enabled).
- **Legacy Modules (Pre-Phase 23, reference only):** `train_phase1.py`, `train_phase2.py`, `train_xyether.py`, `archs/xyether_compact.py` (`XyetherCompactNet`, `XyetherBalancedNet`, `SRVGGNetCompact`), `archs/discriminator.py` (`UNetDiscriminatorSN`), `weights/transplant.py`, `cloud_manager.py`.

### Current Artifacts & Checkpoints (Untracked Binary Folders + HuggingFace)
- **Active Best Model (`checkpoints/Phase29_XyetherLineFlow_v1/`):**
  - `XyetherLineFlow_v29_Final.pth` (`4,175,275 B` / `3.98 MB`, `params_ema`)
  - `XyetherLineFlow_v29_Final.onnx` (`2,077,084 B` / `1.98 MB`, FP32 parity `1.13e-06`)
  - `XyetherLineFlow_v29_Final_FP16.onnx` (`1,057,372 B` / `1.01 MB`, FP16 parity `1.34e-03`)
  - `phase29_metrics_report.json` (`48,752 B`)
  - HF Mirror: `Thanawanit/Kaggle-Backup/checkpoints/Phase29_XyetherLineFlow_v1/` & `evaluations/Phase29_XyetherLineFlow/`
- **Base Anchor & Teacher:** `checkpoints/Phase27_XyetherLineFlow_v5/XyetherLineFlow_v27_Final.pth`, `reference_models/Model/APISR.onnx` (`6.49 MB`).
- **Verification Scripts (`scratch/`):** `verify_p29_pipeline.py` (Phase 29 E2E CPU smoke + loss/teacher check), `verify_p28_pipeline.py` (backward compat), `kaggle_logs_p29/` (full Kaggle v41 outputs & `xyether-training-runner.log`).

---

## 3. Current Session Delta & Phase 30 Open Hypotheses
- **Completed in This Session:** Phase 29 (`v41`) finished cleanly on Kaggle (`6,500` iters, `8,684.6s`), logged in Section 17. Additionally (post-Section 17), generated 2025–2026 visual comparisons (`25` source frames in `scratch/frames_2025plus/`: `H264` 6-scene & `CLEAN` 4-scene grids in session artifacts `p29_2025plus_h264_compare.jpg` / `p29_2025plus_clean_compare.jpg`), confirming visibly crisper `Frieren S2` / `JJK S3` / `DanDaDan S2` vector lines than `v27` and cleaner cel surfaces than `APISR_2x`.
- **Quantified Gaps Remaining (Target Candidates for Phase 30, spec NOT yet locked):**
  1. **Missing `v29` Multi-Scene Baseline:** `v29` has NOT yet been quantitatively benchmarked on the `17`-scene curated set (`scratch/frames_2025plus_clean/` + `scratch/eval28_*`) or `25`-scene `scratch/frames_2025plus/`, and `XyetherLineFlow_v29_Final_FP16.onnx` has NOT been directly latency-benchmarked on Colab T4 (`242.1 ms` in Sec. 16.5 was measured on `v27_fp16`).
  2. **`Harsh_Q62` Trade-off in `test_anime.png` (Sec. 17):** `LineFlow_v29` cut `HL2` `61 -> 43` (beating `APISR=50`), **but** raised `Halo` (`+2 -> +16`, vs `APISR +10`) and still trails `APISR` slightly on `Harsh_Q62 LapVar` (`796.1` vs `895.7`) & `Shuffled LapVar` (`850.4` vs `936.5`).
  3. **Colab T4 Speed Margin:** `8B FP16` (~`242 ms`) is slower than `Pruned6B_Leaky FP16` (`187.9 ms`, `-32.3%` vs FP32, `0.81 MB`), which can be initialized directly from `XyetherLineFlow_v29_Final.pth` if Phase 30 targets higher throughput alongside quality.

---

## 4. Hard Constraints & Guardrails
1. **100% Own Student Lineage:** Always initialize student weights strictly from our own checkpoints (`XyetherLineFlow_v29_Final.pth` or `v27_Final.pth`). Never train `RepHybrid`/random-init replacements from scratch without proving `line_depth = 0.0` preservation first.
2. **Teacher Purity & Line Masking:** `APISR.onnx` is Teacher-only (`Option B`) and must ALWAYS be inferred on **clean** `F.avg_pool2d(gt_t, 2, 2)` — never degraded LR. Never apply unmasked `gt_y` or unmasked `l_cel` across high-contrast contours (`span >= 0.018`), which immediately softens `v27`/`v29` 1px dark lines.
3. **Architecture & Halo Bounds:** Maintain `nearest(x)` skip + `[0, 1]` clamp (or verify `SSAA` alternative rigorously on `h264_yuv420` first). Bound overshoots against `APISR` 3x3 max (`<= +0.002`) so highlights do not ring or clip neon white.
4. **Local Environment (CPU-Only):** Local machine has `torch 2.12.0+cpu` (`cuda: False`), `onnxruntime` (CPU), `lpips`, `huggingface_hub`, `skimage` — **`onnx` and `onnxconverter_common` are NOT installed locally**. Always redirect `$env:TEMP`, `$env:TMP`, `$env:TORCH_HOME` to `d:\MODEL UPSCALE ME\scratch` to keep Drive `C:` clean.
5. **Cloud Execution Rules:**
   - **Git-First Rule:** `Kaggle_Xyether_Training.ipynb` clones `https://github.com/Thanawanit/upscale-training.git`; ALWAYS `git commit` + `git push origin main` BEFORE `kaggle kernels push -p .`.
   - **Kaggle:** `12h` session cap; poll `kaggle kernels status thanawnit/xyether-training-runner` every **`1800s` (30 min)**; install `onnx`, `onnxconverter-common`, `onnxruntime-gpu==1.23.2` (CUDA 12) + call `ort.preload_dlls(cuda=True, cudnn=True)` to prevent silent `CPUExecutionProvider` fallback.
   - **Colab (`colab-mcp`):** Keep each `run_code_cell` `<130s` (`3m` timeout disconnects kernel); clear PyTorch VRAM (`gc.collect(); torch.cuda.empty_cache()`) before ORT sessions to prevent `BFCArena` OOM; NEVER `display(IPyImage)` large PNGs (export Q92 JPEG base64 in `3000`-char chunks).
6. **Binary & Secret Hygiene:** Never commit `.pth`, `.onnx`, `scratch/`, or `kaggle_logs*` to Git; never expose raw `HF_TOKEN` literals in docs or logs (use `os.environ["HF_TOKEN"]`).

---

## 5. Immediate Next Action (First Turn in New Chat)
1. **Local Multi-Scene Quantitative Baseline (17/25-Scene `2025++` Eval):**
   - Inspect `scratch/` evaluation layout (`scratch/frames_2025plus_clean/` `[17 frames]` + `scratch/eval28_{clean_540p,h264_yuv420,curvejit_webp}` `[000.png..016.png]`, plus `scratch/frames_2025plus/` `[25 frames]`).
   - Run an evaluation script comparing `checkpoints/Phase29_XyetherLineFlow_v1/XyetherLineFlow_v29_Final.pth` against `XyetherLineFlow_v27_Final.pth` and `reference_models/Model/APISR.onnx` on all 3 regimes (`PSNR`, `SSIM`, `LPIPS`, `Halo`, `LowR`, `HiR`, `Chr`, `Chk`) and `XyetherEvaluator`.
2. **Colab T4 `FP16 ONNX` Real Benchmark (when Colab connected):**
   - Load `checkpoints/Phase29_XyetherLineFlow_v1/XyetherLineFlow_v29_Final_FP16.onnx` (`1.01 MB`) on Colab T4 `CUDAExecutionProvider` (`540x960 -> 1080x1920`) to verify exact T4 inference latency (`ms`) and FP16/FP32 output parity (`LineFlow_v29` vs `v27` vs `Pruned6B_Leaky` probe).
3. **Lock Phase 30 Spec With User Before Editing Code:**
   - Using the `2025++` metrics + T4 latency + `Harsh_Q62` (`Halo=+16`, `LapVar=796.1`) diagnostic, propose concrete Phase 30 spec choices (e.g., `v29 -> 6B_Leaky` distillation/pruning for `<190 ms` speed vs `8B` Harsh-Q62 Anti-Halo + crispness tuning inside `losses/xyether_losses.py` `Phase29VectorCleanLoss` at `L2050` / `train_xyether_p23.py` `L197`).
