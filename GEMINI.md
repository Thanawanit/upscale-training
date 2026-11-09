# Workspace Rules (`MODEL UPSCALE ME`)

## 1. Immediate Discovery & Benchmark Persistence (`.md`)
- Whenever any important root-cause analysis, mathematical/architectural breakthrough, dataset anomaly, or verified evaluation metric (`PSNR`, `SSIM`, `HL1/HL2`, `Halo`, `FlatBR`, `RedIn`) is discovered, **immediately document it in [`RESEARCH_AND_PROGRESS_LOG.md`](file:///d:/MODEL%20UPSCALE%20ME/RESEARCH_AND_PROGRESS_LOG.md)** in the workspace root.
- Do not wait until the end of a long task or risk losing findings across session boundaries.

## 2. Kaggle / Cloud Training Status Polling (30-Minute Rule)
- When waiting for a Kaggle kernel (`kaggle kernels status`) or cloud training run to complete, schedule status checks **every 30 minutes (`DurationSeconds: 1800`)** to minimize quota consumption.

## 3. Project-Specific Invariants
- **100% Own Student Checkpoint Lineage:** Student models must always be initialized strictly from our own checkpoints (e.g., `XyetherKiller_OwnTwinRefl_Init_v21.pth`, `XyetherKiller_APISRMaster_v21_Final.pth`). External models (`APISR.onnx`) may only be used as Teachers inside the loss function (`Option B`).
- **Scratch & Temp Redirection:** Always set `$env:TEMP="d:\MODEL UPSCALE ME\scratch"; $env:TMP="d:\MODEL UPSCALE ME\scratch"; $env:TORCH_HOME="d:\MODEL UPSCALE ME\scratch\torch_cache"` so Drive `C:` remains clean.
