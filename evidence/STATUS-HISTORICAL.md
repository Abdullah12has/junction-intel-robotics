## Latest update — 14 Nov 2026, 13:42 Helsinki

GPU ACT smoke training and OpenVINO runtime verification completed. See [GPU training recovery](../docs/GPU-TRAINING-RECOVERY.md), [training evidence](act-training-completed.json), and [export evidence](act-openvino-verification.json). This supersedes older missing-dataset/training statements below. Seven saved episodes; one full epoch on Intel GPU; no physical policy evaluation yet.

# Build status

Updated 2026-11-14 12:45 Helsinki. Nothing below has moved the arm.

| Piece | State | Evidence |
|---|---|---|
| Anomalib 2.7.0 env on laptop | Installed, `~/junction-cell/anomalib-env`, torch XPU sees the iGPU | `versions.txt` |
| Detector train → export → calibrate → held-out | Works end to end on **synthetic** images; static batch-1 FP16 OpenVINO model; heatmap localises the mark | PatchCore p50 5.7 ms iGPU, 46 ms CPU (synthetic model, 256×256) |
| Inspection camera capture | Not run; `CAM` and `ROI` in `cell/inspection.py` still placeholders | — |
| `cell/run_cell.py` | Protocol checked against backend source; `check`/`home`/`run` never run live | — |
| `cell/split_dataset.py` | Written, untested (no dataset yet) | — |
| Power sampler | Temperature, NPU busy %, iGPU busy % readable; **watts need one sudo chmod** | — |
| NPU | **Not usable**: kernel driver and firmware present, userspace driver missing, OpenVINO lists CPU and GPU only | `versions.txt` |
| Modal | Not authenticated on the laptop; local XPU training is the fallback | — |
| Studio datasets / models / jobs | None | — |
