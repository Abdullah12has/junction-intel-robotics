# Intel GPU training recovery — 14 November 2026

Original ACT job `5f3a6be6-067e-473f-900a-311e2d609c8d` stalled before its first training step. Kernel logs recorded GPU memory faults and a compute-engine timeout for its worker. A process stack inspection showed Lightning stuck moving tensors to CPU during exception cleanup; the original exception was not recovered.

A deterministic forward/backward check isolated incorrect arithmetic in Studio's PyTorch 2.11.0+xpu stack on this laptop. For a 1024×1024 all-ones matrix, `(a @ a).square().mean()` must be 1048576 and the mean gradient must be 4. Version 2.11 instead produced varying losses such as 139264 and 102403.609375 and incorrect gradients. The separate PyTorch 2.15.0+xpu environment passed all five repetitions. This establishes a working stack change, not which individual runtime component caused the failure.

Stopped the hung worker after cooperative cancellation and TERM did not terminate it. Canceled the queued CPU fallback before it started. Updated the existing Studio environment with the official PyTorch XPU index:

```bash
cd /home/intel/Documents/physical-ai-studio/application/backend
/home/intel/.local/bin/uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/xpu \
  'torch==2.15.0+xpu' 'torchvision==0.29.2+xpu'
```

This also updated the bundled Intel SYCL/runtime dependencies. The package list before the change is saved on the laptop at `/home/intel/junction-cell/evidence/studio-packages-before-gpu-fix.txt`. No system GPU driver, kernel, calibration, dataset, or robot limits were changed during this repair.

Run `cell/check_gpu_math.py` with Studio's `.venv/bin/python` to reproduce the arithmetic check. The repaired Studio environment passed. ACT training job `ffb3171d-2149-408c-aacc-5e156cf88728` uses xpu:0, one epoch, batch 8, FP32, two cameras, all seven episodes, and OpenVINO export. Its worker opened the xe render device and advanced with finite losses after the change.

## Package constraints

This is a station-specific environment override. Upstream Studio's XPU extra still pins PyTorch 2.11 and torchvision 0.26; running `uv sync --extra xpu` would undo the repair. Start Studio with `.venv/bin/physicalai-studio serve` as in the current recovery setup. Do not blindly resync this environment.

`uv pip check` reports LeRobot 0.6.0's declared torch/torchvision upper bounds and its pre-existing setuptools upper bound. The ACT run uses PhysicalAI's policy implementation and LeRobot's dataset reader. Validate this workflow end to end; successful ACT training does not establish compatibility for every other Studio policy. TorchCodec 0.11.1+cpu still imports successfully and dataset decoding is exercised by the training run. Upstream lockfiles have not been rewritten or package metadata falsified.

## Completed and verified

The replacement GPU job completed at 13:39:26 Helsinki in 224 seconds, including export. Checkpoint global step is 545 (one full epoch); all checkpoint tensors are finite. Studio model ID is `fa42ee76-bcfb-43ba-ba21-124e678ee785`, named **ACT inspection smoke - GPU repaired - 1 epoch**.

Laptop artifacts:

```text
/home/intel/.local/share/physicalai/models/fa42ee76-bcfb-43ba-ba21-124e678ee785/model.ckpt
/home/intel/.local/share/physicalai/models/fa42ee76-bcfb-43ba-ba21-124e678ee785/exports/openvino/act.xml
/home/intel/.local/share/physicalai/models/fa42ee76-bcfb-43ba-ba21-124e678ee785/exports/openvino/act.bin
/home/intel/.local/share/physicalai/models/fa42ee76-bcfb-43ba-ba21-124e678ee785/exports/openvino/manifest.json
```

`cell/verify_act_export.py` loaded the exported package through PhysicalAI's `InferenceModel`, including manifest preprocessing and camera ordering. It ran the first three saved observations (both real camera frames and measured joint state) on OpenVINO CPU and GPU. Every result was a finite 100×6 action chunk. CPU inference was 81–85 ms; GPU 8.5–10.3 ms after compilation. Maximum absolute CPU/GPU differences were 0.066–0.129 action units; this is recorded, not asserted to establish physical equivalence. No robot command was issued.

Evidence: `evidence/act-training-completed.json`, `evidence/act-openvino-verification.json`, `evidence/gpu-math-check.json`. The verification scripts and evidence are also under `/home/intel/junction-cell/` on the laptop. This is a one-epoch smoke model with no validation split, not a demonstrated autonomous skill. Supervised physical evaluation and further demonstrations/training remain necessary to establish task performance.

## Supervised robot smoke test

With operator readiness confirmed, loaded the trained export on GPU in Studio with both live camera feeds. Ran a bounded model-control test at 10 Hz with a 3-second gradual start, 5-second maximum and a 15-normalized-unit displacement guard. Encoder movement confirmed the model-to-arm path. Shoulder lift displacement reached 15.26 units, so the test stopped early. Studio acknowledged control=None, then the session closed; a live query confirmed zero sessions. This was the additional test guard, not a calibrated motor-limit violation. No full pick/place success was established. Evidence: `evidence/act-robot-test.json`.

## Travel-guard diagnosis and longer trial — 14 November 2026

The initial 15-unit displacement guard was a short-test restriction, not a calibrated motor limit. All seven demonstrations require much larger shoulder-lift excursions; initial predicted targets lay inside the demonstrated range. Replaced that test guard with recorded state/action min/max plus 5 normalized units, intersected with the existing normalized joint limits (five rotary joints −100..100, gripper 0..100). This is a monitoring guard, not a collision detector or hard real-time limiter. Preserved motor calibration/limits. Trials run at the dataset rate30Hz,3s gradual start,30s duration, with stale-camera/joint-feedback checks.

The first longer trial stopped at27.44s for gripper target−0.4796. Added per-joint bounds to the OpenVINO graph output. Original checkpoint/export retained; original XML/BIN and manifest are under the export's `original/` directory. In this runtime version `_get_model_path` selects `act.xml` directly, ignoring a different artifact filename in the manifest. An initial retry consequently still loaded the unbounded artifact and stopped at1.84s. The bounded XML/BIN were then installed at canonical `act.xml`/`act.bin`, with backup retained. Synthetic boundary test and CPU/GPU recorded-input inference passed. `cell/bound_act_export.py` records the corrected deployment procedure; do not rerun on an already modified export.

Final trial completed30.04s with no guard violation. Control stopped and subsequent live query returned no sessions. It began from the stopped mid-task posture, not a reset demonstration start. Camera review and final joint state show the arm near the pickup area; no successful pick-and-place was established. A successful control-path test does not establish policy quality. One-epoch smoke model needs further evaluation/training and controlled initial setup before claiming task completion. Evidence: `act-robot-full-test.json`, `act-robot-bounded-test.json`, `act-robot-final-test.json`, `act-bounded-export-verification.json`.
