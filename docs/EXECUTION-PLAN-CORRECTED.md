# Corrected execution plan: Intel Physical AI challenge

> Corrected review copy — 14 November 2026 (Helsinki). The original `EXECUTION-PLAN.md` is preserved unchanged. Read `EXECUTION-PLAN-CORRECTIONS.md` for the review findings. This is a revised plan, not a claim that the draft scripts have been fixed or validated. All station/model/device performance requires fresh verification.

Written Saturday 14 November 2026, 04:40 Helsinki. Deadline: **Sunday 15 November 10:00**; target FINAL at 09:00. This corrected copy revises the build order in [Intel-winning-plan.md](Intel-winning-plan.md) and [Intel-speed-and-reliability-research.md](Intel-speed-and-reliability-research.md) where they differ. Rules, logistics and the submission checklist in the winning plan still apply.

**Honest status.** The integration scripts remain untested drafts. Hardware setup and standalone camera capture have been verified previously, but no complete autonomous cycle is established by this review. The laptop was unreachable over SSH at 02:32 and 03:51. The challenge page and Discord were not rechecked (sign-in needed), so a rubric clarification may exist that this plan has not seen. No plan can guarantee a win; this one removes the known ways to lose and puts a measured gate before each expensive step. Claims marked *(code)* were read in the pinned Studio source at `17714488`; *(reported)* comes from cited third-party results; *(estimate)* is a guess to be replaced by a measurement.

## 1. What changed and why

| Earlier plan | Now | Reason |
|---|---|---|
| One policy, inspect in the gripper or "somewhere" | **Two stages around a fixed inspection spot** | Kickoff slide 9 shows exactly this: a policy moves the brick in front of a fixed camera, Anomalib inspects, a language policy sorts it. A static brick on a taped mark gives a fixed crop, no motion blur and no gripper in the image. |
| SmolVLA default, ACT fallback | **Train both from one dataset, choose by test** | Modal jobs have no resume and no intermediate checkpoints *(code)*, so a single bet is too risky. |
| "Run via OpenVINO on iGPU" | **SmolVLA starts on CPU; GPU only after a NaN check** | The pinned SmolVLA export lacks the f16 NaN fix *(code)*; iGPU and NPU default to f16. |
| Orchestrator unspecified | **A small script driving Studio's runtime WebSocket** | Reuses the tested "Run model" path; each `set_control` resets policy state *(code)*. |
| Reinspection as the differentiator | **Presence check and bounded retry** | A second look at a static brick adds nothing. Checking that the brick actually arrived and actually left catches real failures and matches the kickoff rubric's "recovery" line. |
| 30 FPS / camera unknowns | **Inspection camera owned by our script, not Studio** | Keeps a high-resolution locked-exposure image out of the policy's inputs and off the WebSocket. |

## 2. Target system

```
HOME ─► Stage 1 policy: "move the brick to the inspection spot" ─► arm back at HOME
     ─► presence check (brick on the mark?)  ── no ─► retry once ─► FAULT
     ─► Anomalib (PatchCore, OpenVINO) on fixed 256×256 crop ─► good / defective
     ─► Stage 2 policy: "put the brick in the green bowl" | "put the brick in the red bowl"
     ─► arm back at HOME ─► spot empty? ── no ─► retry once ─► FAULT ─► log row
```

- **Every demonstration ends at HOME.** Studio has no "move to pose" command *(code)*, so validated completion requires fresh named-joint samples, observed departure from HOME, then a stable return to HOME, plus the task-specific visual postcondition. A HOME sample alone is not completion.
- **One recorded dataset, three task strings**, copied verbatim into the script. SmolVLA needs the exact training strings *(reported)* and fails on an empty task *(code)*.
- **Primary:** one SmolVLA handles all three strings (no model swap, language picks the bowl).
- **Fallback, no new recording:** three ACT policies (one per string), swapped with `load_model` between stages. Costs a few seconds per swap *(estimate)* and loses the language story, but ACT is far faster on Intel hardware (86 ms vs 1,328 ms on an older iGPU *(reported)*).
- **Mixed, if tests favour it:** ACT for stage 1, SmolVLA for stage 2. This is literally slide 9.

Device plan to test, not assume:

| Model | First choice | Notes |
|---|---|---|
| SmolVLA | CPU through Studio; iGPU if the NaN check passes | iGPU defaults to f16 and the pinned export lacks the NaN fix (commit `083f7d78`, unmerged). Outside Studio, `InferenceModel(..., device="GPU", INFERENCE_PRECISION_HINT="f32")` is the workaround to try. Do not attempt the NPU: the graph is one large FP32 model with ten unrolled denoising steps. |
| ACT | iGPU, plus an NPU row in the table | Small and static; no published NPU number, so ours is new evidence either way. |
| PatchCore | NPU, static shape | Runs once per brick. iGPU is the fallback. |
| Control loop, cameras | CPU | |

`device="auto"` in Studio and the `physicalai` runtime means **CPU**, not automatic selection *(code)*. Always pass the device by name and log `EXECUTION_DEVICES`.

## 3. Ask the mentors first (10 minutes, do not block on answers)

1. Which rubric is final? (Three differ. Kickoff slides: end-to-end 25, Anomalib 20, VLA/Studio 20, OpenVINO 20, reliability 10, innovation 5.)
2. Which brick faces carry defects? If the underside can be marked, the inspection spot cannot see it.
3. Is placing the brick at an inspection spot acceptable, as on slide 9? Is ACT acceptable for one stage?
4. Judging format: live run at our table, how many bricks, who places them, are resets allowed?
5. Is SmolVLA on the iGPU known to work on this branch (NaN in f16)? Is there a known-good NPU model?

If answer 2 is "any face": add a second inspection camera at a low angle, and say plainly in the pitch that the underside is not covered.

## 4. Roles

| Role | Owns |
|---|---|
| **R** Robot | Rig, demonstrations, policy tests on the arm |
| **V** Vision | Inspection camera, Anomalib data, model, thresholds, device benchmark |
| **I** Integration | Modal, training jobs, `cell/run_cell.py`, the running baseline |
| **E** Evidence | Trial log, power/latency table, video, deck, submission |

With fewer people: R+I and V+E. One person owns the submission regardless.

## 5. Schedule and gates

Times assume the team is at the station by 09:00 Saturday. If later, shift everything but keep the three hard gates in bold.

| When | Phase | Gate to pass |
|---|---|---|
| 09:00–09:30 | 0 Station up, mentor questions | Teleop works, cameras stable for 5 min |
| 09:30–11:00 | 1 Rig + pipeline smoke test | A 5-episode model trained on Modal moves the arm from OpenVINO |
| 11:00–14:00 | 2 Record data (R), build detector (V) | 120 clean episodes; detector passes held-out bricks |
| **14:00** | 3 Launch all training jobs | **Hard gate: jobs submitted** |
| 14:00–17:00 | 4 Build while training | Script runs a full cycle with the smoke model |
| 17:00–19:00 | 5 Test trained models, choose | 8/10 per skill for the chosen model |
| **21:00** | 6 Loop closed | **Hard gate: both bowls reached autonomously, detector deciding** |
| 21:00–00:00 | 7 One improvement round | Retrain launched by 22:00 or skipped |
| 00:00–03:00 | 8 Evaluation and evidence | 30 logged trials, device table filled |
| 03:00–08:00 | 9 Video, deck, README; sleep in shifts off-site | — |
| **Sun 08:00** | Freeze | **No model or code changes** |
| Sun 09:00 | Submission set to FINAL | Links open in a signed-out browser |

Saturday 12:00 is the Intel technical session: send one person (E), keep recording.

## 6. Steps

### Phase 0: station up (all, 30 min)

1. Find the laptop's current IP (it changed once already) and reconnect per the [runbook](ROBOT-STATION-RUNBOOK.md). Check live session state before touching anything.
2. Confirm both arms respond and teleop tracks. Confirm you can stop.
3. Run the two RealSense cameras at 640×480/30 **locally on the laptop screen**, with no remote browser attached, for 5 minutes. If the stalls from the handoff recur locally, stop and get a mentor; do not record on unstable cameras. Work on the laptop directly from here on; remote preview is optional.
4. E posts the mentor questions from section 3.
5. I records versions into `evidence/versions.txt`: Studio commit, `openvino` version, `ov.Core().available_devices`, kernel.

### Phase 1: rig and pipeline smoke test (90 min)

**R: rig (30 min).**
1. Clamp the arm base. Tape outlines for: both bowls, the inspection mark (a brick-sized rectangle), a pick zone about 15×15 cm with six numbered positions. A dense small zone beats a wide one *(reported: 75 demos over 10 cm worked, 50 over 30 cm failed)*.
2. Use two clearly different bowls and put them on opposite sides. Green = good, red = defective.
3. Plain matte mat under everything. Friction tape on the gripper tips.
4. Policy cameras: overhead + wrist, 640×480/30, clamped hard. Mark mount positions with tape; a nudged camera breaks the policy *(reported)*. Name them now and never rename: camera names must match between recording and inference *(code)*. One must be mapped to SmolVLA's required "Overview" slot *(code)*.
5. Define HOME: a relaxed pose clear of all cameras' view of the mark. Photograph it.

**V: inspection camera (30 min).**
1. Mount the 2K webcam (or the Sonix camera) 15–20 cm from the inspection mark, slightly above, looking down at about 45° so it sees the top and one side. Add a matte backdrop that is not a brick colour.
2. Do **not** add this camera to the Studio environment. Our script opens it directly.
3. Resolve the inspection camera by the discovered identity, check device ownership, explicitly set and read back capture resolution/FPS, and validate the ROI lies wholly inside the actual frame. The verified Sonix profile was 640×480/30; the draft ROI (420,180,360,360) exceeds that frame. Measure a valid ROI at the station. Lock exposure and white balance (`v4l2-ctl -d /dev/videoN -l` to list controls, then set manual). Diffuse the light; glare on glossy plastic looks like a defect.
4. Set `ROI` in `cell/inspection.py` so the brick fills most of the crop with about 15% margin. Require a valid empty reference of matching shape. Use timestamped fresh frames acquired after arrival/settling with a bounded wait; dropping five frames is not a freshness guarantee. Keep capture/inference work off the runtime receive loop.

**I: Modal and smoke test (60 min, start immediately).**
1. Backend Modal auth as the `intel` user: `uvx --python 3.13 --from 'modal[api-proxy-support]>=1.6.3,<2' modal setup`, then Studio → Settings → Training Targets → Modal → Deploy and connect. First image build takes several minutes.
2. Get a **second teammate's Modal target** connected too. Only one job runs per target per policy type *(code)*; parallel jobs need a second account.
3. Set the Hugging Face read token in Studio settings.
4. R records 5 quick stage-1 episodes into a throwaway dataset. Train ACT, 1 epoch, A10G. Confirm the job returns an OpenVINO export. Export failures are only logged, not raised *(code)*; if the export is missing run `physicalai-studio models export <model_id> --backend openvino` on the laptop to see the real error.
5. "Run model" on the arm from the Models page with device GPU. It will move badly; you are testing the plumbing. Set the control rate to the dataset fps (30) by hand *(code)*.
6. Practise the dataset split you will need for ACT: download the dataset, `physicalai-studio datasets import-dir`, delete the other tasks' episodes. There is no merge *(code)*, so always record into one dataset and split copies.

**Gate 1:** arm moved from an OpenVINO model trained on Modal. If this is not true by 11:30, get a mentor before recording 120 episodes.

### Phase 2a: record demonstrations (R + one helper, 3 h)

One dataset, 30 fps, about 120 episodes:

| Task string (exact) | Episodes | Start → end |
|---|---:|---|
| `move the brick to the inspection spot` | 60 | HOME → pick from zone → place on mark → HOME |
| `put the brick in the green bowl` | 30 | HOME → pick from mark → green bowl → HOME |
| `put the brick in the red bowl` | 30 | HOME → pick from mark → red bowl → HOME |

Rules:
- Stage 1: six positions × ten episodes, yaw within ±45°, all brick colours, last 10–15 with one or two distractor bricks nearby.
- Stage 2: **choose the bowl by coin flip and alternate often**, using the same bricks for both. The instruction must be the only thing that differs, otherwise the policy learns to ignore it *(reported failure mode)*.
- Start moving at once; initial stillness teaches the policy to stall *(reported)*. One grasp style, always from the same side. Move briskly: the policy copies your speed.
- Each episode 8–12 s. Watch the camera feeds, not the arm.
- Bad take (fumble, hesitation, collision, drop): delete it right away. After `unload_dataset`, not during a session; deletes made while a session holds the dataset may be undone *(inferred from code)*.
- Episodes only land in the dataset on unload *(code)*. Unload and check the count every 30 episodes so a crash cannot cost the whole session.
- Keep a tally in `evidence/dataset.md`: recorded, deleted and why, per task. This is the "dataset quality" evidence.

### Phase 2b: build the detector (V, 3 h, parallel)

1. Environment (never inside Studio's): `uv venv anomalib-env && source anomalib-env/bin/activate && uv pip install "anomalib[openvino,xpu]"`. If the `xpu` extra fails on `pytorch-triton-xpu`, install torch from `https://download.pytorch.org/whl/xpu` first, then `anomalib[openvino]`. Check `torch.xpu.is_available()`.
2. Sort physical bricks into four trays and never mix them: **A** good/train, **B** good/validation, **C** good/held-out, plus defect bricks split **D1** validation and **D2** held-out. Never write IDs on bricks; pen marks are the defect.
3. Capture at the inspection mark with `python cell/inspection.py capture <subdir>`: 150–250 `train/good` (every colour, small position and yaw offsets, the lighting range you expect), 30–50 `val/good`, all D1 into `val/defect`, C and D2 into `test_holdout/`.
4. `python cell/train_export.py patchcore` (minutes). It exports FP16 OpenVINO and a static-shape copy for the NPU.
5. `python cell/inspection.py calibrate` writes `thresholds.json` from validation scores. `python cell/inspection.py holdout` reports held-out results. Run holdout once per model; do not tune on it.
6. Benchmark the same model on all three devices and save the output:
   `for D in CPU GPU NPU; do benchmark_app -m export/patchcore/weights/openvino/model_static.xml -d $D -hint latency -niter 300; done`
7. If NPU is slow or fails, try `coreset_sampling_ratio` lower, then `python cell/train_export.py efficientad` (pure CNN, 15–30 min, pre-download ImageNette now). Timebox NPU work to 45 minutes; GPU is an acceptable answer if you report why.

**Gate 2b:** held-out false accepts = 0 and false rejects ≤ 1 in about 40 images, or you know which colour fails. If one colour is the problem, use a per-colour threshold before trying a new model.

### Phase 3: launch training (I, 14:00 sharp)

Studio defaults are 5 epochs, batch 8 *(code)*: too little. There are no intermediate checkpoints and cancelling discards everything *(code)*, so cover the range with separate jobs:

| Job | Dataset | GPU | Batch | Epochs | Target |
|---|---|---|---:|---:|---|
| SmolVLA-A | full | H100 | 64 | 12 | account 1 |
| SmolVLA-B | full | H100 | 64 | 24 | account 2 |
| ACT-stage1 | stage-1 copy | A10G | 8 | 30 | account 1 |
| ACT-green | green copy | A10G | 8 | 30 | account 2 |
| ACT-red | red copy | A10G | 8 | 30 | whichever frees first |

Epoch counts are estimates aimed at roughly 8k and 16k SmolVLA steps and about 80k ACT steps; adjust once you know the real frame count (`steps = frames / batch × epochs`). Expected 1–3 h each *(estimate)*. Watch Modal's dashboard for spend; a dropped connection does not stop billing *(code)*. Write job IDs, settings and cost into `evidence/training.md`.

### Phase 4: build while training (14:00–17:00)

- **I:** get [`cell/run_cell.py`](../cell/run_cell.py) running a full cycle with the smoke-test model. It is an untested draft: the WebSocket message shapes come from a code audit, so verify each command against the live backend in this order: connect, `load_environment`, read a state frame, `load_model`, `set_control` on, `set_control` off. Close Studio's own runtime page first; a second session gets `DeviceInUseError` *(code)*. Record HOME by joint name from fresh observations while physically parked. Validate the joint set, units, finite values and tolerances; do not rely on JSON dictionary order. Complete the controller acceptance gate below before any autonomous run.
- **V:** finish the device benchmark; wire `classify()` into the script; add the empty-spot reference image.
- **E:** start the power/latency sampler and the trial CSV (section 7). Draft the deck skeleton and video shot list.
- **R:** eat, rest. Then record 10 spare stage-2 episodes per bowl in case of a retrain.

### Phase 5: test and choose (17:00–19:00)

For each finished model, from the Models page or the script:

1. **Device check for SmolVLA:** run on CPU first. Before GPU motion, compare CPU/GPU outputs on identical recorded inputs and check for non-finite actions. NaNs warrant investigation of precision/export compatibility; poor motion alone does not diagnose the f16 issue. Check preprocessing, camera mapping, calibration, task strings and action rate. Keep the proven device until the cause is established. Record chunk latency for each.
2. **Per-skill test:** 10 trials per task string, brick positions chosen before you start. Count successes.
3. **Instruction-swap test (stage 2):** same brick, same start, alternate the two strings ten times. Fewer than 9/10 correct bowls means the policy is not reading language reliably.
4. Sync vs async execution: if SmolVLA latency exceeds about 0.8 s per chunk, use async with `request_threshold` 0.7–0.8 *(estimate)*; compare smoothness and success on 5 trials each.

Choose:

| Result | Ship |
|---|---|
| SmolVLA ≥ 8/10 on all three and passes swap | SmolVLA alone |
| SmolVLA passes stage 2, weak on stage 1 | ACT stage 1 + SmolVLA stage 2 |
| SmolVLA fails swap | ACT for all three; keep SmolVLA in the benchmark table and say why |
| Nothing ≥ 6/10 | Look at the failures; usually camera moved, wrong control rate or bad demos. Fix and retrain the cheapest job |

### Phase 6: close the loop (19:00–21:00)

1. Run 5 good and 5 defective bricks through `cell/run_cell.py` with nobody touching the decision.
2. Test faults offline first. For supervised physical tests, arrange a missing brick or blocked completion before enabling motion, with hands clear. Never reach into an operating arm to sabotage a trial. A timeout, stale telemetry or failed runtime faults immediately; only a completed skill with a failed visual postcondition may use a validated retry from its required starting state.
3. A stop key and a person at the power switch for every autonomous run. `set_control null` holds the pose; it is not an emergency stop. Nothing in the stack limits joint speed *(code)*.

**Gate 6 (21:00):** both bowls reached autonomously with the detector deciding. If not met, drop to the ACT fallback now and spend the improvement round on getting this gate, nothing else.

### Phase 7: one improvement round (21:00–00:00)

Pick **one**, by what the last hour showed:
- Policy failures cluster at certain positions → record 20–40 episodes there, retrain one job, launched by 22:00.
- Detector errors → recapture that colour or lighting, retrain (minutes).
- Everything works → optimisation evidence: INT8 or FP16 comparisons for the detector, ACT on NPU, async vs sync cycle time.

Do not migrate versions, add a language model, or start a second improvement.

### Phase 8: evaluation (00:00–03:00)

1. Freeze the configuration; note model IDs and thresholds in `evidence/config.md`.
2. Write the 30-trial list before starting: 15 good, 15 defective, across nominal, shifted/rotated and cluttered placements, in shuffled order.
3. Run all 30 through the validated script with the sampler running. Persist an attempt-start record before motion, then finalize it even on exceptions, cancellations, timeouts or disconnects. Preserve partial results and fault reasons; no silent dropped attempts or reruns.
4. Record one uninterrupted video of at least 6 consecutive cycles including both bowls.
5. Fill the tables in section 7.

### Phase 9: submission (03:00–09:00)

Follow the checklist and video outline in [Intel-winning-plan.md](Intel-winning-plan.md). Additions:
- 120-second video: 15 s problem, 50 s uninterrupted run with on-screen decision and score, 25 s architecture with the real device table, 20 s measured results, 10 s limits.
- README: how to run, pinned commit, model IDs, the tables, what does not work.
- Back up the dataset, exports and `evidence/` off the laptop before it is returned.
- Sunday 08:00 freeze, 09:00 FINAL, then everyone to the demo table for peer review.

## 7. Evidence to collect

| Rubric line | What we show |
|---|---|
| End-to-end / task success | `evidence/trials.csv`: 30 trials, success count, p50/p95 cycle time, correct sorts per minute |
| Anomalib | Held-out confusion matrix by physical brick; score histogram; heatmap in the video |
| VLA + Studio | Three task strings, instruction-swap result, Studio training jobs and settings |
| Dataset quality | Episode counts recorded/deleted per task, coverage grid, held-out positions |
| OpenVINO / devices | Table: model × device × precision → compile time, p50/p95 latency; `EXECUTION_DEVICES` printed by the script, never `AUTO` |
| Efficiency | Package power and temperature idle vs running, per configuration; energy per sort labelled as CPU-package only |
| Reliability / recovery | Retry counts and outcomes from the CSV; the two sabotage tests on video |

**Device and latency proof.** [`cell/ov_prove.py`](../cell/ov_prove.py) `MODEL.xml DEVICE [f16|f32]` prints the device list, compile time, `EXECUTION_DEVICES`, precision and p50/p95 as one JSON line. Run it twice per row (cold, then cached compile). Keep compile time and the first inference out of latency numbers.

**Power and temperature.** [`cell/power_sampler.py`](../cell/power_sampler.py) `OUT.csv` writes package watts, package temperature, NPU busy % and iGPU busy % twice a second; change the row label live with `echo npu_fp16 > /tmp/pwr_label`. Both scripts are syntax-checked only, not run on the laptop.

```bash
sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*:*/energy_uj   # until reboot
sudo turbostat --list | tr ',' '\n' | grep -iE 'watt|tmp|gfx'   # see which columns this chip really has
```

Traps:
- RAPL `package-0` covers CPU cores, iGPU and NPU together; it excludes screen, cameras and the servos. Say so next to every number.
- The NPU has no power counter. Report NPU **busy time**, not watts.
- `intel_gpu_top` does not support the `xe` driver this laptop uses. Use the sampler's iGPU busy column or `qmassa`.
- Plug in AC, fix the power profile (`powerprofilesctl get`) and record a 2-minute idle baseline first. Energy per inference = (mean W under load − idle W) ÷ inferences per second.

**The 90-minute table** (rows: ACT on CPU/GPU/NPU; SmolVLA on CPU and GPU; PatchCore on CPU/GPU/NPU; whole system per sorting cycle):

| Model | Device | Precision | Compile cold/warm s | p50/p95 ms | Package W idle→load | Max °C | Busy % | Accuracy or success, NaN check |
|---|---|---|---|---|---|---|---|---|

Cheap optimisations worth a row each, only after gate 6: `CACHE_DIR` (compile time), FP16 vs FP32 on iGPU, async `request_threshold` 0.5 vs 0.2 (fewer inferences per cycle; check success holds), INT8 weight compression of a **copy** of the SmolVLA export. PatchCore INT8 is risky: it shifts scores, so thresholds must be recalibrated and held-out rechecked.

**Robotics AI Suite** is named in one rubric. Honest low-cost use: its docs list Physical AI Studio and OpenVINO Physical AI as its physical-AI stack, so cite that, and run its `accel-check` device checks natively. Anything ROS 2-based is out of reach on Ubuntu 26.04 in the time left.

## 8. What would most likely lose it, and the guard

| Risk | Guard |
|---|---|
| Training returns nothing usable at 17:00 | Smoke test at 11:00; five jobs across two accounts; ACT fallback from the same data |
| Policy ignores the instruction | Coin-flip balanced stage-2 data; swap test; per-bowl ACT fallback |
| SmolVLA NaN or too slow on Intel devices | CPU first; measured before the loop is built; ACT is ~15× faster *(reported)* |
| Camera stalls return | Local-only operation; 5-minute stability check before recording |
| Detector fooled by glare, colour or a hidden face | Locked exposure, fixed crop, per-brick held-out split, mentor answer on faces |
| A camera or bowl gets bumped overnight | Tape outlines and photos of every mount; re-check before evaluation |
| Demo fails live in front of judges | Uninterrupted video already recorded; bounded retries; rehearsed manual reset |
| Time lost to NPU or optimisation rabbit holes | 45-minute timebox; only after gate 6 |

## 9. Explicitly not doing

Pi0.5 (40 GB VRAM, gated weights), RTC, SnapFlow distillation, policy quantisation, FunctionGemma or any chat interface, version upgrades, a third bowl category, Robotics AI Suite ROS 2 components.

## 10. Mandatory controller acceptance gate (added by review)

Do not run the current `cell/run_cell.py` on hardware just by filling its placeholders. Implement and test the following first; these requirements supersede permissive behavior in the draft.

1. **Preflight:** validate camera/model IDs, trained camera names, exact task strings, named HOME vector, finite limits/tolerances, dataset action rate, thresholds, empty-reference image and model artifacts. Reject missing configuration before loading devices. Inspect ownership and use `unload_incompatible=False`; do not take over another session.
2. **Supervised communication:** monitor the receive task and propagate runtime failure/disconnect immediately. Track observation receipt age with a monotonic clock. Missing, stale, malformed, non-finite or mismatched joint telemetry must fault rather than satisfy HOME. Handle late/unknown acknowledgements without killing the receive task; clean up timed-out futures.
3. **Completion:** require observed departure, stable fresh return and the visual postcondition. Propagate `skill()` timeout as a fault. Both existing call sites currently ignore its Boolean result. An empty inspection spot proves departure only, not arrival in the intended bowl; log the commanded destination separately from an observed destination or human-verified outcome.
4. **Retry:** never blindly retry from an unknown pose or after transport/control failure. Retry only a validated recoverable visual failure with a known valid start state, at most once, then fault. Faults must prevent the next stage.
5. **Responsive stop:** use nonblocking operator input and move blocking capture/inference off the asyncio receive/control loop. On every exit, attempt supported stop with a bounded acknowledgement wait, then release the session and camera resources. A lost connection makes stop confirmation unknown; do not claim the arm is holding based on an unacknowledged command. The local operator retains physical power access.
6. **Complete evidence:** initialize logging before motion and finalize each attempt in exception-safe cleanup. Record failure, elapsed time, sample freshness, model/config identity and stop acknowledgement status. Unknown physical outcomes stay unknown.
7. **Offline tests:** initial HOME without departure, stale HOME, missing/non-finite telemetry, skill timeout, receive-task failure, late ACK, camera read timeout, out-of-bounds ROI, missing reference, classifier failure, cancellation, retry exhaustion and CSV persistence on every failure. Only then run supervised single-command and single-skill hardware checks.

### Evidence and planning corrections

- The external ACT/SmolVLA speed figures are hypotheses for prioritization, not laptop benchmarks. Measure identical tasks/inputs with device and precision recorded. Do not infer a cause from bad motion alone.
- `ov_prove.py` feeds random synthetic inputs; it cannot prove policy correctness, language behavior or representative end-to-end latency. Use captured, correctly typed model inputs for policy validation. Reject unexpected dynamic dimensions rather than silently replacing every dynamic dimension with 1.
- RAPL domain coverage must be verified for this platform; do not assume package energy includes NPU. Label exactly which counter was measured and do not present it as whole-system/servo energy.
- Pin the independently installed Anomalib/OpenVINO environment after the smoke test. Third-party API/version compatibility and official rubric/deadline claims were not reverified in this review. Resolve them from organizers/current primary sources before relying on them.
- Keep the smoke-test gate ahead of mass recording. Schedule/model counts are estimates; prioritize one proven end-to-end route before optional parallel candidates or device optimization.
