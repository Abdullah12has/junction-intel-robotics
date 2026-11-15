# Blue-plate parallel training campaign

15 November 2026. Planning only: no instance provisioned, no training launched. This supersedes the sequential experiment ordering in BLUE-PLATE-VLA-PROCESSING-PLAN.md; its dataset audit remains applicable. “SmallVLA” means pretrained SmolVLA. All numerical augmentation and mixing settings below are proposed starting settings, not established optima.

## Campaign: 32 independent configurations

Full factorial: **2 policies × 2 target-conditioning methods × 4 augmentation recipes × 2 data sources = 32 runs per seed.** Launch seed 0 for screening; repeat finalists with seeds 1 and 2 before selecting a deployment model. Training can run concurrently once each input dataset passes its preparation gate.

| Factor | Values |
|---|---|
| Policy | SmolVLA initialized from pinned `lerobot/smolvla_base`; ACT with its documented vision initialization and newly trained action policy |
| Conditioning | N: original images, corrected blue-plate task; X: target X rendered in overhead image, wrist image unchanged |
| Augmentation | A0 none; A1 lighting; A2 lighting + small framing changes; A3 lighting + framing + mild sensor degradation |
| Data source | R: reviewed real training episodes only; M: 80% real / 20% accepted MuJoCo samples |

For **each** policy, run this entire table (16 runs):

| Images | Real A0 | Real A1 | Real A2 | Real A3 | Mixed A0 | Mixed A1 | Mixed A2 | Mixed A3 |
|---|---|---|---|---|---|---|---|---|
| N, normal | N-A0-R | N-A1-R | N-A2-R | N-A3-R | N-A0-M | N-A1-M | N-A2-M | N-A3-M |
| X, marked target | X-A0-R | X-A1-R | X-A2-R | X-A3-R | X-A0-M | X-A1-M | X-A2-M | X-A3-M |

Run identifier: `{smolvla|act}-{N|X}-{A0|A1|A2|A3}-{R|M}-seed{0|1|2}`. Independent outputs, optimizers, RNG state and logs; no weight sharing across conditions. The main matrix answers whether the augmentation package helps, whether X helps, and whether simulation helps within each matched condition. A3 is a combined recipe; any individual transform attribution requires a later ablation.

**ACT does not consume task language in its standard implementation.** Correct its metadata for provenance, but do not count that as an ACT conditioning change. Its X condition changes pixels. SmolVLA uses “Pick up a cube and put it on the blue plate” for N, and “Pick up the marked cube and put it on the blue plate” for X. [ACT inputs](https://huggingface.co/docs/lerobot/act), [SmolVLA inputs](https://huggingface.co/docs/lerobot/smolvla).

## Shared real-data preparation

Source is the immutable Best-left backup: 64 episodes / 24,444 frames, with overhead, fixed close and wrist views. Review outcomes before deciding retained counts. Correct the erroneous green-plate text in all derived task references. Annotate target identity, reset/layout family, success, phase boundaries, tracking confidence and approved trim bounds. Do not change originals.

Freeze a roughly 70/15/15 split by actual reset/layout family, adjusted to available groups and colour coverage. Keep all derivatives and simulator seeds reconstructed from a real demonstration in that demonstration's partition. Synthetic generation uses training layouts/ranges only, not held-out target positions. Real validation and test remain real, unaugmented, and separate from simulation validation. Compute normalization from real training frames only and reuse it across R/M within each architecture; audit synthetic outliers instead of silently clipping them.

For this campaign fix inputs to **overhead + actual wrist**, including matching camera identity and preprocessing at deployment. This resolves the earlier document's alternative all-three-view starting point. Keep full field of view in A0/A1. Adding fixed close or a permanent workspace crop is a follow-up, not another uncontrolled change in the core matrix. Validate actual trainer tensors, timestamp alignment, episode boundaries, state/action units and gripper convention before launch.

## X pipeline is part of the experiment

Annotate the object demonstrated, establish its identity before approach, and run a causal tracker with the same availability as deployment. Use a thin high-contrast X that remains visible after model resizing and does not cover most of the block. Store coordinates separately; render on demand. Manual retrospective identity annotation is legitimate goal labelling; future trajectories, hidden object positions and future phase labels are not valid online inputs.

At deployment an operator initially clicks the desired unplaced block. Maintain that identity throughout the attempt; do not switch to the nearest block. Define visibility-based marker suppression under occlusion and explicit tracking-loss handling, and apply the same rule to training. Test this pipeline on withheld recordings before training X runs. In simulation, use the same image-based tracking pipeline initialized from a visible target; simulator ground truth is for auditing, not privileged hidden marker placement.

Apply scene photometric augmentation first, transform image and target coordinates together for geometry, then draw the same digital X used at deployment. Do not independently jitter X positions in the main matrix. Tracker noise/dropout robustness is a separate later test. No X is added to wrist images in this campaign.

## Augmentation recipes

Generate fresh variants on the fly from the training partition. Do not materialize ten copies and claim ten times as many independent demonstrations. Keep action/state sequences, colour semantics, target identity and timing unchanged. Store transform seeds/configuration for reproducibility.

| Recipe | Proposed implementation |
|---|---|
| A0 | No stochastic image augmentation. Only required deterministic model resizing/padding/normalization. |
| A1 | With probability 0.5, brightness and contrast factors independently sampled from 0.9–1.1. No hue change or grayscale. |
| A2 | A1 plus overhead-only crop retaining 95–100% of width and height, resized with preserved aspect ratio/padding; probability 0.5. Reject crops losing any relevant source block, destination, fingers or motion corridor. If this cannot be certified from the training coverage audit, disable A2/A3 until it can. Wrist stays uncropped. |
| A3 | A2 plus, with probability 0.2, either weak Gaussian pixel noise (standard deviation up to 0.01 on [0,1]) or weak blur (sigma up to 0.5 pixels at source resolution). Audit that edges/fingers/target remain discernible. Bound magnitudes by observed station images before freezing the config. |

Sample consistently across any observation-history window. Camera photometric parameters may differ to mimic separate exposures; geometry remains fixed per camera within the window. Apply identical recipe definitions to real and synthetic samples. Do not use flips, large rotations, object relocation with unchanged actions, strong colour shifts, target erasing, action noise, time reversal or arbitrary frame deletion. More aggressive transforms are not automatically more useful data.

## MuJoCo: research conclusion and station gate

Simulation is worth a controlled branch, not assumed beneficial. [RoboCasa](https://robocasa.ai/) reports improved real-world performance from sim/real co-training in its setting. [MimicGen](https://mimicgen.github.io/) demonstrates scalable physically executed trajectory generation; its main simulated successes do not establish transfer to this station. See the companion MUJOCO-SYNTHETIC-DATA-RESEARCH.md for evidence and limitations.

The local project already has a simulator and old synthetic data. TRAINING-STATUS.md records 150 accepted scripted episodes, but the trained sim-only pilot produced one grasp and zero stable placements in six simulator rollouts. VERDA-TRAINING.md also records a previous old-data co-training run without an outcome evaluation there. Neither proves whether adding simulation helps the new 64-episode dataset. Low imitation loss alone was insufficient.

Do **not** blindly append that old dataset: it includes two destinations, previous views/action conventions, and known mismatch. NORMALIZED-INTERFACE.md documents a roughly 3.6-normalized-unit gripper gap at grasp and nominal mapping limitations. WRIST-CAMERA-HANDOFF.md documents a provisional wrist-mounted camera that was not wired into the old dataset collector/exporter. Historical documents contain superseded sections; verify current artifacts and code before reuse.

Before releasing M jobs:

1. Verify current normalized SO-101 command/state conversion, camera mapping and 30 Hz recording against the new real data. Use existing calibration without changing limits; quantify residual jaw/contact and pose error. If compatibility cannot be established, retain simulation as an isolated diagnostic rather than mix incompatible labels.
2. Fit overhead and moving wrist observations to the actual camera geometry; explicitly extend collection/export for the wrist stream. Never substitute the old fixed view named `wrist` for `wrist_mounted`.
3. Match the blue-plate task, relevant block geometry, table and reachable workspace. Generate new positions, spacing, yaw and plate clutter within valid training ranges. Randomize lighting/materials without changing semantic colours; vary friction, mass, actuation delay and controller response within measured/plausible bounds recorded in provenance.
4. Record true closed-loop physics trajectories with state, command, both RGB views, task, scene seed, target and outcome. Reject penetrations, nonphysical attachments, limit violations and unstable placements. An IK path is a proposal until contact simulation confirms grasp, lift, release and stable placement.
5. Start with 100 accepted episodes as a pipeline pilot, then target 500–1,000 diverse accepted episodes for the matrix, subject to verification. These counts are engineering proposals. Report attempts and rejection causes so success filtering does not hide a narrow easy-layout distribution.
6. Split by scene/generator family. Audit rendered observations and outcomes; confirm deterministic replay where supported. Generate/cache data before the main GPU campaign so rendering does not compete with trainers.

Use explicit domain sampling at **80% real / 20% sim**, not dataset concatenation or repeated files. Sample domain, then episode, then frame; log actual proportions. Keep sample policy identical across conditions. Preserve real and sim provenance and allow removing the synthetic branch without rebuilding the real dataset.

The core comparison holds optimizer steps and effective batch fixed within an architecture, so M sees fewer real samples. It measures performance at equal total training budget. If promising, repeat with matched real-sample exposure (M needs about 1.25× total samples at 80/20) and report the extra compute. Follow up on finalists with 90/10 and 50/50 ratios, and simulation pretraining followed by real-only finetuning; compare that last recipe to a real-only control with matching total compute. These are follow-ups, not hidden changes to the 32 runs. Sim-only checkpoints can diagnose simulator learning but cannot substitute for real evaluation.

## One Verda instance, concurrent job queue

Plan for **one 8-GPU instance**, initially one independent job per GPU: eight concurrent runs, about four waves for 32 runs. Verda advertises H100 configurations with 1/2/4/8 GPUs and 80 GB per H100. This is a planning option, not a reserved machine or proof of regional availability. [Verda H100 specifications](https://verda.com/h100).

An existing/recreated single RTX PRO 6000 96 GB machine is also possible, but it is not equivalent to eight GPUs. Benchmark two co-resident jobs before increasing concurrency; keep at least 15–20% observed peak-memory headroom, monitor utilization and step throughput, and prefer a queue if contention raises total completion time. Do not promise all 32 jobs simultaneously on one GPU. The prior Verda allocation is documented as released; no live instance has been checked for this plan.

Suggested eight-GPU schedule, chosen to make early paired results available:

| Wave | Jobs |
|---|---|
| 1 | Both models × N/X × A0/A1, real-only (8) |
| 2 | Both models × N/X × A2/A3, real-only (8) |
| 3 | Both models × N/X × A0/A1, mixed (8) |
| 4 | Both models × N/X × A2/A3, mixed (8) |

Scheduler may interleave real and mixed jobs as soon as simulation passes its gate. All jobs are independent; waves are resource allocation, not model dependencies. X runs wait only on marker QA. If simulation is not ready, do not keep paid GPUs idle waiting for it.

Use a pinned standalone LeRobot training environment with a supervisor/queue and explicit `CUDA_VISIBLE_DEVICES` binding per job. Do not use distributed training across all GPUs for each small experiment. Verify trainer compatibility with actual dataset format and mixed-domain sampler; these configurations are not claimed to be executable in the current Studio API.

Before the full launch, run 200–500 representative training steps per architecture, including X+A3+mixed tensors, then validate checkpoint reload and resume. Measure VRAM, wall seconds/step, RAM, CPU decode and storage throughput under planned concurrency. Pre-decode/cache immutable frames on local fast storage if video seeks bottleneck, preserving exact correspondence. Limit workers per job and share only read-only caches.

Starting budgets: SmolVLA 20k updates, effective batch 32; ACT 100k updates, effective batch 8, adjusted from training-only pilot evidence before freezing all conditions. Use the pinned architecture's documented optimizer settings and chunking, saving their exact resolved config. These are model-specific starting budgets, not an equal-compute architecture comparison. Keep each model's settings identical across its 16 conditions. Record both sample exposures and GPU time; do not interpret cross-architecture loss scales as comparable. [SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla), [ACT guide](https://huggingface.co/docs/lerobot/act).

Save periodic full resumable checkpoints (model, optimizer, scheduler, RNG and sampler progress), logs and provenance outside ephemeral process state. Test resumption before using spot instances. The existing Studio remote protocol has documented resume limitations; a standalone supervisor must not inherit an unverified claim of recovery support. Use unique run directories and periodically copy artifacts off the instance.

Estimate wall time from concurrent pilot rates and queued work, not the old unrelated single-run timing. Estimate cost as live instance hourly rate × campaign elapsed hours plus storage/transfer, then record the quoted SKU, region and rate before provision. No fixed cost or runtime is asserted here. Back up and verify artifacts before releasing the dedicated compute; follow current Verda deletion/billing instructions and do not touch unrelated resources.

## Evaluation and selection

Offline: fixed unaugmented real validation frames; trajectory/action errors per joint and phase, bounds, dropout sensitivity, inference latency and marker counterfactuals. These detect failures and shortlist checkpoints; they do not establish closed-loop success. Use validation for checkpoint selection, keep test sealed.

Physical screening: 10 standardized single-pick attempts per configuration, balanced over spacing, colour, reach and plate clutter, randomized model order and reset layouts. Across 32 configurations this is 320 attempts, so robot evaluation is a substantial sequential workload even when training is parallel. Avoid claiming a winner from 10 trials; promote approximately four configurations, including the best real-only control and any promising synthetic variant, to 30 paired trials on fresh layouts. Repeat those four training configurations with two additional seeds (8 more training jobs); test each seed on the same 30-layout confirmation protocol with reset/randomized ordering. This yields 90 trials per finalist across three seeds, while retaining seed-level variation in reporting. Test-set reuse for tuning requires a fresh final holdout.

Metrics: successful lift, stable blue-plate placement, requested-target adherence for X, pre-contact switches, drops, interventions, time, clipping and tracker failures. N is instructed to pick any valid cube; picking a different cube is not a selection error for N. Compare common pick-and-place success across all methods; report X target adherence separately. Run same-layout target-A/target-B trials for X to establish marker use. Record both policy performance with reliable selection and end-to-end performance including tracker failures.

Physical trials remain supervised through the established station controls with current session/serial ownership and calibration checks. Training does not authorize unattended robot control. Keep deployment execution horizon and camera processing fixed within model comparisons; report architecture-specific settings and observed latency. Full multi-block clearing and completion detection are subsequent tests.

Adopt synthetic data only if improvement persists on real held-out layouts and repeated seeds without worse interventions or placement reliability. If simulation looks excellent but real success declines, retain the real-only winner and investigate camera, jaw/contact, timing and action mismatch. If A3 hurts, keep A1/A2; maximal augmentation is not the goal.

## Deliverables before launch

- Reviewed real-data manifest, corrected export, grouped splits and original checksums.
- Validated target tracking/rendering and reproducible augmentation examples at model resolution.
- Accepted simulation dataset with camera/action compatibility report, generation code revision and provenance.
- Machine-readable 32-run manifest with resolved model/config hashes, seeds, resource bindings and output paths.
- Measured concurrency/cost estimate, tested queue/resume, and checkpoint backup procedure.
- Frozen physical evaluation sheet and result table comparing matched conditions.

The earlier audit is in [BLUE-PLATE-VLA-PROCESSING-PLAN.md](BLUE-PLATE-VLA-PROCESSING-PLAN.md). This document specifies the campaign; dataset exports, simulator validation, executable configs, provisioning and training remain implementation work.
