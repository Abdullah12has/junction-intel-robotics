# Blue-plate VLA: dataset findings and experiment plan

The user subsequently requested parallel model/marker/augmentation/simulation comparisons. See [BLUE-PLATE-PARALLEL-EXPERIMENTS.md](BLUE-PLATE-PARALLEL-EXPERIMENTS.md) for the superseding 32-configuration campaign and fixed overhead+wrist camera choice. The dataset inspection below remains applicable.

15 November 2026. Decision: prepare a trustworthy single-pick dataset, establish a raw-image baseline, then compare explicit target-colour language and overhead target markers. Do not assume target confusion is the current failure mechanism. This is an inspection and plan, not a trained-policy result.

## What was actually inspected

Source: Intel Studio `554ce492-3fb1-496b-862a-347551ed38c0`, named Best-left. The immutable local copy and source SHA-256 manifest are in `../../station-backups/2026-11-15-best-left/`. All 11 files previously matched the Intel source. No original recordings were changed.

This review checked all action/state rows and episode indices; decoded all six video files sequentially (24,444 frames per camera); visually reviewed overhead frames at start, 33%, 66% and end for all 64 episodes; checked five samples from all three views for nine episodes (0, 7, 16, 24, 32, 40, 48, 56, 63); and checked 12 samples from overhead and wrist for episodes 41, 52 and 62. Episode numbers here are zero-based. This is a sampled visual audit, not frame-by-frame certification of every grasp or a measurement of acquisition synchronization.

Evidence and reproducible inspection script: `../../station-backups/2026-11-15-best-left/inspection/`. Start with `overhead-00.jpg` through `overhead-07.jpg`, `multiview-*.jpg`, `numeric-audit.json`, and `leading-idle-audit.json`.

| Finding | Evidence | Consequence |
|---|---|---|
| 64 short episodes, 24,444 rows at nominal 30 Hz, 814.8 seconds total | Metadata, parquet, full sequential video decoding | About 13.6 minutes of demonstrations; median episode 12.13 s, range 7.53–24.4 s. |
| Stored instruction names the left green plate | `meta/tasks.parquet`; user explicitly corrected it | Correct all derived task references to blue plate. Use colour in policy instructions, avoiding camera-left/operator-right ambiguity. |
| Demonstrations are predominantly one-block transfers among several candidates | All overhead contact sheets | Train a single-pick skill first. These episodes do not alone establish a reliable autonomous all-blocks stop condition. |
| Consecutive episodes reuse layouts, with blocks accumulating in the plate | For example episodes 2–5 and 57–63 | 64 episodes are not 64 independent layouts. Keep each reset/layout family together in a split. |
| Overhead captures task context; blocks occupy a small image region | Overhead sheets | Test a fixed workspace crop retaining all source blocks, destination and gripper motion. Do not crop to only the object. |
| Wrist shows fingers and grasped block clearly in sampled approaches/transports | Multiview sheets | It is a valuable candidate policy input. Preserve its finger geometry and close-up detail. |
| Fixed side view sometimes misses the active grasp and includes human torso/hands | Multiview episodes 7, 24, 32, 48, 56, 63 | Compare overhead+wrist against all three views. Do not conclude the side view is useless: it also supplies lift/placement cues. |
| About 119.7 seconds precede the first >1-unit measured joint displacement; median 1.8 s/episode | Explicit heuristic in leading-idle-audit.json | Candidate lead-in reduction, not 119.7 seconds automatically safe to delete. Confirm visually and preserve anticipation context. |
| All actions/states finite; all episode frame indices contiguous | Numeric audit | Basic structure is usable. |
| Action reaches shoulder-lift -100 in 92 rows and elbow +100 in 135 rows | Numeric bounds audit | Review affected phases and action semantics. Do not expand limits or silently clip/relabel to hide them. These values alone do not prove a calibration defect. |
| State/action differences are nonzero | Median elbow absolute difference 1.83 normalized units; shoulder-lift p95 10.07 | Audit timing and tracking; differences can reflect ordinary commanded-versus-measured behavior. They are not proof of sensor lag or wrong labels. |

An initial inconsistency in thumbnail interpretation prompted sequential decoding. A later nine-position random-seek comparison on overhead file-002 had zero pixel differences against sequential decoding (`decoder-seek-check.json`). No decoder defect or failed empty transport was established. Training-reader/frame correspondence remains a check before training, not a diagnosed fault.

## Recommended processing

### 1. Create a versioned annotation manifest before exporting

One row per original episode: source ID, layout/reset group, target identity/colour, intended destination, observed outcome, confidence, approach/close/lift/release/retreat frame boundaries, camera defects, recovery tags, and approved trim bounds. Preserve uncertain labels explicitly. Use the wrist to confirm what was held and overhead/side to verify destination; do not infer grasp success from closed fingers alone.

Correct the general instruction to `Pick up a cube and put it on the blue plate.` This agrees with the user's right-blue-plate goal without introducing conflicting image coordinates. For target-conditioned variants use the verified picked colour, e.g. `Pick up the white cube and put it on the blue plate.` Use stable names for the orange/brown and red/pink-looking blocks, whose appearance varies between views. Do not automatically label them solely by a brittle colour threshold.

Deliverables: annotation CSV/JSON, camera mapping manifest, reviewable excluded/uncertain list, grouped split manifest. Metadata corrections must reach task tables, episode task references and any exported training text—not only a README.

### 2. Validate timing and decoding through the intended trainer

Validate video offsets at episode and file boundaries for all cameras, compare a sample of actual training tensors with sequentially decoded source frames, verify frame freshness during movement, and preserve RGB channel order and aspect ratio. The table stores regularly spaced episode timestamps, not camera hardware acquisition timestamps, so it cannot certify frame-level synchronization. The local vendor recorder reads latest camera/state/action store values; that source snapshot is not proof of the exact code that recorded this dataset.

Use visible motion events and available capture logs for any timing investigation. Do not shift videos or command targets merely because a shift reduces loss. Record joint names, command-versus-measurement semantics, units, calibration provenance and gripper convention in the model package.

### 3. Trim conservatively and preserve complete skills

Mark obvious recording lead-in and post-retreat dead time. For reviewed leads, retain roughly 0.5 s of context before meaningful motion as a starting convention, adjusting when approach or finger motion begins earlier. Keep closure, contact settling, lift, release and retreat. Do not drop every low-motion frame or compress long waits into synthetic adjacent samples. Internal edits require explicit new segment boundaries and updated video offsets.

Do not smooth joint/gripper targets by default, nor replace action commands with measured positions. Both change the learning/control problem. Preserve successful corrective approaches as labelled examples; quarantine verified mistakes from the first success-imitation baseline without deleting them. Outcome review is still needed before declaring all 64 suitable.

Research supports caution: [Exploiting Policy Idling](https://arxiv.org/abs/2508.15669) finds that small-action filtering/control-rate changes can damage other aspects of performance. [DemInf](https://arxiv.org/abs/2502.08623) studies quality in terms of both diversity and predictability; for this small collection, manual outcome review is the first practical method, with automated ranking a later option.

### 4. Keep visual preprocessing simple

Keep original three streams for provenance. First train all-three-view raw baseline; separately test overhead+wrist. Use a fixed, versioned camera-to-policy-key map. The old frozen checkpoint's fixed-close input cannot simply be replaced with wrist images.

If testing a crop, choose one constant overhead workspace ROI from training examples that retains both plates, every valid target and all relevant gripper positions across the demonstrations. The lower overhead image contains much non-task equipment; cropping can allocate more pixels to objects. Check held-out coverage without tuning on outcomes. Deploy exactly the same transform. Keep the wrist uncropped initially; do not remove fingertips. No horizontal flips, strong hue jitter, synthetic object relocation or background replacement. Mild brightness/contrast variation can be a later isolated ablation.

### 5. Split by scene family; normalize on training only

Reconstruct actual reset groups from end/start frames, including near-duplicate layouts across resets where appropriate. Aim approximately 70/15/15 train/validation/test by whole groups, but use counts determined by the groups and colour coverage rather than forcing a frame quota. Freeze the test set before selecting crops/checkpoints. With few groups, use grouped cross-validation for model selection and collect fresh physical test layouts.

Fit action/state normalization on the training partition, then package those statistics with each checkpoint. Preserve normalized absolute joint-command representation for the first experiments. Do not invent a relative-action flag: the inspected upstream SmolVLA ALOHA delta-action option is unimplemented. A genuine relative-action experiment needs validated conversion and inversion in both training and control.

## How to test the X idea properly

Explicit target identity is promising here, but the data does not establish harmful decision switching. A generative action policy can represent multiple valid choices; [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) demonstrates multimodal behavior. The real question is whether explicit targeting improves successful execution and commitment.

Test three conditioning variants on the same retained demonstrations and grouped split:

| Variant | Input | Deployment dependency |
|---|---|---|
| A: generic | Raw views; pick a cube to blue plate | Policy selects a valid unplaced block. |
| B: colour | Same views; pick the verified target colour to blue plate | Operator or selector picks a colour once per attempt. |
| C: visual marker | Thin consistent marker in overhead; pick the marked cube to blue plate; other views raw | Operator click initially, then validated causal target tracking/selection. |

Start B before C because the current blocks have distinguishable appearances and text annotation is cheaper than tracking. Test C if B fails target adherence, or if future tasks include visually identical blocks. [CrayonRobo](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Object-Centric_Prompt-Driven_Vision-Language-Action_Model_for_Robotic_Manipulation_CVPR_2025_paper.pdf) and [RT-Sketch](https://rt-sketch.github.io/) support visual goal conditioning, but use different training/architectural setups. They do not prove an X beats text on SmolVLA with 64 demonstrations.

For C, annotate the demonstrated target before its approach. Retrospective inspection may identify the desired object; it must not supply future trajectory, hidden object position or true future phase as though known online. Render the same thin outline/cross in training and deployment, inspect it after model resizing, and keep identity fixed through the entire attempt. Do not move the marker to the nearest block each frame. Establish a causal occlusion policy and test tracking loss explicitly; do not use backward-smoothed perfect tracks only at test time. A target identity selected by a human is a legitimate goal input, not autonomous target selection.

Prove instruction use with same-layout counterfactual trials: select A, reset, select B. For colours change the text; for markers change the marked object. Test shuffled object positions so the policy cannot succeed solely by memorizing a location. Compare pick-any success across all methods and requested-target adherence for B/C separately; picking a different valid block is not a failure of the generic baseline's instruction.

## Training and evaluation sequence

1. **Data gate:** annotation review, correct blue-plate labels, grouped splits, decoder/timing checks and matched camera mapping. No hardware changes needed.
2. **Baselines:** fine-tune pretrained SmolVLA on the clean real dataset; train ACT separately as a lightweight non-language reference. Start from their documented recipes and pin software/checkpoint revisions. Do not mix simulation into the first real-data baseline.
3. **Prompt comparison:** B versus A with the same images, training budget and test layouts. Add C as the next run. Save checkpoints and compare actual closed-loop performance, not training loss alone.
4. **Camera/crop comparison:** test overhead+wrist versus all three on the strongest conditioning recipe; then test a fixed workspace crop separately. Avoid changing camera count and cropping in the same experiment.
5. **Runtime comparison:** execute 10, 25 and 50 actions per observation with unchanged weights and bounds. Record inference/image age and holds. Consider [RTC](https://huggingface.co/docs/lerobot/main/rtc) if synchronous pauses dominate; shorter execution alone is not RTC.
6. **Corrective collection:** use actual failure categories to collect missed-grasp recovery, displaced targets, harder spacing, plate clutter and new start configurations. [LeRobot HIL collection](https://huggingface.co/docs/lerobot/main/hil_data_collection) documents this feedback loop, subject to installed-version compatibility.

Screen promising variants on 20–30 paired physical single-pick trials with randomized model order, identical initial layouts and explicit operator readiness. Repeat promising training conditions with multiple seeds when practical. Report successes/trials and uncertainty, target switches before contact, verified lifts, placements, drops, interventions, time and limit clipping. A 30-trial result has wide uncertainty and is screening evidence, not a reliability certificate. Then test complete multi-block clearing, with a separate validated completion/stop decision. Target selection should exclude blocks already on the plate and should not silently change during an attempt.

Use an engineering gate: retain a more complicated preprocessing method only if it improves the relevant failure rate consistently across held-out layout groups without materially worsening grasp/placement, latency or interventions. Exact deployment success requirements need to be set before final acceptance; no paper supplies them for this station.

## Defer until evidence warrants them

- **Segmentation-only or target-only crops:** lose gripper/obstacle/destination context. If needed later, retain a global view and explicit crop geometry.
- **Historical visual traces:** [TraceVLA](https://tracevla.github.io/) encodes past motion and uses a different adaptation scale. Consider only if phase/motion ambiguity persists.
- **Future trajectory sketches:** [RT-Trajectory](https://rt-trajectory.github.io/) needs an inference-time plan; perfect demonstrated future paths are unavailable at deployment. A 2D X also says nothing about approach height or grasp orientation.
- **Depth/point clouds:** these recordings contain RGB, not stored depth; such a model would require new input data and architecture support.
- **Larger VLAs, synthetic action augmentation and RL:** not the first response to wrong labels or unverified input correspondence. Compare a larger pretrained model only after a stable dataset/control baseline.
- **More identical repetitions:** target new layout coverage and observed failures. [Data Scaling Laws in Imitation Learning](https://arxiv.org/abs/2410.18647) provides evidence for environment/object diversity, but its results on other tasks are not a numerical scaling guarantee for this station.

The research inventory and model-specific caveats are in [VLA-PREPROCESSING-RESEARCH.md](VLA-PREPROCESSING-RESEARCH.md). No training, dataset relabelling/export, automatic target marking or physical rollouts were launched in this task.
