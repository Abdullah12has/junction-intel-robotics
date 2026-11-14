# Junction 2026 Intel challenge: winning plan

> **Follow [EXECUTION-PLAN.md](EXECUTION-PLAN.md) for what to do and when.** This document remains the reference for rules, logistics and the submission checklist.

Researched 13 November 2026, Helsinki time. Official rules and observed mentor messages are distinguished from recommendations. Discord reviewed from the channel beginning through the latest loaded messages at 22:28; this is a snapshot, not ongoing monitoring.

## Updated build decision — 14 November 2026

Read the [researched speed and reliability plan](Intel-speed-and-reliability-research.md) for model comparisons, primary sources, optimization gates, experiments and team execution order. It supersedes the earlier generic implementation recommendations below where they differ; the saved event rules remain historical source material.

**Recommended stack:** Studio-trained SmolVLA through the supported OpenVINO path, Anomalib inspection, and a deterministic state machine. Keep a mentor-approved ACT fallback if export/training cannot meet the deadline. Do not replace an already working policy just to follow this recommendation.

**Build order:** resolve local camera/session stalls → complete both autonomous sorting branches → improve data and inspection visibility → profile the cycle → optimize the measured bottleneck → add one bounded reinspection → collect honest evaluation and submit.

**Gemma decision:** optional operator-language interface only. FunctionGemma translates language to tools; it is not a trained SO-101 control policy. The good/bad decision should use the validated Anomalib result directly. There is no measured reason to insert a text-generation call into every sorting cycle.

**Speed candidates:** fixed-shape batch-one OpenVINO inference, compile/warm once, explicit CPU/GPU/NPU comparisons, fresh local frames, and the existing PhysicalAI asynchronous execution path if the installed event version supports it. RTC and INT8 are conditional experiments, not baseline requirements. Never change motor limits or the trained action timebase to chase speed.

**Current bottleneck evidence:** the station handoff recorded camera/session disconnects, stalls up to ~12 seconds and a 6 FPS preview workaround. Recheck live state before acting. Model latency improvements cannot compensate for stale observations or repeated failed picks.

**Winning evidence:** correct autonomous sorts per minute, success counts, defect false accepts/rejects, p50/p95 cycle time, actual execution devices and a baseline-versus-final comparison. Suggested 30-attempt evaluation is an internal target, not an official minimum. No performance gains have yet been measured by this research.

## Sources and saved material

- [Participant guide](https://hackjunction.app/participate/45/guide)
- [Intel challenge](https://hackjunction.app/participate/45/challenges/330)
- [Intel Discord](https://discord.com/channels/1268184974784336004/1564622436773658624)
- [Pinned setup instructions](https://github.com/open-edge-platform/physical-ai-studio/blob/junction2026/JUNCTION2026.md), saved as JUNCTION2026-official-setup.md. Retrieved blob SHA: a49c3365ae499ea063be5567e009b1e19a81d3f9.
- Intel_Physical_AI_Hackathon.pdf: original kickoff presentation posted by Rajesh at 22:08, downloaded from Discord.
- Ignite_Next_Challenge_Terms_Junction_2026.pdf: original one-page terms, version 1.0, saved from [Drive](https://drive.google.com/file/d/1nKs-e-4FNk-AoMq1fCeRKM_6mpIkArRt/view).
- [Modal credit instructions shared by Marvin](https://modal-com.notion.site/Modal-at-Junction-2026-use-cases-for-every-challenge-4161e7f16949816e9540de36da40a90a): read and redeemed; $100 Junction grant verified active. See Junction-credit-status.md.

## What to build

Official task: use an SO-101 arm to pick Duplo bricks, present them to an inspection camera, distinguish normal bricks from marked defective bricks with Anomalib, and autonomously place them into good/defective bowls. Record teleoperation demonstrations and train the robot policy in Physical AI Studio. Train in the cloud and deploy with OpenVINO on the supplied Intel laptop. The detector must actually influence robot action; separate component demos are insufficient.

Recommendation: complete that loop early. Your distinguishing feature should be uncertainty-driven reinspection or recovery, added only after a stable baseline. Make a small interface showing the camera view, anomaly heatmap/score, decision, robot state, actual execution device, cycle count and latency. Avoid a chatbot, extra app infrastructure, or a new robotics framework.

## Hardware and people

The challenge lists 25 stations, one per team: Core Ultra Series 3 Panther Lake laptop with 64 GB RAM; SO-101 leader/follower with wrist camera; two external cameras including a RealSense D415; bricks, bowls, markers; cloud GPU access. The slides identify HP OmniBook Ultra or Dell XPS 16 machines with Core Ultra X9 388H, Ubuntu 26.04 and preinstalled drivers, and Anomalib 2.7.0. Inspect your actual machine before installing anything: generic Intel hackathon documentation describes an older Ubuntu/software configuration.

Discord: stations were first come, first served; receiving the laptop commits the team to Intel and it is only for that challenge. Olinca asked each team to have three bowls. Daniil offered additional clamps. Do not infer a third reject category from the bowl count; confirm the intended layout.

Mentors named on the kickoff slides: Albert van Houten, Daniil Lyakhov, Jayabalaji Sathiyamoorthi, Olinca Beltran, Rajesh Gangireddy, Ada Betancourt, Marvin Mueller, Ricky Watts. Main contact: Marvin Müller, marvin.mueller@ignitenext.com. Challenge area: Hardware Lab, floor 5.

## Three conflicting rubrics

The challenge page:

| Criterion | Weight |
|---|---:|
| Task success: accuracy, cycle time, shifted/cluttered bricks | 30% |
| Intel integration and NPU/iGPU offload | 30% |
| Dataset and pipeline quality | 20% |
| Efficiency: latency, thermals, power | 20% |

The pinned repository guide: task success 25, Intel integration 25, dataset/pipeline 20, efficiency 20, creativity 10.

Kickoff slides, page 7:

| Criterion | Points |
|---|---:|
| End-to-end Physical AI solution | 25 |
| Defect detection with Anomalib | 20 |
| VLA and Physical AI Studio integration | 20 |
| OpenVINO and Core Ultra optimization | 20 |
| Robotic execution and reliability | 10 |
| Innovation and technical demonstration | 5 |

Ask which is final. Terms refer to criteria announced at kickoff, making the slides particularly relevant, but that does not establish that no later update superseded them. Build evidence for all shared priorities while awaiting clarification. Do not spend the weekend optimizing for speculative creativity points.

## Setup traps

Rajesh explicitly instructed teams to use the junction2026 branch for training AND inference. Use the saved event-specific guide over generic main-branch tutorials.

Backend Modal authentication is separate from browser login. Configure the backend's credentials and then deploy/connect its Modal training target. The guide advertises $100 credit per participant; confirm redemption and budgets. Closing Studio does not stop running cloud jobs or billing.

Keep Anomalib in its own environment. Studio is for robot learning; anomaly detection is a separate component. Start with a mentor-supported baseline, check its OpenVINO export immediately, and only then collect a large dataset. The guide suggests locally training anomaly models on XPU, preserving Modal credit for robot-policy training.

Use a read-scoped Hugging Face token where required. Never place tokens in the public submission. Do not overwrite an existing configured .env just because setup instructions show copying an example.

## Implementation plan (recommendation)

1. Secure mounts, calibrate the arms, verify camera identity, teleoperation and safe stop with a mentor. Mark camera/bowl locations; preserve calibration and configuration.
2. Decide what data the detector passes to the robot: good/defective decision, score, and optionally defect location. Define completion and timeout conditions for each phase.
3. Build one successful pick-inspect-sort cycle. Use the supplied policy path first. If choosing ACT, confirm how branching is implemented; do not assume a non-language policy follows task strings. Ask mentors whether separate skills or a hybrid controller meet VLA requirements.
4. Record clean, consistent demonstrations that vary brick location and orientation within a bounded workspace. Include both destinations. Remove collisions, failed grasps and long idle portions. Keep unseen placements for evaluation.
5. Collect normal inspection images covering legitimate brick colours, poses and lighting. Keep defective examples for validation/testing as appropriate to the model. Split by physical brick or capture session, rather than nearly identical neighbouring video frames.
6. Train and export an anomaly baseline. Choose the threshold on validation data, freeze it, then evaluate on held-out bricks. Count defects falsely accepted separately from good bricks falsely rejected.
7. Integrate the autonomous loop. A human must not choose each destination. Limit retries, detect stale/missing frames, and use the supported safe stop on unrecoverable failures.
8. Benchmark the exported pipeline on supported devices. Candidate allocation: CPU for orchestration, iGPU for robot policy, NPU for a supported vision stage. This is a hypothesis to test, not guaranteed model compatibility or a required allocation.
9. Add one recovery feature: if the inspection view is inconclusive, reposition for a second view and retry within a fixed budget. Compare against the baseline on the same trial set. An anomaly score alone is not a calibrated probability of uncertainty; distinguish poor visibility from a clear defect.

Do not change camera geometry after training without assessing the effect. Do not quantize until the uncompressed exported model works. Preserve that model for rollback and remeasure task success after optimization.

## Evidence that earns points (recommendation)

Run at least 20-30 documented autonomous attempts if time permits; this is a suggested small hackathon evaluation, not an official minimum or proof of production reliability. Include normal placement, shifted/rotated bricks, clutter and unfamiliar defect markings. Report counts per condition.

- End-to-end success: correct picks, inspections and destination placements divided by all attempts, with no hidden manual intervention.
- Defect quality: confusion matrix, defect recall, false accepts and false rejects.
- Performance: median and p95 complete cycle time, separate perception/policy latency, throughput including retries.
- Deployment: model/precision/device table, actual execution-device evidence, and CPU baseline versus chosen configuration.
- Efficiency: sustained temperature and power readings where available, specifying measurement scope. Never equate CPU package power with total robot-system power. Label estimated energy as estimated.
- Dataset: number of clean episodes, retained/rejected counts, collection time, coverage and held-out evaluation.
- Recovery: successful recoveries and extra time spent, with before/after comparisons.

Record every test attempt in a simple CSV and save an uninterrupted multi-cycle video. Keep a separate polished demonstration video. Never present cherry-picked footage as your success-rate measurement.

## Execution schedule (recommendation; Helsinki time)

- Friday draft milestone is now historical: verify the team has a draft and correct challenge selected; do not wait for Sunday to prepare submission fields.
- Saturday morning: one autonomous end-to-end cycle and both sorting outcomes; fix integration before polishing.
- Saturday 12:00: Intel technical session. Venue is inconsistent: main guide schedule says floor 2 registration area, detailed workshop listing and public website say floor 5 Hardware Lab. Confirm in Discord or at the desk.
- Saturday afternoon: held-out evaluation, dataset improvements, OpenVINO deployment and device measurements.
- Saturday evening: freeze baseline, add only one justified improvement, collect honest results, record the video and draft slides.
- Sunday 08:00: feature freeze; rehearse and check links with a signed-out browser.
- Sunday 09:00: target submission in FINAL state, leaving an hour for upload or access problems.
- Sunday 10:00: hard final deadline, no late submissions.
- Sunday from 10:30: be present for peer and partner review. Guide schedule says 10:30-13:00; reviewing section says 10:30-14:00. Reserve through 14:00 until organizers clarify.

For four people, assign robot/data, anomaly detection, deployment/integration, and evaluation/demo/submission owners. The integration owner maintains the runnable baseline. If fewer people, combine roles but keep one submission owner. Arrange sleep off-site: the participant guide says sleeping at the venue is prohibited.

## Winning the overall event

Intel selects its challenge winners. The Junction overall finalists are selected through face-to-face peer review; top five then pitch to seven judges. Junction's general criteria are innovation, impact and implementation. Main prize: EUR 10,000; Intel advertised prizes: EUR 5,000 / 3,000 / 2,000.

Translate the technical demo into a concrete user benefit: a small manufacturer teaches an affordable inspection cell through demonstrations, and it runs locally. Treat industrial deployment as the future application; this weekend proves a bounded prototype. Do not invent customer interviews, cost savings or factory readiness.

Prepare a 30-second explanation, a repeatable tabletop demo, a video of at most 120 seconds, and a separate three-minute stage pitch followed by three minutes of questions. The guide says the stage microphone is cut at three minutes and there is no time to prepare after finalist selection.

Suggested 120-second video: 0-15 seconds problem; 15-65 uninterrupted working task; 65-90 architecture and substantive Intel usage; 90-110 measured results and recovery; 110-120 impact and limits. Aim for 115-118 seconds in the encoded upload.

Suggested punchline: "Teach an inspection cell by demonstration; inspect and sort locally on one AI PC." Replace broad claims with measured results: "X/Y held-out cycles, Z seconds median, N demonstrations, model on device D."

## Submission and eligibility checklist

From the participant guide:

- Team size 1-4; participate in at least one challenge.
- No idea-specific code from before the event. Open-source libraries are allowed; the guide also prohibits projects related to an existing company.
- Friday draft identifies the intended challenge; final projects can target multiple challenges, subject to partner-specific restrictions.
- One team member submits project name, punchline, description, challenge, technologies, public pitch video <=2 minutes, demo/source links, source publicity/credit choices, PDF pitch deck, team phone numbers and Discord username.
- Set DRAFT to FINAL before Sunday 10:00. Drafts are not reviewed. FINAL submissions can still be edited until the deadline.
- Organizers must be able to access source; submissions carry the MIT licence.
- All team members must participate in assigned peer reviewing to remain eligible for the main prize; coordinate keeping your assigned demo table staffed.
- Keep phone/Discord reachable for finalist or prize contact.

## Terms and unresolved questions

Terms summary: kit is borrowed, challenge-only, venue-only; do not open/modify/remove the kit from the Hardware Lab. Follow lab safety rules and report faults. Back up work before cloud shutdown and before returning/wiping the laptop. Keep cloud credentials within your team. Eligibility is 18+ or written guardian consent. Teams retain project rights, while sponsor publicity may include team name, project description, images and demo videos. Resources have separate licences, including evaluation terms for pre-release items. Prize delivery is stated as within ten days after winners confirm details; winners bear taxes. Junction rules take precedence.

Clarify with mentors, without blocking baseline work:

1. Which of the three scoring rubrics is final?
2. Which policy/model/export combination is known to work on these machines, particularly NPU support?
3. What are the exact test conditions, object/defect rules, retry penalties and permitted manual resets?
4. How will a hybrid controller or ACT-based skill selection be scored under the VLA requirement?
5. Which power/thermal measurement tools and device evidence do judges expect?
6. Are the euro-listed prizes cash? The terms say prizes cannot be exchanged for cash, which is ambiguous alongside the challenge page.
7. Confirm workshop location, review window, and hardware demo logistics if selected for the main stage. Do not move the borrowed kit out of the lab without staff authorization.

No models, training jobs, hardware performance or actual prize payout were tested or confirmed by this research. No Discord messages were sent.
