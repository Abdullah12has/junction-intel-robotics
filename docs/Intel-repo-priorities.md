# Intel repo assessment and first build

Reviewed 13 November 2026. Repository: https://github.com/Abdullah12has/junction-intel-robotics

GitHub's contents API returned “This repository is empty”; repository metadata showed size 0, no language and no license. This assesses the published repository, not unpushed work on the Intel laptop. No code was changed or pushed.

For the November 14 research update and current model decisions, see [speed and reliability plan](Intel-speed-and-reliability-research.md). The empty-repository observation above is historical; documentation has since been added.

## Build first

Build one autonomous pick → inspection → good/defective sort cycle. Anomalib's result must choose the destination, and the robot policy must run through the supplied Physical AI Studio/OpenVINO path. A detector demo beside a teleoperated arm is not a complete solution.

Start with the official junction2026 branch for BOTH training and inference:
https://github.com/open-edge-platform/physical-ai-studio/blob/junction2026/JUNCTION2026.md

Do not rebuild Studio or make a separate app framework. Your repository should contain the small amount of integration and evidence that is specific to your team.

## Ordered milestones and acceptance checks

1. Hardware baseline: secure mounts, calibrate follower/leader, verify camera identities and safe stop, then successfully teleoperate pick/present/place. Preserve calibration. Check actual laptop software before installing dependencies.
2. Data and perception in parallel: record consistent demonstrations with position/orientation variation; capture normal inspection images and held-out good/defective examples. Train a fast supported Anomalib baseline and prove OpenVINO export immediately. Split evaluation by brick/session, not adjacent frames.
3. First complete cycle: execute pick, confirm inspection position, capture a fresh frame, classify, select destination, place and return. Log every attempt. Demonstrate both destinations without a person choosing them.
4. Repeatability: run a documented 20–30 attempt trial set if time allows. Count dropped bricks, failed picks, wrong bowls, manual rescues and timeouts as failures. This is a proposed hackathon benchmark, not an official minimum or a claim of industrial reliability.
5. Intel optimization: record actual model execution devices and compare a working baseline with supported iGPU/NPU configurations. Preserve model versions and precision. Measure complete cycle latency as well as inference; include retries. Quantize only once baseline quality is known.
6. One differentiator: on a poor inspection view, take a second bounded view and retry; stop safely after the retry budget. Compare success and cycle time with recovery disabled. Anomaly score is not automatically calibrated uncertainty.
7. Submission: MIT source, reproducible launch instructions, accessible model/data manifests, measured results, <=2-minute video and PDF deck. Set submission FINAL before Sunday 15 November 10:00 Helsinki; target 09:00. Prepare Friday preliminary submission before 23:55 as a buffer.

## What is missing in the published repo

| Missing | Smallest useful deliverable |
|---|---|
| Runnable entry point | Thin Python orchestrator using provided inference APIs; explicit states and timeouts |
| Actual robot learning | Demonstration collection, trained policy artifact reference, Studio configuration and lineage |
| Actual inspection | Anomalib training/export path, saved threshold, good/bad result tied to destination |
| Safety/recovery | Supported safe stop, missing/stale-frame checks, bounded retries, fault state |
| Reproducibility | README, pinned upstream branch/commit and working package versions, config example without secrets |
| Data quality | Episode counts, rejected takes, coverage, split manifest, dataset/model references |
| Evaluation | Per-attempt CSV; end-to-end success, defect confusion matrix, p50/p95 cycle time |
| Intel proof | Actual execution-device logs, precision table, comparative latency and sustained thermal/power scope |
| Submission assets | MIT license, demo video, PDF deck, accessible links |

Suggested first files, not a prescribed architecture: README.md, LICENSE, .gitignore, run_cell.py, config.example.yaml, models/manifest.json, data/manifest.json, evaluation/trials.csv. Keep large recordings/weights outside Git, referenced by version/hash. Avoid introducing config formats if existing upstream configuration already suffices. Ignore .env, tokens, credentials and raw large artifacts.

Add small offline tests for timeout, stale camera input and retry exhaustion when implementing the orchestrator. Tests cannot validate calibration or safe motion; those need supervised hardware checks.

## Four-person split for Prometheus.

- Robot owner: calibration, camera layout, clean demonstrations, policy training.
- Vision owner: inspection dataset, anomaly model, threshold and export.
- Integration owner: cycle controller, deployment, logging and failure handling.
- Evidence/submission owner: held-out trials, metrics, demo, deck, deadlines; help collect data while waiting for training.

At each integration checkpoint, all four should work against the same known-good configuration. Do not wait until Saturday night to join the components.

## What could separate you from other teams

Reliability on unseen placements, low defect false-accept rate, an honest uninterrupted multi-cycle video, and a measured recovery improvement are stronger than a large feature list. Explain the value as an affordable inspection cell taught through demonstration and running locally. Do not claim factory readiness from a small demo.

Qualia's dataset curation fits the data-quality criterion directly: https://hackjunction.app/participate/45/hackerpacks/53 . Use it if setup is quick; its purpose is removing bad episodes and preserving dataset versions. Modal supports the required policy training. Additional services should not delay a working robot.

Intel mentors suggested other Open Edge Platform repositories at 22:46–22:49 on Discord as a creativity option: https://github.com/orgs/open-edge-platform/repositories?q=sort%3Astars . Treat this as an optional extension after the baseline.

## Clarify with mentors while building

Three rubrics disagree: challenge page 30/30/20/20; pinned setup 25/25/20/20/10; kickoff slides end-to-end25, Anomalib20, VLA/Studio20, OpenVINO20, reliability10, innovation5. Ask which applies. All reward complete integration, data quality, Intel execution and reliability.

Also confirm the known-working VLA/export/device combination; whether ACT or separate learned skills plus a controller meet the VLA requirement; exact test placements, defects and reset/retry rules. Never assume an ACT policy understands a text command.

Sources: official challenge https://hackjunction.app/participate/45/challenges/330 ; guide https://hackjunction.app/participate/45/guide ; Intel Discord https://discord.com/channels/1268184974784336004/1564622436773658624 ; saved official setup and kickoff slides in this folder. See Intel-winning-plan.md for the full event brief.
