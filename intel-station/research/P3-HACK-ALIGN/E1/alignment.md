# P3-HACK-ALIGN alignment (E1, stage align-v1)

Status: PARTIAL. Written 2026-11-14 (UTC) by worker E1. Companion files in both folders: requirements.json, metrics.json, STATUS.md. laptop-log.md is in the sandbox folder only.

Question: are we aligned with the Intel Physical AI challenge as the materials state it? Short answer: partly. We have most of the parts, but no recorded end-to-end run.

## Sources and abbreviations

- BRIEF = /home/intel/intel-junction-2026/discord/Intel_Physical AI Hackathon.pdf (15 slides; text layer on slides 1-3 and 8-15; slides 4-7 read as rendered images).
- GUIDE = /home/intel/intel-junction-2026/repos/physical-ai-studio/JUNCTION2026.md (140 lines, read in full).
- ARCH = /home/intel/intel-junction-2026/submission/architecture.md (grep and selected lines).
- RULES = /home/intel/intel-junction-2026/app/loop_v2/app/loop/policy_rules.yaml (grep and selected lines).
- ANOM = /home/intel/intel-junction-2026/app/anomaly/anom3/models/v000/pre_close/metrics.json (fields read by grep).
- FACTS = /home/claude/kb/shared/FACTS-LIVE.md (sandbox shared file; grep and selected lines).
- MODAL = /home/claude/kb/shared/T09B-MODAL-BROWSER.md (sandbox shared file; grep and selected lines).
- MODELS = /home/claude/kb/shared/curation/models.tsv (sandbox; rows 1-7).
- REQ = requirements.json in this folder: 77 entries, each with a source, one quote of 15 words or fewer, and a checker result.

Not read: the mentor answers (outside the allowed inputs) and the team folders /home/intel/junction-cell and /home/intel/junction-handoff. Facts about team work come only from FACTS and are marked as such.

## Summary

- The brief sets four must-deliver items (BRIEF slide 9): a working closed-loop SO-101 prototype, VLA plus Anomalib integration, OpenVINO-optimised deployment, and a live Challenge-Day demonstration with explanation.
- Most parts exist. Studio runs locally on ports 3000 and 7860 (GUIDE lines 30, 35, 42; FACTS line 11). Anomalib 2.7.0 is installed, and PatchCore exports to an OpenVINO IR (REQ GD-04). Four ACT models and one SmolVLA model are listed with an OpenVINO backend (MODELS rows 2-6). Run B measured CPU and NPU (ARCH lines 59-67). Gates G-A (14:13Z) and G-M (14:33Z) passed (FACTS).
- No end-to-end run is recorded. The 16:09Z spot test moved the arm but did not grasp the brick (REQ JC-05). The loop has no runnable entry point (ARCH line 104).
- Counts over 73 rated rows: met 17, partial 49, missing 3, unknown 3, not applicable 1. Four further entries are background or open (MD-05, OQ-01, OQ-02, OQ-03). The checker output is in metrics.json.
- Largest gaps: no closed-loop run or grasp success (BR-22, JC-05, RB-01); placeholder model ids (RULES lines 31, 40; BR-08); no runnable entry point (ARCH line 104); live GPU use for ACT not verified, because RULES says device auto (ARCH lines 77, 108; GD-13); Robotics AI Suite not used (RB-02); no PaDiM decision (GD-05).
- Open decisions: the rubric conflict (OQ-01), the Challenge Day date and main submission format (OQ-02), and mentor answers (OQ-03). The Modal track closes at the end of Monday 16 November (MODAL lines 50, 422).
- Model fusion: the term does not appear in the materials. The brief and the guide do require detection to feed the VLA or robot workflow (must-deliver 2; GUIDE Step 6). Our wiring exists on paper only (section c).
- E1 did not open cameras, serial ports, Studio or any process, and did not command the arm. Laptop files were read only, except this folder's own files.

## Requirement table

Effort: S = under 1 hour; M = 1 to 3 hours; L = more than 3 hours, or a robot motion run that only Opus may authorise.

| ID | Requirement | What we have (source) | Gap | Recommendation (effort) |
|---|---|---|---|---|
| BR-01 | End-to-end physical AI solution for defect detection and manipulation | Parts exist: det2_v1 live (FACTS 16:04Z); gates G-A 14:13Z and G-M 14:33Z (FACTS); ARCH status 15:11Z | No full live run recorded | Log one end-to-end pass: frame, det2 decision, routing, SO-101 motion (L; Opus authorises the motion) |
| BR-02 | VLA workflow combined with Studio and Anomalib detection | RULES plates lines 28-43; ACT on Modal (FACTS 14:08Z); gate G-A | Policy path never run live; RULES lines 31 and 40 hold PLACEHOLDER model ids | Replace the placeholders with the anom3 model id; run one decision-to-policy trace with no motion (M) |
| BR-03 | Perceive objects, identify defects, reason, act | det2_v1 live (FACTS 16:04Z); app/loop_v2/app/loop/triage.py exists | triage.py never run live | Run triage.py offline on 10 saved frames and log the decisions; no camera, no arm (S) |
| BR-04 | Physically sort or respond per the kickoff rules | Kickoff rule and OD-10 (FACTS line 1; FACTS 13:47Z): good to blue plate, defective to green; RULES plates | Routing not run on the robot; the team note in FACTS line 20 says good to green bowl, defective to red bowl | Settle the rule with Hassan (see requests); then one sorted run under the motion rule (M) |
| BR-05 | Use CPU, GPU and NPU acceleration | ARCH lines 59-67 (run B: ACT and PatchCore on CPU and NPU); GPU in offline replay, line 77 | Run B not precision-matched (ARCH line 108) | One precision-matched benchmark on CPU, GPU and NPU for the same model (M) |
| BR-06 | Closed loop through five stages (PERCEIVE, DETECT, REASON, ACT, OPTIMIZE) | BRIEF slide 3 pipeline; ARCH sections 1-2 (status 15:11Z) | Stage mapping table not re-read; OPTIMIZE only in offline replay | Write a five-row stage map with file and line for each stage (S) |
| BR-07 | Optimize and deploy supported AI workloads with OpenVINO | ARCH lines 59-67 (ACT FP32 and FP16, PatchCore FP16); offline replay GPU p50 9.29 ms vs CPU 121.00 ms (line 77) | No end-to-end optimised loop run | Deploy the ACT and PatchCore IR in the loop path and time it (M) |
| BR-08 | Anomalib output consumed by the VLA or robot workflow (fusion point) | RULES lines 31, 40 (PLACEHOLDER ids); triage.py | Wired on paper only | Replace the placeholders; pass the decision into the policy call and log what the policy receives (M) |
| BR-09 | Map inference over NPU, GPU and CPU | ARCH lines 59-77 (device rows); D-080 (ACT on GPU, line 77) | Device rows come from separate runs | One table per model: device, precision and latency from the same run (M) |
| BR-10 | SO-101 closes the loop to real action | FACTS 14:13Z G-A passed (probe_handover run); 14:33Z G-M passed | No closed-loop run | One logged closed-loop cycle with camera trace, after the grasp fix (L) |
| BR-11 | Design and integrate the VLA-based robotics workflow (perception, task reasoning, robot control, orchestration) | RULES studio block lines 18-19 and plates lines 28-43; ARCH section 6 item 1 | No runnable entry point (ARCH line 104: no run_policy.py, no __main__, no dry-run flag) | One entry script with a dry-run flag that runs perception, routing and the policy in order (M) |
| BR-12 | Detector gives an anomaly score and localization | det2 decision and tape_frac (FACTS 16:04Z, 16:09Z); ANOM threshold | Localization not evidenced | det2 logs a region (box or heatmap maximum) with the score for one frame (S) |
| BR-13 | Anomalib identifies anomalous or defective objects | ANOM: auroc 0.869, n_good 12, n_defect 7, fit_on_heldout true; FACTS 16:04Z | Threshold fit on held-out data; small sample | Report AUROC on a separate test split, with counts (M) |
| BR-14 | Defect output reaches the policy as signal, label, score or region | RULES plates (PLACEHOLDER ids); BRIEF slide 6 (image): the defect output should reach the downstream policy | No live check of what the policy receives | Pass the decision and tape_frac as a label or score, and log the policy input (M) |
| BR-15 | VLA application flow built in Studio | RULES studio block (project ab767473; environment SO101 Recording - two cameras); ARCH line 94 (route A, D-042) | Not run as one Studio flow end to end | Run Studio route A once with ACT or SmolVLA and save the output (M) |
| BR-16 | VLA reads task text and perception results | Per-plate task text (RULES lines 32, 41); MODELS row 5 (SmolVLA, language-conditioned) | No live check; plate strings differ from the dataset strings (FACTS 14:00Z) | Align the plate task strings with the dataset; one task-text run (S) |
| BR-17 | Action becomes robot motion on the SO-101 | FACTS 14:13Z and 14:33Z gates passed; 16:09Z spot pick: 23 of 23 steps arrived, brick not grasped | Grasp not validated | Fix the grasp and repeat with a brick under the jaws (L; Opus authorises) |
| BR-18 | OpenVINO optimisation of supported models | ANOM model.xml (FP16 IR); MODELS rows 2-6 with openvino backend | No output-parity or speed check recorded for the ACT or SmolVLA exports | Parity check on 5 frames and timing for each exported model (M) |
| BR-19 | Consider conversion, precision, graph optimisation, pre/post-processing, device | ARCH lines 59-67 (FP32 and FP16 IR rows) | Graph optimisation and pre/post-processing not in the rows read; precision not matched | Add precision, pre/post-processing time and device columns to the benchmark (M) |
| BR-20 | Deploy on Core Ultra Series 3 | FACTS 11:40Z: Intel Core Ultra X9 388H; components run locally | Full loop not run on the device | Include the loop run in the BR-07 timing run (S) |
| BR-21 | Choose NPU, GPU or CPU per workload | ARCH line 76 (PatchCore GPU first, NPU time-boxed; D-033, D-045); line 77 (ACT on GPU, D-080) | PatchCore batch 1 not benchmarked (ARCH line 76) | Benchmark PatchCore at batch 1 on the NPU; write each device choice with its reason (M) |
| BR-22 | Demonstrate full sequence from image to detection to robot action | None; ARCH line 104 (no runnable entry point) | Missing | Same run as BR-01 and BR-11 (L; Opus authorises) |
| BR-23 | Concise architecture and optimisation explanation | BRIEF slide 6 (image): a concise explanation of architecture and optimisation decisions; submission/architecture.md and README.md drafts | Architecture status may be stale (15:11Z); evidence.md not reviewed | Refresh the architecture status and reconcile it with FACTS (S) |
| BR-24 | Must deliver 1: working closed-loop SO-101 prototype | BRIEF slide 9 (text layer) | As BR-01 and BR-10 | As BR-22 (L) |
| BR-25 | Must deliver 2: VLA and Anomalib integration | BRIEF slide 9 (text layer) | No live run; placeholders in RULES | As BR-02 and BR-08 (M) |
| BR-26 | Must deliver 3: OpenVINO-optimised deployment | BRIEF slide 9 (text layer); ARCH section 3 run B; ANOM FP16 IR | Deployment path not run end to end | As BR-07 (M) |
| BR-27 | Must deliver 4: live Challenge-Day demonstration with explanation | BRIEF slide 9 (text layer) | No run yet; Challenge Day date not in the materials | Ask Hassan for the date (OQ-02); rehearse a run after BR-22 (L) |
| BR-28 | Install Physical AI Studio and Anomalib for the challenge | FACTS 13:24Z Studio state (line 11); GUIDE lines 30 and 35 (ports 3000 and 7860); FACTS 14:43Z anomalib 2.7.0 environment | None | Keep Studio running; no change needed (S) |
| BR-29 | Anomalib version 2.7.0 | FACTS 14:43Z (anomalib-env 2.7.0); ANOM field 2.7.0; BRIEF slide 8 (AnomaLib v2.7.0) | None | State the version in the README (S) |
| BR-30 | Train on a Modal GPU instance | FACTS 14:08Z: Modal trainer ca5568b5 (A10G), job dccc8afa, 2056 steps | None | Keep the job record in submission/evidence.md (S) |
| BR-31 | Pre-installed Ubuntu, NPU driver and iGPU driver | FACTS 11:40Z: Ubuntu 26.04.1; /dev/accel/accel0 (intel_vpu); /dev/dri/renderD128 (xe); BRIEF slide 8 | None | Cite the device nodes in the README (S) |
| BR-32 | One RealSense D415 and one Pro QHD webcam (kit) | BRIEF slide 8 (text): one RealSense Depth Camera 415 and one Pro QHD 2K/60fps webcam; FACTS: two RealSense D415 (line 1), wrist camera USB2.0_CAM1 (12:00Z); no Pro QHD in FACTS | Our kit differs from the brief | Ask Hassan whether the Pro QHD webcam is available; if not, document the substitution (S) |
| BR-33 | SO-ARM101 leader and follower with one wrist camera | FACTS line 1 (robots calibrated); FACTS 12:45Z (wrist camera) | Brief says SO-ARM101, guide says SO-101; same arm not verified | Note in the README that the two names refer to the same arm (S) |
| BR-34 | Core Ultra X9 388H AI PC (HP OmniBook Ultra or Dell XPS 16) | FACTS 11:40Z line 4 (Intel Core Ultra X9 388H, Panther Lake) | None | None (S) |
| JC-01 | End-to-End Physical AI Solution (25 points) | As BR-01; handover probe and gates only | No closed-loop run | As BR-01 (L) |
| JC-02 | Defect Detection with Anomalib (20 points) | det2_v1 live (FACTS 16:04Z); 16:09Z false CONFIDENT_GOOD on a taped brick (tape face hidden from the close camera); ANOM fit_on_heldout true | Weakly evidenced: 7 defect images | Confusion table on a held-out set; fix the taped-brick case by camera view or threshold (M) |
| JC-03 | VLA and Physical AI Studio Integration (20 points) | As BR-02, BR-15, BR-16; BRIEF slide 7 (image): integration across perception, reasoning, planning and manipulation should be clearly demonstrated | No live demonstration of the integration | As BR-02 (M) |
| JC-04 | OpenVINO and Core Ultra Series 3 Optimization (20 points) | ARCH lines 59-67 (run B) | No power (needs root, ARCH line 109); precision not matched | Measure power without root, or document the block; precision-matched rerun (M) |
| JC-05 | Robotic Execution and Reliability (10 points) | FACTS 14:13Z (G-A) and 14:33Z (G-M); 16:09Z spot test 1: brick not grasped; 16:16Z hand watchdog validated | Repeatability not measured | N grasp trials with recorded successes (L; Opus authorises) |
| JC-06 | Innovation and Technical Demonstration (5 points) | ARCH; app/dashboard_v5 exists (not reviewed) | No written innovation claim | Three-line claim with evidence, e.g. UNCERTAIN routed to GREEN as a fail-safe (FACTS 16:04Z) (S) |
| RB-01 | Task Success Rate (25 points, guide rubric) | No success-rate run; GUIDE rubric: accuracy, cycle time, robustness to shifted objects and clutter | No run; cycle time not measured | Define N trials and a success rule; run after the grasp fix (L) |
| RB-02 | Intel Tech Integration (25 points, guide rubric) | ARCH lines 61, 67 (NPU offload measured); line 77 (GPU offline replay); GUIDE line 136 names the Robotics AI Suite; BRIEF slides 11 and 14 name it | Robotics AI Suite not used; ARCH does not mention it | Decide with Hassan; if used, add one documented integration point (M) |
| RB-03 | Dataset and Pipeline Quality (20 points, guide rubric) | curation/datasets.tsv: S1 v0 4 episodes, 2569 frames; FACTS 15:03Z: PP-BLUE-v2 3, PP-GREEN-v2 2, MT-ALL-v2 9 episodes; MODELS labels: PP-BLUE v1 20, PP-GREEN-v2 130, PP-BLUE-v2 50, MT-ALL-v2 8 | Datasets very small; counts conflict | Reconcile counts against Studio dataset 6d6bdce3; state one count per dataset (S) |
| RB-04 | System Efficiency (20 points, guide rubric) | ARCH lines 59-77 (latency per device); line 85 (ACT thermal on CPU only); line 109 (power needs root) | No power; no thermal on NPU or GPU | Thermal on each device in the same run; power by a root-free method or a documented block (M) |
| RB-05 | Creativity (10 points, guide rubric) | FACTS 16:04Z line 35 (UNCERTAIN routed to GREEN as fail-safe); FACTS 13:45Z line 18 (two RealSense views) | Subjective | One demo clip of the fail-safe, stated as the creative point (S) |
| GD-01 | Use Anomalib for detection | ANOM PatchCore IR; det2_v1 (FACTS 16:04Z) | None | None (S) |
| GD-02 | Train anomaly models on the laptop Intel GPU (XPU) | GUIDE Step 4 note: "They are small enough to train locally"; PatchCore fit on CPU after the XPU failed (FACTS 15:03Z) | XPU training not working in our run | Note the XPU failure in the README; keep the CPU fit (S) |
| GD-03 | Train on good images and cover what good means | ANOM n_train 6, n_good 12; GUIDE Step 4 hint on covering good | Only 6 training images | Collect more good images across lighting and placement, then refit (M) |
| GD-04 | Export the anomaly model to OpenVINO | ANOM models/v000/pre_close: model.xml and model.bin (FP16) | None | None (S) |
| GD-05 | Consider the listed anomaly models (PaDiM row) | GUIDE Step 4 table (PaDiM, PatchCore, Dinomaly, EfficientAd, FoundAD); ARCH lines 65, 67 (PatchCore IR) | No PaDiM decision in ARCH (an earlier PaDiM benchmark claim was wrong and is removed) | One-line decision on PaDiM, chosen or rejected with a reason (S) |
| GD-06 | Account for language-conditioned and plain policies | RULES plates (task text); MODELS row 5 (SmolVLA, language-conditioned); ACT per plate; GUIDE Step 5 hint (line 111) | No live check | README table of plate to policy; one language-conditioned run (M) |
| GD-07 | Robustness to shifted objects and clutter | FACTS 13:08Z (cluttered scenes recorded); GUIDE Step 5 hint (judges score robustness) | No robustness test | Shift test on saved clips, offline (M) |
| GD-08 | Review episodes before training | FACTS 13:45Z: Opus visual review of 11 episodes | None | None (S) |
| GD-09 | Arms connected: leader and follower powered and connected | FACTS 14:13Z and 14:33Z gates; FACTS line 1 (calibrated); GUIDE line 50 (USB hub) | None | None (S) |
| GD-10 | Stop the arm safely | FACTS 16:16Z hand watchdog validated; torque off at end of G-M (FACTS 14:33Z); GUIDE Step 6 (build a safe stop) | No safe-stop test in policy mode; 16:00Z parked-arm event (FACTS line 34) | Stop test with the policy running (M; Opus authorises) |
| GD-11 | Use Anomalib OpenVINO inferencer | app/anomaly/static_ir.py (OpenVINOInferencer); GUIDE Step 6 useful pieces | None | None (S) |
| GD-12 | OpenVINO PhysicalAI runtime for exported policies | ARCH line 77 (offline replay through InferenceModel); GUIDE Step 6 (runs exported Studio policies outside the UI) | Offline only; the live path is Studio route A (ARCH line 94) | Decide the live path; if live, run one exported policy through the runtime on the laptop (M) |
| GD-13 | Use all three Core Ultra devices | ARCH lines 59-67 (CPU and NPU rows); GPU offline replay (line 77); live ACT GPU per D-080 | Live GPU use not verified: RULES plates say device auto (lines 32, 41; ARCH lines 77, 108), and auto resolves to CPU (line 77). Status downgraded from met to partial in this pass | Set the device explicitly in RULES; re-run one live ACT load and log the device (M) |
| GD-14 | Measure latency per step and per cycle, power, temperature | ARCH lines 59-77 (per-step latency); line 85 (CPU temperature); line 109 (power needs root) | No per-cycle latency; no power; no GPU or NPU temperature | Per-cycle timer in the loop; power and temperature per device (M) |
| GD-15 | Show measured evidence, not claims | ARCH section 3 (benchmark JSON and summaries) | Run B not precision-matched | Precision-matched rerun; table with raw JSON paths (M) |
| GD-16 | Physical AI Studio runs locally (UI port 3000, backend 7860) | FACTS 13:24Z (line 11); GUIDE lines 35 (backend 7860) and 42 (UI 3000) | None | Keep Studio running (S) |
| GD-17 | NVIDIA GPU for VLA training | FACTS 14:08Z: Modal A10G trainer ca5568b5; GUIDE Step 2 (training needs an NVIDIA GPU) | None | None (S) |
| GD-18 | Hugging Face token for Hub-backed policies | FACTS 13:35Z (token saved in Studio settings; value not read); GUIDE checklist line 52 | None | None (S) |
| GD-19 | Decide the role of each camera | FACTS 13:45Z (cam1 top, cam0 close); FACTS 12:55Z (wrist); GUIDE checklist line 51 | None | Write each camera's role in the README (S) |
| GD-20 | Record teleoperation episodes with the leader arm in Studio | Studio dataset 6d6bdce3, 16 episodes (FACTS 14:00Z); curated subsets | Task text differs by episode (FACTS 14:00Z) | Unify the task strings before the next training run (S) |
| GD-21 | Connect detection to action so the robot sorts without someone choosing (GUIDE Step 6) | RULES plates lines 28-43 route by decision (OD-10; FACTS 13:47Z) | No live connection run; method left to the team | Dry run of the connection first, then one live sort under the motion rule (M; L for the live run) |
| GD-22 | Test a trained model on the arm from the Studio Models page with Run model (GUIDE Step 5 hint, line 114) | None; FACTS has no Run model entry | Missing | Use Run model once a model is ready; it is a motion run, so Opus only (M) |
| MD-01 | Modal page suggests moving perception to Modal | Detection stays local, as GUIDE Step 4 says (small enough to train locally) | Conflicts with the guide; not adopted | None (not applicable) |
| MD-02 | Optional Modal track entry alongside the main challenge | MODAL line 418 (F123): submit alongside the main challenge | No decision recorded | Decide with Hassan before the end of Monday 16 November (S) |
| MD-03 | Modal track submissions close Monday 16 November | MODAL line 422 (F127); line 50 | No entry recorded | If entering, submit before the close (S) |
| MD-04 | Modal track judging weights | MODAL line 51 (F121): effective use 40%, technical ambition 25%, impact 20%, demo quality 15% | Applies to the Modal track only | If entering, plan the Modal use first, since it carries 40% (S) |
| MD-05 | Event dates as written on the Modal page | MODAL line 40 (13-15 November, Hype Areena, Espoo); line 53 (dates recorded as written) | None (background) | None |
| BG-02 | Policies named in the materials, none mandated | BRIEF slide 11 (text) lists ACT, SmolVLA, GR00T, LeRobot and Pi0.5; BRIEF slide 9 labels ACT and SmolVLA; GUIDE line 107 (ACT, SmolVLA, Pi0.5); MODELS rows 2-6 | No Pi0.5 or GR00T model; the brief and the guide list different policies | Keep ACT and SmolVLA; Pi0.5 only after a working loop (L) |
| BG-03 | Export Studio models to OpenVINO | MODELS rows 2-6 (backend openvino) | None | None (S) |
| OQ-01 | Two rubrics conflict | BRIEF slide 7 (six criteria, 25/20/20/20/10/5); GUIDE judging rubric (five criteria, 25/25/20/20/10) | No source says which governs | Ask Hassan; plan to both until answered (S) |
| OQ-02 | Challenge Day date and submission format not stated | GUIDE intro (line 13: sorting rules announced at kickoff) and Steps 1-7 give no date or format; BRIEF text layer gives no date; MODAL gives Modal dates only | Missing from all three | Ask Hassan for the kickoff sheet (S) |
| OQ-03 | Model fusion term and mentor answers | Term absent from BRIEF, GUIDE and MODAL (whole-word search: 0); GUIDE Step 6: "How you connect them is up to you." | Mentor answers not read (outside the allowed inputs) | Hassan exports the mentor Q&A into the shared folder (S) |

## (a) Intel tooling

- OpenVINO. BRIEF slide 3 (text) names the Studio, Anomalib and OpenVINO stack and asks for deployment on Core Ultra Series 3 across CPU, GPU and NPU (text lines 18 and 23-24). Ours: run B rows for CPU and NPU (ARCH lines 59-67), an FP16 IR for PatchCore (ANOM), openvino backends on MODELS rows 2-6, and the offline GPU replay (ARCH line 77). Gap: no precision-matched table across the three devices (BR-05, BR-19, GD-15). Recommendation: one benchmark run with the same model and precision on all three devices, raw JSON paths in the table (M).
- NPU and GPU. GUIDE Step 7 asks the team to decide which model runs where and to measure it. The rubric scores offloading to the NPU and iGPU (RB-02). BRIEF slide 8 lists the NPU and iGPU drivers as pre-installed (BR-31; FACTS 11:40Z shows the device nodes). Ours: NPU offload measured (ARCH lines 61, 67). Live ACT on GPU is not verified: RULES plates say device auto (lines 32, 41; ARCH lines 77, 108), and auto resolves to CPU per D-080 (line 77). PatchCore has no batch-1 NPU benchmark (ARCH line 76; BR-21). Recommendation: set the device explicitly in RULES and log the device on one live ACT load; benchmark PatchCore at batch 1 on the NPU (M each).
- Physical AI Studio (datasets, training, inference). Datasets: team recordings in Studio dataset 6d6bdce3, 16 episodes (GD-20; FACTS 14:00Z); the episode counts conflict (RB-03). Training: Modal A10G trainer ca5568b5, job dccc8afa, 2056 steps (BR-30; FACTS 14:08Z); FACTS line 11 also lists a Verda trainer. Inference: Studio route A is the live path (ARCH line 94, D-042), and the Studio flow in RULES lines 18-19 has not been run end to end (BR-15). Studio runs locally on ports 3000 and 7860 (BR-28, GD-16). Recommendation: reconcile the dataset counts, then run route A once (M).
- Anomalib. BRIEF slide 8 lists AnomaLib v2.7.0 (text line 74). GUIDE Step 4 says to install it in its own virtual environment, and its table lists PaDiM, PatchCore, Dinomaly, EfficientAd and FoundAD. Ours: the 2.7.0 environment (BR-29), a PatchCore OpenVINO IR (GD-01, GD-04), and the OpenVINO inferencer used by static_ir.py (GD-11). Gaps: no PaDiM decision (GD-05); XPU training failed and PatchCore was fit on CPU (GD-02); only 6 training images (GD-03). Recommendation: a one-line PaDiM decision (S); document the XPU failure (S); collect more good images (M).
- Geti. No material mentions Geti. A search of the brief text layer, GUIDE, MODAL and the requirement texts found no match. Slides 4-7 were read as images, and no Geti text was recorded from them. A geti directory sits in the repos folder (log entry 5); it was not read. Recommendation: record Geti as not required and do not use it (S).
- MuJoCo plugin. repos/physicalai/packages/physicalai-mujoco-plugin exists (log entries 15 and 18; README lines 1-120 of 742 read). No material mentions MuJoCo or the plugin (GUIDE and brief search: no match). Our own MuJoCo 4.0.1 twin produced 150 scripted episodes; 73 of them contradict OD-10, and the twin was not used for training (FACTS line 25, D-055). Recommendation: record as not required for the challenge; no further work (S).
- Robotics AI Suite. BRIEF slides 11 and 14 name it. GUIDE line 136 scores it under Intel Tech Integration (25 points). Ours: not used; ARCH does not mention it (RB-02). Recommendation: decide with Hassan; if used, add one documented integration point (M).
- OpenVINO PhysicalAI runtime (offline replay). BRIEF slide 11 has an OpenVINO Physical AI Framework column. GUIDE Step 6 says the runtime runs exported Studio policies outside the UI. Ours: offline replay only (ARCH line 77; GD-12). Recommendation: decide the live path; if live, run one exported policy through the runtime on the laptop (M).

## (b) VLA usage

- What the materials expect. No policy is mandated (BG-02). BRIEF slide 11 lists ACT, SmolVLA, GR00T, LeRobot and Pi0.5 as policies Studio can run (text lines 116-117 and 136-140, on pages 11 and 12). BRIEF slide 9 labels ACT and SmolVLA. GUIDE line 107 names ACT, SmolVLA and Pi0.5 as supported, so the brief and the guide list different sets, and GR00T is not in the guide. The VLA expectation itself is must-deliver 2 (BRIEF slide 9) and the scored VLA and Studio integration criterion (JC-03).
- What we trained (MODELS rows 2-6, all openvino backend). Four ACT models: S1 curated v0; PP-BLUE v1 (20 episodes); PP-GREEN-v2 (130 episodes); PP-BLUE-v2 (50 episodes). One SmolVLA model: MT-ALL-v2 (8 episodes, pretrained start; model id d128dc1a).
- Other VLA runs in FACTS (team and shared record). ACT inspection smoke fa42ee76 (1 epoch) and SmolVLA inspection pilot on Verda, 10 epochs, model 6540166d (FACTS line 11). FACTS line 20 records that the pilot ran 20-60 s physical trials from HOME, that the brick was missed or dropped, and that the team planned a SmolVLA and task-specific ACT launch at 15:00Z. Lineage: 6540166d appears only in FACTS; d128dc1a appears only in MODELS row 5. The materials do not link the two. Unresolved (open issue).
- Trainers. Modal: A10G trainer ca5568b5 (FACTS 14:08Z). Verda: an RTX Pro 6000 trainer reached through a tunnel (FACTS line 11). FACTS line 20 records that a team Modal target was blocked on a payment method for A10G (team note).
- Pi0.5 and GR00T: not trained, no model rows, not mandated. Pi0.5 needs the Hugging Face token to download weights (GUIDE checklist line 52). GR00T is not in the guide's supported list (line 107).
- Language handling. GUIDE line 111: some policies accept a language task and some do not, which decides how the detection result reaches the robot. Our plates carry task text per plate (RULES lines 32, 41). The dataset task strings differ (FACTS 14:00Z), and the team's planned strings (FACTS line 20) differ from OD-10. RULES line 28 records D-060: one model per plate, or one text model.
- Recommendation. Keep ACT and SmolVLA, with no Pi0.5 or GR00T, until a closed loop works. Choose one D-060 design and align the task strings (M). Run the Studio Run model test (GD-22) once a model is ready; it is a motion run, so Opus authorises it. Train Pi0.5 only if time allows (L).

## (c) Model fusion

- Is the term in the materials? No. A whole-word search for "fusion" finds 0 matches in the brief text layer, GUIDE and MODAL (grep -c -w, log entry 61). The term appears only in our own labels (BR-08, BR-25 note, OQ-03). A substring search also matches "diffusion" (brief text line 172); that is a false positive.
- What the materials ask for instead:
  - Must-deliver 2 (BRIEF slide 9, text): VLA plus Anomalib integration, a VLA robotics workflow with visual anomaly or defect detection.
  - Slide 4 (image): the detection output must be consumable by the VLA or robot workflow (BR-08).
  - Slide 6 (image): the defect output should be available to the downstream robotics policy (BR-14).
  - Slide 7 (image): JC-02 (Defect Detection with Anomalib, 20 points) and JC-03 (VLA and Studio integration, 20 points). JC-03 asks that integration across perception, reasoning, planning and manipulation be clearly demonstrated.
  - GUIDE Step 6 (read this pass): "Connect detection to action so the robot sorts objects without someone choosing what it should do. How you connect them is up to you."
  - GUIDE Step 5 hint (line 111): some policies accept a language task and some do not, which matters for how the detection result reaches the robot.
- Expected or rewarded? The materials require the integration and score it. They do not name a fusion method (score-level, feature-level or otherwise), and they leave the connection to the team (GUIDE Step 6). So the required item is the integration point, not a named technique. BR-14 lists signal, label, score or region as options, and the materials do not choose one.
- What we have. The det2 decision and tape_frac (FACTS 16:04Z, 16:09Z), and RULES plates that route GOOD to the blue plate and DEFECTIVE or UNCERTAIN to the green plate (lines 28-43, OD-10). The plate model ids are PLACEHOLDER (lines 31, 40). Whether the policy call receives the detection decision is not verified in code, and triage.py has not run live (BR-03, BR-14). No live run shows detection changing the policy's action.
- Recommendation. Pick one integration point (a label or score is the smallest), pass it into the policy call, log exactly what the policy receives, and dry-run the connection before any motion (M). Write the choice in the README so the judges can see the integration (S).
- Mentors. Not read (OQ-03), so any mentor meaning of "fusion" is unknown. If the mentor answers define it, this section must be revised.

## (d) Submission format and deadlines

- Must-deliver list (BRIEF slide 9, text layer): four items (BR-24 to BR-27). Slide 6 (image) asks for a concise explanation of the architecture and optimization decisions (BR-23).
- Format. No source states a submission format (file type, portal, length or video rules). The brief text layer, GUIDE and MODAL give none (OQ-02). The must-deliver items fix the content only.
- Challenge Day. BRIEF slide 9 asks for a live run of the Challenge-Day task with an explanation of the architecture and optimization choices. The date is not in the materials (BR-27, OQ-02).
- Modal track (separate, optional). Event 13-15 November at Hype Areena, Espoo (MODAL line 40). Submissions close at the end of Monday 16 November (MODAL lines 50, 422). Modal credit redemption also ends 16 November (MODAL line 52). The page records both dates as written, one day apart (MODAL line 53). The track is entered alongside the main challenge (MODAL line 418). Judging: effective use of Modal 40%, technical ambition 25%, impact 20%, demo quality 15% (MODAL line 51).
- Main challenge deadline. Not stated in the materials.
- Recommendation. Ask Hassan for the kickoff sheet with the date, deadline and format (S). Decide the Modal track entry before Monday 16 November (S).

## Rubric-conflict note (OQ-01)

- Brief slide 7 (image, read at 110 dpi and in the PDF description image) has six criteria that sum to 100: End-to-End Physical AI Solution (25), Defect Detection with Anomalib (20), VLA and Physical AI Studio Integration (20), OpenVINO and Core Ultra Series 3 Optimization (20), Robotic Execution and Reliability (10), Innovation and Technical Demonstration (5). These are JC-01 to JC-06.
- The GUIDE judging rubric table (Intel row at line 136) has five criteria that sum to 100: Task Success Rate (25), Intel Tech Integration (25), Dataset and Pipeline Quality (20), System Efficiency (20), Creativity (10). These are RB-01 to RB-05.
- The two differ in count, names and weights. No source says which governs. The table keeps both sets covered, so the evidence tracks both. Ask Hassan (OQ-01).

## Coverage checklist

Brief (15 slides):
- [x] Slide 1: text layer read (log entry 11).
- [x] Slide 2: text layer read (log entry 11).
- [x] Slide 3: text layer read; VLA sentence at text line 19; Studio, Anomalib and OpenVINO stack at lines 18 and 23-24.
- [x] Slide 4: rendered at 80 dpi and viewed (log entry 17).
- [x] Slide 5: rendered at 80 dpi and viewed (log entry 17).
- [x] Slide 6: rendered at 80 dpi and viewed (log entry 17).
- [x] Slide 7: rendered at 110 dpi and viewed (log entry 13); also the PDF description image (log entry 20).
- [x] Slide 8: text layer read (kit, AnomaLib v2.7.0, Ubuntu, drivers, Pro QHD).
- [x] Slide 9: text layer read (must-deliver items; Challenge-Day line 93).
- [x] Slide 10: text layer read.
- [x] Slide 11: text layer read (policy list; Robotics AI Suite and OpenVINO columns).
- [x] Slide 12: text layer read.
- [x] Slide 13: text layer read.
- [x] Slide 14: text layer read (Robotics AI Suite).
- [x] Slide 15: text layer read.
- [ ] Images on slides 8-15 (photos, diagrams) not viewed; only the text layer was read. Reason: time. The requirements use the text layer for these slides.

Discord folder:
- [x] Intel_Physical AI Hackathon.pdf (15 pages; pdfinfo, log entry 10).
- [x] github_description.png (rubric; log entry 19).
- [x] pdf_description.png (judging criteria; log entry 20).
- [x] Photo of a projection screen (rn_image_picker_lib_temp_c32b9452): the QR code was not decoded; not a requirement source (log entry 21).
- [x] Photo of slide 3 (rn_image_picker_lib_temp_7cc03046): viewed in an earlier pass; not a requirement source (log entry 22).
- [ ] Portraits among the PDF's embedded images: not identified or recorded (log entry 7). Not needed for any requirement.

Repositories:
- [x] physical-ai-studio/JUNCTION2026.md: 140 lines read in full (log entries 8, 48, 50, 51, 70, 72, 74, 75).
- [x] physical-ai-studio git state: clean, HEAD 285924d9 (log entry 9).
- [ ] Other physical-ai-studio files: not read (not needed for the alignment table).
- [x] physicalai MuJoCo plugin: directory listing (log entry 15); README lines 1-120 of 742 (log entry 18). [ ] Lines 121-742 not read; MuJoCo is not required.
- [x] repos listing (log entry 5): names only. geti, robotics-ai-suite and anomalib directories exist; not read.
- [ ] robotics-ai-suite folder: listing only; not read in this pass.

Laptop application and submission:
- [x] submission/architecture.md: lines 77, 104, 108 and 109 read; sections 1-3 and 6 via grep in an earlier pass. [ ] Full file not read.
- [ ] submission/evidence.md, README.md, recording-checklist.md and storyboard.md: listed only (log entry 3); not read.
- [x] app/loop_v2 policy_rules.yaml: grep for PLACEHOLDER (lines 31, 40), refuse and OD-10 (lines 3, 28); selected ranges 18-19 and 28-43. [ ] Full file not read.
- [x] ANOM metrics.json: fields n_good, n_defect, auroc, fit_on_heldout and anomalib version (grep).
- [x] app/anomaly/static_ir.py: OpenVINOInferencer use (earlier pass).
- [ ] app/dashboard_v5: directory exists; not reviewed (JC-06).
- [x] docs folder: names only (log entry 2); not read.

Sandbox shared files:
- [x] FACTS-LIVE.md: grep for the terms used; lines 1, 4, 11, 19, 20, 25, 27 and 34 read. [ ] Full file not read.
- [x] T09B-MODAL-BROWSER.md: grep for the dates, weights and track terms; lines 14, 40, 41, 50-53, 151, 297, 314, 394, 416, 418 and 422 read. [ ] Full file not read.
- [x] curation/models.tsv: rows 1-7 read.
- [ ] curation/datasets.tsv: values taken from RB-03 evidence; not re-read in this pass.
- [x] Protocol and task files read this pass: P3-HACK-ALIGN.md, LAPTOP.md, RULES.md.

Web:
- [x] developer.robotics.intel.com was checked with WebFetch in an earlier pass; no dates found (REQ OQ-02).

Excluded on purpose:
- [x] Mentor answers file and the friend folders /home/intel/junction-cell and /home/intel/junction-handoff: not opened.
- [x] QR code in the photo: not decoded or opened.

## Self-check (facts re-verified in this pass, with method)

1. Brief slide 3 contains "combines Vision-Language-Action (VLA) capabilities" (text line 19; page map shows the line on page 3). Method: grep -n and awk page split on form feeds.
2. Brief slide 9 holds the must-deliver items and the Challenge-Day line (text line 93; page 9). Method: awk page split; the four item texts were checked by the checker (text quotes BR-24 to BR-27).
3. Brief slide 8 lists one RealSense D415 and one Pro QHD 2K/60fps webcam (page 8). Method: awk page split; BR-32 quote checked as text.
4. Brief slide 11 lists the policies and the Robotics AI Suite and OpenVINO columns (text lines 110-117; pages 11 and 12). Method: grep -n and awk page split.
5. Brief slide 14 names the Robotics AI Suite (text line 179; page 14). Method: grep -n and awk.
6. GUIDE line 136: Intel Tech Integration row, 25 points, names the Robotics AI Suite. Method: grep -n.
7. GUIDE lines 30, 35 and 42 name the ports 3000 and 7860; line 50 names the USB hub. Method: grep -n.
8. GUIDE has no Geti, MuJoCo or plugin text. Method: grep -n -i over the guide.
9. GUIDE Step 6 contains "How you connect them is up to you." Method: read_file over lines 116-140 (log entry 74).
10. GUIDE lines 107 and 111 hold the policy list and the language-task hint. Method: grep -n (log entry 62) and read_file (log entry 72).
11. ARCH line 77 says RULES plates still say device auto (lines 32, 41) and that auto resolves to CPU per D-080. Method: sed -n (log entry 71).
12. ARCH line 104 has no runnable loop entry point; line 109 says power needs root. Method: sed -n (log entry 71).
13. ARCH line 108 has the device-auto conflict. Method: grep -n -w auto (log entry 71).
14. FACTS line 11 lists SmolVLA pilot 6540166d; no FACTS line contains d128dc1a. Method: grep -n on both ids (sandbox).
15. FACTS line 25: MuJoCo 4.0.1 twin, 150 episodes, 73 contradict OD-10, not used for training. Method: grep -n -i (sandbox).
16. FACTS line 34: parked arm at 16:00Z, rescued. Method: grep -n -i (sandbox).
17. MODELS rows 2-6: four ACT and one SmolVLA with openvino backend (row 5 is d128dc1a). Method: awk on the tsv (sandbox).
18. MODAL line 51 weights 40/25/20/15; lines 50 and 422 close Monday 16 November; line 40 event 13-15 November. Method: grep -n (sandbox).
19. Checker: 48 laptop text quotes OK and 5 sandbox text quotes OK; 0 mismatches; 0 quotes over 15 words; 24 image quotes transcribed, not machine-checked. Method: check_align.py on the laptop (log entry 77) and in the sandbox (scratch folder).
20. Counts over 73 rated rows: met 17, partial 49, missing 3, unknown 3, not applicable 1. Method: checker metrics (log entry 77) and a Python count of statuses by kind (sandbox).

Corrections found by this self-check: GD-13 downgraded to partial (ARCH lines 77 and 108); GD-21 and GD-22 added from the guide's Step 6 and Step 5 hint; a checklist pass confirmed guide lines 30, 35, 42, 50, 51 and 52; the BR-28 and GD-16 line citations are correct; the earlier PaDiM benchmark claim is removed (GD-05 partial); the patch run's 19 met included MD-05, so the correct count is 17 after the GD-13 change.

## Requests to Opus

1. Decide OQ-01 with Hassan: which rubric governs, the brief's six criteria or the guide's five. Until then both are tracked.
2. Get the Challenge Day date and the main-challenge submission format from the kickoff sheet (OQ-02).
3. Export the mentor Q&A into the allowed inputs so OQ-03 can close. E1 did not open the mentor file.
4. Replace the PLACEHOLDER model ids (RULES lines 31, 40) before any routing test (BR-02, BR-08).
5. Set the ACT device explicitly in RULES (lines 32, 41) and confirm the device on a live load (GD-13).
6. Settle the routing rule with Hassan: FACTS line 1 and OD-10 (good to blue, defective to green) against the team note in FACTS line 20 (good to green bowl, defective to red bowl).
7. Reconcile the dataset episode counts (FACTS 15:03Z against MODELS) and the task strings (FACTS 14:00Z against RULES lines 32, 41) (RB-03, GD-20, BR-16).
8. Confirm whether SmolVLA 6540166d (FACTS lines 11, 20) and MODELS row 5 (d128dc1a) are the same training run (BG-02, BR-16).
9. Decide on the Robotics AI Suite (RB-02). The brief names it on slides 11 and 14, and the guide scores it at line 136.
10. Schedule any closed-loop, grasp or Run model test under the robot motion rule (BR-22, BR-17, JC-05, GD-22, GD-10). E1 has not commanded the arm.
11. Decide the Modal track entry before the end of Monday 16 November (MD-02).
12. Confirm whether the Pro QHD webcam is in the kit (BR-32).
13. Decide the PaDiM row (GD-05) and how power is measured without root (JC-04, RB-04).
14. Serial ports: FACTS line 1 and line 19 disagree on the ttyACM numbers. Use the by-id paths (LAPTOP.md rule 8). No action for E1.
