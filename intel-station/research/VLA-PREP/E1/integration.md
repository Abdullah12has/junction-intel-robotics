# OD-10 decision to policy: smallest real link (JC-03)

Status: documented, not wired. The loop is unchanged. No policy is connected to motion.

## 1. Inputs

- Inspector result, shared/contracts/LOOP-CONTRACT.md section 3: label GOOD, DEFECTIVE or UNCERTAIN, score, threshold, and reasons such as tape and not_2x2. Until the real inspector exists, builders use FakeInspector (section 3).
- OD-10 rule (docs/PICK-ANYWHERE.md, header): GOOD (2x2, no tape) goes to BLUE; DEFECT (tape or not 2x2) goes to GREEN. The task brief sends UNCERTAIN to GREEN as well. The rule's home is app/loop/rules.yaml (contract section 4).
- via: direct, inspected or fail_safe (contract section 8, line 51).

## 2. The link

One mapping from plate to (model_id, task_text), and one policy start per run.

| plate | option 1: per-plate ACT (D-060) | option 2: one text-conditioned model |
|---|---|---|
| BLUE | 36edc07e-a80c-4544-916a-48a692c01190, task "move the red brick to the blue plate" | d128dc1a-cb90-4c99-a931-d6c0f6eb9ada, same task |
| GREEN | 4d43afc8-6841-4fd4-9928-3b5be5180eac, task "move the red brick to the green plate" | d128dc1a-cb90-4c99-a931-d6c0f6eb9ada, same task |

D-060 (ledger/DECISIONS.md:92) allows either form: one text-conditioned model, or per-plate ACT switched with load_model. Opus picks. Option 1 is the one with a per-plate model for each side.

The call path in the loop: the orchestrator loads the environment (app/loop/orchestrator.py:615 and :959, its own draft with tele_operator none at :157). The policy then goes through the client: load_model(model_id, "openvino", device="GPU") (studio_runtime_client.py :333-344; the default device is "auto", which resolves to CPU, D-080), then start_policy, which sets the model control with task, hz 30, sync, request_threshold 0.5 and grace_s 3.0 (client :197-225 and :351). Switching the model loads the next one through start_policy (contract section 1).

Gate: the offline dry run (work/VLA-PREP/E1/dryrun.json) found the only laptop-available export (389da0fd, not one of the two candidates above) not sane. Neither candidate has an offline check yet. Do not connect the mapping to motion until a candidate passes an offline check on the team's episodes and the live test in procedure.md.

## 3. What gets logged

Contract section 8 (lines 51-52) and section 9 (lines 56-60):

- decision: {target_id, plate BLUE or GREEN, rule 'OD-10', via direct, inspected or fail_safe}. Add the inspector label, score and reasons that produced it.
- policy_run: {phase, model_id, task_text, plate, t}. Phases per the contract: start, grasp, carry, release, return, done, timeout, error.
- latency: section 9 says the dashboard reads latency_ms from policy_run or inspection events. Add latency_ms to policy_run. The runner writes per-tick timing only to latency.csv.

Mismatches to fix before JC-03:

1. Phase enum. The contract lists carry (line 52). The runner's Episode.step emits start, grasp, release, return, done and timeout, with no carry (work/P2-B-OVRUN/E1/code/runner.py:116-137). Its phases come from the gripper and home distance, not from the plate.
2. latency_ms is missing from events (above).
3. verify is listed in section 8 but the runner does not emit it.

## 4. Acceptance check (no robot, no Studio)

1. Feed a recorded inspector label through the mapping. Check that BLUE selects the chosen BLUE model and task, and GREEN selects the chosen GREEN model and task.
2. One events.jsonl run ends with a decision event, then policy_run start, then one terminal policy_run (done, timeout or error).

Context for the criterion: the ledger notes that the hold-state comparison is strict for these policies (ledger/DECISIONS.md:116, D-083).
