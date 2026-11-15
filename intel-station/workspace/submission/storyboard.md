# Storyboard: 2 to 3 minute demo video (plan)

Status at 2026-11-14 15:11Z, revised after the 15:25Z decision corrections (D-070, D-080): plan only. Nothing in this storyboard has been filmed, and no take has run.

Rules for this video:
- Every number on screen comes from the take's own files: the run folder's events.jsonl (path open, README section 8, item 4), R/logs/telemetry/YYYYMMDD.jsonl, or the benchmark JSON named in evidence.md. Run B is R/work/P2-B-OPT/E1/bench_runB_quiet.json. Do not use bench.json while a later run is writing it. If a number is not in one of those files, do not show it.
- If a shot cannot be captured live, use a replay and label it REPLAY on screen. The take plan names this fallback [PLAN section 2, Take row].
- A shot marked scripted mode only is used only if the run is in scripted mode. The mode is chosen at gate G4-S1 [DEC OD-11].
- No hype words. Narration states what the frame shows.
- The policy device is GPU by rule (D-080). Show the device that the run's own files record, never 'auto'.
- The policy is a smoke model (389da0fd). Its replay executed-action error (4.978) is above the hold-state error (1.556) in-sample [RPL replay_summary.json]. Narrate a policy success only when the take's verify event says ok.

Source keys as in README.md. Times are the planned run length in seconds.

| Shot | Time (s) | What is on screen | Rubric line it supports | Source for on-screen values | Fallback |
|---|---|---|---|---|---|
| S01 Title | 5 | Project name and the sentence "Inspect each brick, then sort it." | none | none | none |
| S02 Scene wide | 10 | Black mat, BLUE plate on the left, GREEN plate on the right, cluster of DUPLO-size bricks, arm at home | Task Success (clutter) | none | photo of the same scene, labelled |
| S03 Two views | 10 | Top RealSense and close RealSense live frames side by side | Creativity (two-view inspection); Task Success (perception) | live frames; no number | none |
| S04 Dashboard idle | 8 | Operator page in WAIT_BRICK, device names (CPU, GPU, NPU), telemetry strip (package temperature, NPU busy) | System Efficiency (telemetry is on); Intel Tech (device names) | TEL line (temp_c.pkg, npu.busy) from the take | none; the dashboard is not on disk yet |
| S05 Scan and triage | 15 | Per-brick boxes from SCAN; one brick marked CONFIDENT_DEFECT and sent straight to GREEN; others marked for inspection | Creativity (triage, I-52); Task Success (routing) | triage and scan events in the run folder events.jsonl [LC section 8] | replay, labelled |
| S06 Inspect | 20 | 1280x720 still of the chosen brick from both views; score bars against the threshold; verdict with reasons (tape or not 2x2); device and latency in ms | Intel Tech (OpenVINO on a named device); System Efficiency (latency per inspection); Task Success (inspection) | inspection event: score, threshold, reasons, device, latency_ms [LC section 8] | replay, labelled |
| S07 Policy move | 25 | Slow motion of the policy moving the brick to the plate; policy name, backend and device on screen | Dataset and Pipeline (Studio-trained policy); Intel Tech (OpenVINO backend); Task Success | policy_run events (phase, model_id, task_text, plate) [LC section 8] | replay, labelled |
| S08 Verify | 10 | Top-view frame at release: brick on the plate; VERIFY result on screen | Task Success (correct sorts) | verify event (plate_intended, release_side, ok) [LC section 8] | none |
| S09 Handover (scripted mode only) | 10 | Dashboard shows HANDOVER_OUT, port free, HANDOVER_IN with model kept | Creativity (model kept across a sort); System Efficiency (handover time) | handover events; PROBE pattern [PROBE] | none |
| S10 Scripted sort (scripted mode only) | 20 | Plan S moves to the BLUE or GREEN drop pose at the capped speed; torque state on screen | Task Success (scripted sort); safety | sort_start and sort_end events [LC section 7] | replay, labelled |
| S11 Counts and timing | 10 | Cycles, good, defective, uncertain, mean cycle time | Task Success (accuracy, cycle time); System Efficiency (cycle time) | stats event (cycles, good, defective, uncertain, mean_cycle_s) [LC section 5] | none |
| S12 Stop and close | 10 | STOP pressed; hold, then unload; torque off; telemetry stopped | Safety; System Efficiency (power and temperature summary) | control event, run_end event, telemetry line | none |
| S13 Summary card | 5 | One card with the measured values from evidence.md; every missing value shown as TBD | all five lines | evidence.md rows, filled from the take | none |

Total planned run time: 158 s (2 min 38 s).

## Narration lines

Write each line as a plain statement. Replace each TBD with a value from the take's files, or delete the line.

- S01: "This is a sorting cell built on Intel Core Ultra hardware. It inspects each brick, then sorts it."
- S02: "A cluster of Duplo-size bricks sits on a black mat. Good bricks go to the blue plate on the left. Defective bricks go to the green plate on the right."
- S03: "Two RealSense cameras look at the brick: one from the top, one from low and close. The wrist camera is recorded separately."
- S04: "The dashboard shows the state, the device for each step, and the package temperature from the telemetry log."
- S05: "The cell scans every brick. A brick it is sure is defective goes straight to the green plate. The others get a closer inspection."
- S06: "This is the inspection of one brick, from both views. The detector scores the brick against its threshold. Here the result is [TBD: verdict and reason from the inspection event]. The model ran on the [TBD: device] at [TBD: latency_ms] milliseconds per inspection."
- S07: "The policy, trained in Physical AI Studio, moves the brick to the [TBD: plate from the policy_run event]. It runs through the OpenVINO backend on the [TBD: device from the run's own files]."
- S08: "The top camera confirms where the brick landed. The check reads [TBD: ok or not ok from the verify event]."
- S09: "For scripted sorting, the cell hands the arm to our motion code, then gives it back. The model stays loaded across the handover, and the handover took [TBD: seconds from the handover event]."
- S10: "Scripted sorting moves at a capped speed. The arm stops and holds if anything is wrong."
- S11: "This run sorted [TBD: good] good, [TBD: defective] defective and [TBD: uncertain] uncertain bricks. The mean cycle time was [TBD: mean_cycle_s] seconds."
- S12: "STOP holds the arm and unloads the model. Torque is then switched off. The telemetry sampler is stopped after the run."
- S13: "Every number in this video comes from the run's own logs. Values we did not measure are marked TBD in the evidence file."

## Shot-to-rubric check

| Rubric line | Shots |
|---|---|
| Task Success Rate (25) | S02, S05, S06, S07, S08, S10, S11, S13 |
| Intel Tech Integration (25) | S04, S06, S07 |
| Dataset and Pipeline Quality (20) | S07 (shows the policy trained in Studio); the numbers are in evidence.md |
| System Efficiency (20) | S04, S06, S09, S11, S12 |
| Creativity (10) | S03, S05, S09 |

## Coverage checklist (internal, for review; remove before publishing)

- [x] architecture.md diagrams (policy and scripted modes) used for shots S05 to S10.
- [x] evidence.md rubric tables used for the values in S13.
- [x] LC section 5 events (stats, control, run_end) and section 8 events (scan, triage, inspection, policy_run, verify) named as sources.
- [x] LC section 7 events (handover, sort_start, sort_end) named as sources for S09 and S10. The event names are confirmed in events.py (this run); the field contents are from the contract text.
- [x] PLAN section 2 Take row and PLAN section 6 fallbacks (REPLAY label) used.
- [x] PROBE log pattern used for S09.
- [x] Benchmark source: run B JSON named; bench.json excluded because a later run is rewriting it (15:10Z).
- [x] Correction pass (15:25Z to 15:31Z): DEC D-080 and the RPL replay summary used for the device and no-success rules.
- [x] TEL field names temp_c.pkg and npu.busy confirmed in the last telemetry line (15:08Z).
- [ ] Dashboard (P2-B-DASH): not on disk; S04 and the on-screen layout are plan only.
- [ ] Recording tool and output file: not named in the inputs. TBD.
