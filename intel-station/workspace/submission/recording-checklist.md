# Recording checklist: before and during the take

Status at 2026-11-14 15:11Z, revised after the 15:25Z decision corrections (D-070, D-080). Times are UTC unless marked EET (UTC+2). Source keys as in README.md. Every step below is a plan step taken from the cited source; none has been run.

## A. Clock and go/no-go

| Time (UTC) | Time (EET) | Step | Source |
|---|---|---|---|
| 19:00Z | 21:00 | G-SC2: 16 trials, at least 11 sorted (or report as measured) | PLAN section 2 |
| 20:00Z | 22:00 | Freeze: no code or model changes after this time | PLAN section 1 (hard clocks) |
| 21:00Z | 23:00 | Dry run 1 | PLAN section 3 |
| 21:20Z | 23:20 | Dry run 2 | PLAN section 3 |
| 21:50-21:58Z | 23:50-23:58 | Take window | PLAN section 2 (Take row) |
| 22:00Z | 00:00 | Demo saved (deadline) | DEC D-024; PLAN (hard clocks) |

Go/no-go before the take (all must hold):
1. Gate G-A and gate G-M recorded as PASS (done at 14:13Z and 14:33Z) [PROBE; DEC D-063].
2. The open driver-log question is answered: both G-M connect lines label the by-id follower port role=leader, while FL 13:42Z maps serial 5B3D040939 to the follower [GMB line 2; GMG line 2; FL 13:42Z]. Confirm which port is the follower before any motion. This is a gating item.
3. The episode set for the policy is final: the v2 CSV exists, or the take uses v1 [CUR; DEC D-067]. This is a gating item.
4. The loop entry point and the dashboard exist on disk and pass a dry run. At 15:08Z tools/run_policy.py is absent, policy_run.py has no __main__ entry, and app/dashboard is absent [README section 8, items 3 and 4]. This is a gating item.
5. The arm state reads FREE. At 15:08:09Z it read "BUSY 1436289", a Studio backend publisher worker [ARM; README section 3.1]. This is a gating item for any motion.
6. The live gate inputs in RULES are set: both plate model ids (not PLACEHOLDER), det2.module_dir, and still.exposure. At the 15:03Z save all four were unset [RULES; PLIVE]. This is a gating item.
7. Every number planned for the on-screen summary has a source file that is written by the take [storyboard.md].
8. The policy device is GPU in the rules file: both plate blocks (RULES lines 32 and 41) must read device GPU, not auto (D-080). At 15:27Z both read auto. This is a gating item for the policy run.

## B. Scene

- Mat: black mat [FL 11:30Z].
- Bricks: DUPLO size. The 2x2 is 31.8 mm [FL 13:45Z; P2-B-DET2 facts].
- Good bricks: clean 2x2. A pink or red 2x2 with a printed "2" is good; print is not tape [P2-B-DET2 facts].
- Defective bricks, both kinds: a 2x2 with tape, and a brick that is not 2x2 (for example a 2x4) [DEC OD-10].
- Tape: dark grey gaffer-type tape in large patches on the side faces [DEC D-068]. Open item: DEC D-059 describes clear or brownish tape on a yellow 2x2. Confirm on the bench before the take.
- Count: the cluster has several bricks, placed between the plates [DEC OD-8a, 13:08Z]. Set the defect count in the cluster before dry run 1. If none is in the cluster, the team places one [PLAN R3]. If the take still has none, the take states the defect path as not measured.
- Plates: BLUE on the left and GREEN on the right of the top view [FL 13:52Z]. The robot base is at the top centre of that view [FL 13:52Z]. Proposed: do not move the plates after dry run 2.
- Shifted and clutter scenes (for the 16-trial run): team set-up, or normal-only counts [PLAN C-7; PLAN R10].
- Lighting: not specified in the inputs. The RealSense auto exposure varies strongly, and some frames turn blue [FL 13:30Z]. Lock the exposure or confirm the colour normalisation before the take [P2-B-WRIST fixer item 1]. RULES still.exposure is null (refused for live). Status: TBD.

## C. Cameras and ports

- Two RealSense views: dataset key `realsense 243622061421` is the top view and `realsense 243722064316` is the close low view [FL 13:23Z; RULES].
- Studio camera IDs: the RULES comments map 1d80a0dd-8f62-4f23-ae29-2f1fe406dde9 to the close low view and b6e5da09-28f2-4208-9d15-526580074c75 to the top view [RULES studio block; LC section 1]. These comments are not checked against the live Studio camera list. Confirm before the take.
- Wrist camera (USB2.0_CAM1, /dev/video16): owned by the recorder until it is stopped. Stop the recorder by pid before any loop run: kill $(cat /home/intel/intel-junction-2026/run/wrist_recorder.pid) [wrist_rec.sh header; PLAN R12]. The loop does not check this pid (checked 15:08Z), so the stop is a manual step.
- Follower: only the by-id path /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00. Never a ttyACM number [FL 13:42Z; RULES paths block].
- Leader: the loop does not load the leader [LC section 1].
- Arm state: R/run/arm_state must read FREE before the loop starts [FL 13:55Z; ARM]. At 15:08:09Z it read "BUSY 1436289".
- Team sessions: check the Studio state before each arm window; move the window if a team session overlaps [PLAN R6].
- Presence: the presence flag R/run/HASSAN_PRESENT must exist for a live run [PLIVE; PRUN; LC section 4; DEC D-052].
- Hands and people: camera check for hands and people before each motion run [INS I-44]. A physical cut-off within reach at every arm window [PLAN R11].

## D. Software start order

1. Studio backend (127.0.0.1:7860) and UI (127.0.0.1:3000) are up [FL 11:40Z].
2. Start the telemetry sampler: /home/intel/intel-junction-2026/app/telemetry/telemetry_ctl.sh start. Check it with the status argument [TEL].
3. Stop the wrist recorder by pid (section C).
4. Check the arm state reads FREE [FL 13:55Z; ARM].
5. Start the dashboard: spec only, not on disk. Command in README section 3.2 [P2-B-DASH]. Open http://127.0.0.1:8090 in Chrome full screen at 1920x1080 (or 1366x768) [P2-B-DASH].
6. Dry run: the command set is in policy_run.py (dry SCENARIO LOGS_ROOT). The runner tools/run_policy.py is not on disk, so the dry-run command is TBD (README section 3.2).
7. Live run only after the typed GO, with the presence flag present [PRUN; LC section 4].

## E. During the take

Roles, from PLAN section 5: presenter (narration and slides only), run operator (starts, watches and stops the loop), brick reset team (friends, set-up only).

- Watch the dashboard for HOLD. A HOLD is a stop, not a retry (LC section 4).
- If a brick is held and an error occurs, the torque is held and the run stops. Never drop torque mid-air [LC section 7].
- STOP writes the operator stop command; the arm holds and the model unloads [LC section 4]. Torque off at the end [LAPTOP.md robot motion rule 5].
- Record the take: recording file path TBD (no recording tool named in the inputs).

## F. Fallbacks (PLAN section 6 and take row)

| Trigger | Fallback | Label on stage |
|---|---|---|
| Team S1 model unusable at gate G4-S1 | Local ACT on at-spot and home episodes (F-S1a); if that also fails, a scripted pick from a fixed pose (F-S1b, a COORD decision only) | "scripted pick" |
| Handover reload fails or the port is not free | HOLD and stop. Never sort on a held port [PLAN R4] | none needed |
| A plate under 4 of 5 at gate G4-SORT | Fix that plate's pose (rules only, 21:00-22:00 EET window, PLAN section 3), rerun 5 cycles (F-SORT). If still failing, the team operator teleoperates through Studio (F-T) | "operator-assisted placement" |
| Detector error | UNCERTAIN goes to GREEN and is logged. No retune after WP-21 [PLAN R9] | none needed |
| Wrist camera path changes after a re-plug | Re-point with one verified Studio API call, before WP-10 [PLAN R8; DEC D-037] | none |
| Take cannot reach at least 3 correct sorts (SC1) | Labelled replay of the best take [PLAN section 2, Take row] | "REPLAY" |
| Shifted and clutter scenes not set up | Report normal-only counts [PLAN R10] | "normal scenes only" |
| Studio runtime path fails during a policy run | Fallback runner app/ovrun/runner.py, device GPU set explicitly. Its timeout holds torque, stops the policy and returns home by script [DEC D-080] | "runner fallback" |

## G. After the take

Copy these to the submission folder and update evidence.md with their values:
- The run folder's events.jsonl, frames and live.jpg (run artefacts; LC section 5; folder path open, README section 8 item 4).
- R/logs/telemetry/YYYYMMDD.jsonl for the take window. Stop the sampler with telemetry_ctl.sh stop.
- The final benchmark JSON named in evidence.md. Run B is R/work/P2-B-OPT/E1/bench_runB_quiet.json. bench.json is being rewritten by run_final and is not final until that run has an exit line.
- R/data/curated/episodes_v1.csv (or the v2 CSV when it exists).
- Recording file: path TBD.
- Restart the wrist recorder only if the team needs it [wrist_rec.sh].

## Coverage checklist (internal, for review; remove before publishing)

- [x] PLAN section 1 (hard clocks), section 2 (gates and Take row), section 3 (timeline), section 5 (actors), section 6 (R1-R12), section 7 (C-3, C-7) (earlier run).
- [x] DEC D-024, D-037, D-052, D-059, D-063, D-067, D-068, OD-8a, OD-10 (earlier run); D-063 and D-067 re-read at 15:10Z.
- [x] INS I-44 (earlier run).
- [x] LC sections 1, 4, 5, 7 (earlier run).
- [x] FL 11:30Z, 11:40Z, 13:08Z (via DEC), 13:23Z, 13:30Z, 13:42Z, 13:45Z, 13:52Z, 13:55Z (earlier run).
- [x] GMB line 2 and GMG line 2 (role=leader) re-read at 15:10Z.
- [x] RULES paths, studio and plates blocks; PLIVE refusal checks; PRUN command set; ARM; ps (this run).
- [x] wrist_rec.sh header; telemetry_ctl.sh usage line (earlier run).
- [x] P2-B-DASH (dashboard command, port, resolutions); P2-B-WRIST fixer item 1; P2-B-DET2 facts (earlier run).
- [ ] Recording tool and output path: not in the inputs.
- [ ] Lighting specification: not in the inputs.
- [ ] Dry-run definition (live motion or not): not stated in the inputs.
- [ ] Runner path for dry runs: tools/run_policy.py absent (checked 15:08Z).
- [x] Correction pass (15:25Z to 15:31Z): RULES lines 24-44 (device lines 32 and 41); DEC D-080 read in full.
