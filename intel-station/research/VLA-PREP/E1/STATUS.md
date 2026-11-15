PARTIAL 2026-11-14T215800Z: deliverables written; offline dry run done (389da0fd NOT SANE on any_1740/ep_0015 on GPU, NPU and CPU; GPU p50 11.1 ms). Live Studio "Run model" test not run: robot not touched, API on 7860 down, follower entry points at the leader device, saved environment carries the leader.

Files (sandbox /home/claude/kb/work/VLA-PREP/E1/ and laptop /home/intel/intel-junction-2026/work/VLA-PREP/E1/):
- models.tsv: five model rows with job ids, epochs or steps, dataset, task strings and export status (both places).
- procedure.md: live Studio test; blockers, stop order, payloads, safety, evidence, restore; source lines cited (both places).
- integration.md: OD-10 decision to model and task mapping; logged fields; contract mismatches; acceptance check (both places).
- dryrun_policy.py: offline OpenVINO dry run (both places; sha256 9fbab2de81996352dae3998bf2b55bdd0458392bb063931e2e89d646ca30f503).
- dryrun.json: laptop only (sha256 4d476ce97a5b7891df466334688e9bfabcbd006a7a65087692114896cd361784, 684 lines).
- laptop-log.md: every laptop command and its output (sandbox only).
- STATUS.md: this file (both places).

Dry run (dryrun.json; the second attempt, under envs/ovrun-env):
- Model 389da0fd (S1 ACT curated v0, the only export on the laptop). Task text "move the brick to the inspection spot" (the only task of dataset a4ba6ede).
- Episode any_1740/ep_0015: 1867 joint rows over 67.5 s. 30 query points. Frames from the top and close monitor MKVs (recording started 17:03:51Z), one ffmpeg seek per frame, 30 distinct frames per view.
- Latency (load s; first call ms; p50 and p95 ms over the 29 later calls): GPU 2.30; 21.8; 11.1 and 12.8. NPU 0.18; 71.9; 16.7 and 17.8. CPU 1.19; 231.1; 181.8 and 193.7. The sort loop (run any_2127) was active at 21:51Z, just before the run; its state during the run was not checked.
- Sanity: outputs finite and in range, chunk shape (100, 6) on all three devices. Policy error against recorded goals at horizon 0 is 13.4 units (mean over joints; GPU per joint 16.5, 16.8, 18.5, 7.1, 11.0, 10.3). Hold-state error is 0.90. The policy does not beat hold at horizons 0, 10, 25, 50 or 99. sane = false on all three devices.
- Parity against CPU: GPU max 0.196, mean 0.019; NPU max 0.409, mean 0.029 (normalized units).
- Reading: the device path, frame and state inputs, and load all work. On this episode the policy's commands are far from the recorded ones. 389da0fd is the v0 2k-step smoke model, and the earlier replay (P2-B-OVRUN) showed a joint 1 and 2 gap of about 11 units on the Studio dataset. Not tested: a camera-role swap (top and close), a signed-error analysis, and more episodes.
- First attempt (envs/anomalib-env, jsonargparse 4.46.0) failed on all devices with "'ArgumentParser' object has no attribute 'instantiate'". The second attempt (envs/ovrun-env, jsonargparse 4.53.0) ran clean. Details are in laptop-log.md.

Open items for Opus, in order:
1. Studio API on 127.0.0.1:7860 is not listening (21:50Z). The UI on :3000 is.
2. Follower entry: shared/studio-api/project_robots.json (11:39Z) has SO101 Follower on /dev/ttyACM1 and SO101 Leader on /dev/ttyACM2. At 21:41Z /dev/ttyACM1 is the leader's by-id device. Do not start the test until the entry is fixed (Opus decides).
3. The saved environment names the leader (d361855b). Use the loop's leader-free draft (app/loop/orchestrator.py:153-157) or approve the UI route (procedure.md section 4).
4. Holders to release first: motion server (pid 3522921), monitor_cams (3521843), hand_watch (3522883), wrist_recorder (3521845). Procedure section 3.
5. Build mismatch: running build 17714488 (dirty) against clone 285924d9. Verify each route in the running backend before use.
6. Only 389da0fd is on the laptop. f06a253e and the other candidates are not. The dry run does not check PP-BLUE v1 or v2.
7. The contract's phase list includes carry, which the runner does not emit. latency_ms and verify are also missing from the events (integration.md section 3).
8. The sort loop (run any_2127) was active at 21:51Z. run/sort_any.pid names pid 399110, which was not alive at 21:48Z. E1 did not touch it. Check before stopping anything.

Not done by rule or by design: no live Studio run, no Studio API or UI calls, no robot, camera or serial access, no writes to the Studio data folder, no R/run.

Notes: "(N ep)" in model names means epochs, not episodes. Memory was not read (P3-COMMON HR list).
