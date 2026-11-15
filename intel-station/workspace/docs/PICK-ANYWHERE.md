# Pick-anywhere sorting loop (`app/tools/sort_any.py`)

Opus, 2026-11-14 (I-59/I-60). SO-101 DUPLO sorter, OD-10 rule: **GOOD** = 2x2 with no tape → BLUE plate;
**DEFECT** = tape or not 2x2 → GREEN plate. Bricks can be anywhere the arm reaches on the mat. Only the loop
moves the follower; it goes through the motion server (`app/tools/motion_server.py`).

## Run, stop, pause

**After the robot was lent out:** `app/tools/resume.sh` brings the stack up in a safe order and refuses if
anything else holds the follower port or the cameras. Order: checks, recorders, fresh live frames, hand watchdog,
motion server (torque on at the present pose, no motion). `app/tools/resume.sh --go <run name>` also starts the
loop. Hand back with `mseq.py ../motion/gm/servo.yaml spot_up_home_closed`, then `mcmd.py '{"op":"release"}'`,
then kill `hand_watch` / `monitor_cams` / `wrist_recorder` by their pid files.

Manual start of the loop alone:

```
cd app/tools
nohup ../../envs/data-env/bin/python -u -B sort_any.py --run <name> --n 150 --start <k> >> ../../logs/sort_any_<name>.out 2>&1 &
touch run/sort_stop     # stop before the next brick (removed automatically)
touch run/sort_pause    # wait at the scan pose until the file is deleted (also set by a failed jaw check)
```

Needed processes (each has a pid file in `run/`; restart only by pid): motion server (`run/mserver/pid`),
hand watchdog (`run/hand_watch.pid`), monitor cameras (`run/monitor_cams.pid`), wrist recorder
(`run/wrist_recorder.pid`). Start-up runs a check-only jaw check and carries over a brick still in the gripper.

## One cycle

1. **Scan** from `q_scan` (wrist camera 70 mm above the hover height). Bricks are colour-class blobs (red,
   orange, yellow, green, blue, white). Touching bricks of different colours stay separate. Pieces of one
   brick split by shadow lines or the jaw mask are bridged (15 px).
2. **Candidates** must pass all of these filters:
   - not inside a plate (`plate_frac` < 0.3, plate-coloured rims < 0.15);
   - ≥ 35 mm from a plate centre;
   - 115–335 mm from the pan axis (vertical-jaw reach);
   - not failed in the last 5 minutes.

   Whole bricks go before ones cut by the image border, then nearest first.
3. **Scan px → base xy:** scan Jacobian × learned correction. The correction is refitted every cycle from
   `app/tools/scan_pairs.jsonl`. Each line is a (scan px, servo-verified hover xy) pair, appended after every
   centred approach.
   - Under 15 pairs it uses a similarity fit (scale + rotation) with outlier rejection.
   - From 15 pairs it uses a RANSAC homography.
   - The raw Jacobian alone was 34–48 mm off for bricks more than 150 px from the scan centre.
4. **Approach:**
   1. Descend to the hover height at the scan location.
   2. Move in differential-IK steps of ≤ 60 mm at the hover height.
   3. Wrist servo, damped gain 0.65, predicted-position tracking. It is **locked to the scan brick's colour
      class**, falling back to a compatible class (cream reads yellow from the scan and white at the hover).
   4. Yaw-align with wrist roll.
   5. Re-acquire the same brick (colour + size) and servo again.

   If the brick is not found at the hover, it takes a second look from 30 mm higher.
5. **Grasp:**
   1. Height from brick size: `tip = 9886/short_px − 33`, used when within 15 mm of FK.
   2. Stage to 26 mm, then the final descent to 8 mm above the mat. This keeps the jaws off the mat; contact
      loosens the fixed jaw.
   3. Close, then lift 45 mm.

   **Held = gripper reading ≥ 1.1.** The empty close reads 0.59; a held 2x2 or 2x4 reads 1.57–2.42. Brick size
   in the wrist view is not used: a held brick is cut off at the image edge. On a miss it re-centres at the
   hover and retries once with the larger height estimate and a 6 mm final height.
6. **Shape:** aspect > 1.4 or a narrow short side → `not_2x2` → GREEN with no inspection.
7. **Tape** (2x2 only):
   - The close camera checks for tape before the grasp (DET2 F2 rule). The view is located from the jaw
     tips, and the brick's hue must match.
   - If that check is not confident, the brick goes through a **station inspection**: put it down at the spot,
     retract, take a clean close view, then re-pick.
   - Views of the brick held in the gripper are not trusted: the cable, jaws and background give false tape.
8. **Place** by the short plate path, then return to the scan pose. Transit speed 42, accel 120.
9. **Periodic jaw check** every 5 cycles at the scan pose:
   - Fixed jaw moved > 18 px → pause with a message.
   - Small consistent drift of the jaw gap → the servo target follows it.

Typical clean cycle: about 65–70 s (pick 30–48 s, place 20 s).

## Recorded data (for VLA training)

- One folder per episode: `logs/sort/<run>/ep_NNNN/`.
  - `episode.json`: decisions, blobs, heights, grasp readings, timings, failure reason, `label_source`.
  - `joints.csv`: 30 Hz follower joints from the motion server.
  - `crops/`: hover, pre-grasp close and station views.
  - Snapshots from all three cameras at every phase.
- One line per cycle in `logs/sort/<run>/cycles.jsonl`.
- Continuous video:
  - monitor cameras: `logs/monitor` MKVs, 30 fps, absolute timestamps;
  - wrist camera: `data/wrist/raw/w_<UTC>.mkv`, 60 s MJPEG segments, absolute timestamps.
  - Cut episodes by the `t_start` / `t_end` in `episode.json`.
- Labels corrected by visual review are marked `label_source: opus_visual`.

## Calibration files (`app/tools`)

| file | what |
|---|---|
| `scan_calib.json` | `q_scan`, `q_hover`, `target_scan` (379.3, 202.8), brick size at scan/hover |
| `wrist_calib.json` | wrist Jacobian `M_base` (px/mm) from blob tracking, at psi 113.23° |
| `jaw_calib.json` | wrist-servo target px (jaw gap centre + 3, y 268) |
| `wrist_jaw_mask.png` | static jaw mask in the wrist view (camera rides on the gripper) |
| `scan_pairs.jsonl` | learned scan-map pairs (grows every cycle) |
| `kin.py` | URDF FK, differential IK `ik_rel`, `MAT_Z` −38.3 mm |

## Safety

- **Hand detector**: `app/tools/handdet.py`, one module shared by the watchdog (`hand_watch.py`) and the loop's
  clear check (`sort_loop.hands_seen`), so the two can no longer disagree. They did at 18:03Z, and the loop burned
  40 recovery retries in 1 s.
  - The skin colour rule is unchanged.
  - Close view: per-pixel value ≥ 130 (removes warm speckles on the mat).
  - While the gripper is closed, whole skin blobs that reach the gripper zone are ignored (a carried white brick).
  - A blob is a hand if it is large (top ≥ 900 px, close ≥ 6000, wrist ≥ 20000), or if it is close ≥ 2000 px and
    touching the frame edge.
  - The wrist view is used only with the gripper open and the jaws > 90 mm above the mat.
  - Persistence before a stop: top 1 frame, close 2, wrist 2.
  - Top view region: x ≥ 175, y 0–345. People reach in over the far edge next to the arm. A version that cut the
    top 30 px (18:31–20:02Z, never run live) missed 6 of 17 labelled top-view hands and 25 hands in the 17:03Z
    recording; restored to y ≥ 0 with no extra false stop in 2480 recorded frames.
  - Labelled check on the 41 e-stop images (16:19–18:13Z; labels in `app/tools/tests/fixtures/hand_labels.csv`,
    re-checked 20:00Z against an independent labelling and by eye): all 17 top/close hands are caught and no
    top/close false stop remains. A hand on the laptop off the mat is labelled `hand_outside` and not counted.
    The old rules missed a real close-view hand (2993 px) and stopped on mat speckles.
  - On 4025 recorded close frames: positives 34 → 8. The 28 removed were all cream-brick false stops, reviewed
    by eye; 1 extra real hand is caught.
- The motion server holds the arm on an e-stop and keeps squeezing a held brick.
- The loop resumes by itself 3–5 s after hands leave. A failed recovery now backs off 1, 2, 4 … 30 s, about
  15 min in total before it gives up.
- The follower is used only through `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00`. Torque off
  at hand-back.

## Failure log → fix (17:54–18:23Z)

| failure | fix |
|---|---|
| "nothing held" with a brick in the jaws (cut off at the image edge) | gripper-only held check (≥ 1.1) |
| pink and cream bricks touching read as one blob | colour-class blobs |
| servo switched to the neighbouring brick after the yaw | colour-class lock + size check on re-acquire |
| cream brick half under the jaw mask "not visible" (area < 2500) | min area 1500, raised second look |
| servo swung between two pieces of one white brick | 15 px bridge + predicted-position tracking |
| scan estimate 35–50 mm off for far bricks | learned scan map (similarity → homography) |
| white brick in the gripper tripped the wrist hand check 40× | wrist view only when open and high |
| cream brick tripped the close hand check (3.7k px) | close ≥ 5000 px on 2 frames |
| leaning upside-down brick: jaws closed too high | retry once lower |

## Offline work while the robot was with the team (18:26–19:00Z, I-61)

- **Robot handed back at 18:26Z.**
  - Arm at `home_closed`, torque off.
  - Stopped: loop, motion server, hand watchdog, recorders. No holder on `/dev/ttyACM0` or any `/dev/video*`.
- **Hand detector:** `handdet.py`, see Safety. `hand_watch.py` is rewritten on it, with a `main()` and a docstring
  run line.
- **Recovery back-off** in `sort_any.recover()`.
- **`decide()`**: when no tape view is usable, the decision is now marked `no_evidence`. `episode.json` gets
  `needs_review: true` whenever the decision is not confident.
  - The VLA builder and the anomaly training exclude those episodes.
  - Two such episodes (any_1731/ep_0003, any_1740/ep_0005) had held views showing a *different* taped brick on
    the table. Their GOOD labels are still correct for the brick that was moved, but the crops are polluted.
- **Dead code removed** from `sort_any.py`: `inspect_pose`, `held_verdict`, `multiview`, `CLOSE_HELD_PX`,
  `place_long` (backups in `run/bak_*`). Static checks (`work/lint/undef.py`, `work/lint/attrs.py`; stdlib AST,
  no downloads) find no undefined names and no missing cross-module attributes in the loop modules.
- **Offline smoke test:** `work/lint/smoke_offline.py` covers the hand labels, `hands_seen`, the `decide`
  branches and `candidates` on a recorded scan. 0 failures.
- **Eye-in-hand wrist camera model:** `app/eyeinhand/` (offline tool, README).
  - Fitted from the loop's own frames and joints: in-sample 4 px. The model's projection lands on the brick the
    loop chose in every scan frame (`work/eyeinhand/scan_check.jpg`).
  - On the runtime path the leave-one-episode-out error is 11.5 mm median, against 13.8 mm for the learned
    similarity map, so it is **not switched in** yet.
  - Criteria: refit at ≥ 20 episodes; use it when the median is ≤ 6 mm.
- **Data pipelines**, run by background builders:
  - VLA dataset build of any_1740 (success-only, wrist clock check).
  - Multi-angle brick crops from the wrist/close videos for the anomaly model. With 5 GOOD / 2 DEFECT episodes
    (both DEFECT yellow) no anomaly model can be validated yet; the 150 runs are the bottleneck.
  - Evidence report on every episode.
  - Replay regression tests.

## Known gaps

- Upside-down or leaning bricks can still be missed; they are retried once, then skipped for 5 minutes.
- Same-colour bricks touching each other merge into one blob and read as `not_2x2`.
- Bricks inside or against a plate are ignored. Bricks the vertical-jaw grasp cannot reach are ignored too:
  `sort_any.grasp_reachable` replays the approach and descent IK without moving (reach ends at about 296 mm from
  the pan axis, `work/opus-verify/reach_map.log`). It replaced a fixed 335 mm radius that let 3 of the 6 approach
  failures of runs any_1731/1740 through.
- Tape seen only from the side relies on the pre-grasp close view or the station view.
- The loop cannot un-sort: someone must put bricks back on the mat to keep recording.

## Known perception defects

- **PD-1** (servo frame 18:00:02Z, `tests/fixtures/servo/servo_180002_red.png`): the 15 px bridge that keeps a
  split white brick in one blob also joins a 339 px red speck 78 px from a red brick; the box centre moves 19 px
  and the short side grows 10 %. Dropping pieces under 8 % of the largest fixes this frame but cuts a real piece of
  a fragmented cream brick (servo frame 18:20:19Z: short side 114 → 83 px), so it was not adopted. Needs a brick
  segmentation measured on labelled frames. The replay test reports PD-1 without failing on it.
