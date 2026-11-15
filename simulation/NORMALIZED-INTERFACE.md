# Station-normalized coordinates

Updated 2026-11-14. Replaces the measured-fit design, which needed protractor
measurements nobody could take and so never produced a mapping.

## Conversion

```
ticks   = range_min + (normalized - lower) / (upper - lower) * (range_max - range_min)
q_model = sign * (ticks - zero_tick) * 2*pi / 4096
```

Rotary lower/upper are -100/100, gripper 0/100 (0 closed). The first line is the
LeRobot driver's own formula with the saved calibration in `arm_config.json`
(the driver truncates ticks with `int()`; homing offsets are already applied by
the device). The scale in the second line is hardware fact: STS3215, direct
drive, 4096 ticks per revolution. Only `sign` and `zero_tick` are per-joint
unknowns. `station_coordinates.NOMINAL` sets sign +1 and zero_tick 2048 for all
six, because LeRobot homes the mid-pose to tick 2048 and the Menagerie model's
zero is that mid-pose.

Commands are limited to the intersection of model limits and mapped encoder
limits; nothing is stretched or clipped silently. To override a joint, pass a
JSON file with the same shape as `NOMINAL` via `--mapping`.

**This is a simulation-side convention. It changes no hardware calibration;
`hardware_compatible` stays false until a supervised real-arm trial.**

## Evidence (station checks, 2026-11-14)

Captures are in `outputs/joint-mapping/` (ignored by Git; they show the workspace).

| Joint | Evidence | Status |
|---|---|---|
| shoulder_pan | Operator out-and-back earlier in the day; under teleoperation a swing to +36.8 put the gripper over the blue bowl in both the overhead camera and the mapped render (`teleop-02-pan-open`). | sign and zero confirmed |
| shoulder_lift, elbow_flex, wrist_flex | `check_real_replay.py` on the 7 real episodes (4,366 frames): grasp point never below 13 mm, and 13-29 mm above the table at all 7 gripper-close events. Flipping any one sign drives it 30-85 mm through the table. Rest, forward and swung poses render as photographed. | sign and zero confirmed |
| wrist_roll | Hand-turned a quarter turn counter-clockwise (viewed from the wrist toward the jaw tips): 1128 -> 2091 ticks, +85 degrees as predicted; render matches the camera before and after (`hand-01-roll-ccw-closed`). | sign and zero confirmed |
| gripper | Hand-closed reads tick 2025 (-2 degrees). Holding a 2x2 brick across 32 mm reads 2.2 normalized on the station and 5.8 in the model (`teleop-01-brick`). | direction and closed confirmed; **3.6-unit gap at grasp** |

Zeros are good to about 5 degrees, not better; nothing was measured with an
instrument. The gripper gap comes from the station's flexible red fingers, which
differ from the model's rigid jaw. Shifting the gripper zero by 55 ticks removes
the gap but leaves the model too little squeeze (expert success fell from 30/30
to 17/30), so the mapping keeps zero at 2048 and the gap is a known state offset
while gripping. Real jaw travel is 134 degrees against the model's 100, so
commands above ~74 are outside the common range. Fully-open width in mm was not
measured.

## Converted dataset

`convert_recordings.py` turned `outputs/dataset-150-raw` (radians) into
`outputs/dataset-150-normalized`: **146 episodes, 53,028 frames**, audited by
`verify_recordings.py`. States and actions are both converted. Gripper close
targets below the real closed position were raised to 0; each episode was then
replayed through physics and kept only if the placement still succeeded. Four
were dropped (`conversion.json`); six of the original radian episodes do not
replay to success on this machine either, so this is replay sensitivity, not
the mapping. The radian dataset and its checkpoint are untouched and must not
be mixed with normalized data or statistics.

## Not done

- No LeRobot export or training on the normalized set. `export_lerobot.py
  --min-episodes 146` and `train_simulation.py` run in the GPU environment and
  now accept it; copy `station_coordinates.py` and `arm_config.json` alongside.
  `rollout_policy.py --normalized` evaluates a normalized checkpoint.
- Units do not fix the policy: the radian checkpoint scored one grasp and zero
  placements in six simulated trials, and the sim cameras are hand-fitted.
- No real-arm validation. Do not deploy a sim-trained policy on this mapping
  without the wrist-roll and gripper checks and a supervised trial.
