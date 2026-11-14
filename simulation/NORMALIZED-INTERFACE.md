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

**This is a simulation-side convention. It changes no hardware calibration and
is labelled `nominal_unverified`; `hardware_compatible` stays false.**

## Evidence for the nominal mapping (no new station work)

| Joint | Evidence | Status |
|---|---|---|
| shoulder_lift, elbow_flex, wrist_flex | `check_real_replay.py` on the 7 real episodes (4,366 frames): grasp point never below 13 mm; at all 7 gripper-close events it is 13-29 mm above the table, i.e. brick height. Flipping any one of these signs drives it 30-85 mm through the table. Rest and forward captures render as the photographed poses. | supported |
| shoulder_pan | Model axis points down; operator's counter-clockwise move decreased the reading and returned within 0.44 degrees. | supported, one move |
| gripper | Model opens with increasing angle (expert closes at -0.08, opens at 0.7 rad). Calibrated closed tick is 2046. Real jaw travel is 134 degrees against the model's 100, so commands above ~74 are outside the common range. | direction supported, width unmeasured |
| wrist_roll | Reads about -54 normalized in every capture; nothing constrains its sign or zero. | **open** |

Zero offsets are good to roughly 5-7 degrees (visual zero-pose capture), not better.

Remaining station check, about ten minutes, read-only through Studio's existing
session: one out-and-back on wrist roll, and the gripper opening in mm at two
readings. Edit a copy of `NOMINAL` if either disagrees.

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
