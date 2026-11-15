PARTIAL 2026-11-14T20:52Z E3 P3-TWIN-ROBOT: so101_twin_v3.xml published (jaw box pads; G1 PASS 1.36e-8 mm). Grasp still 0/10 (brick pushed away during the 26 mm descent). Gripper law unresolved.

## v3 state
- Pads: pad_fixed and pad_moving boxes from jaw vertices within 15 mm of the TCP (tools/build_v3.py). Pad gap is monotone with reading: 1.75 mm (0.52), 3.69 mm (1.57), 5.04 mm (2.29), 36.67 mm (24.6).
- act_gripper kp 5.0, forcerange +/-10 (not yet matched to the logs' stall).
- test_grasp.py: default SRC is v3; integral correction in goto; hover centres the jaw midpoint over the brick; normal-force logging.

## Findings
- Diagnosis (tools/diag_v3_contacts.py): the brick touches the world at start (about -0.01 mm), then at the 26 mm descent pad_fixed/gripper_link touch the brick (-0.03 to -0.04 mm distance), and at the 8 mm descent the brick is flung to x -0.19 m, y 0.045 m (363 mm from the TCP).
- The TCP sits over the fixed jaw (about 21 mm from the jaw midpoint at open), so the brick was under the fixed jaw. Centring the jaw midpoint did not change the outcome: pad placement in world frame still needs checking (pads were fitted in body frames, not in the world frame at the grasp pose).
- Pad radius 12, 15 and 30 mm gave the same nominal result (no mesh vertices between radii); not a bug in the file.
- Real logs (outputs/gripper_logs.json): servo stops on the brick with goal 0 at 1.50-2.42 (held); empty stops at 0.52 and 0.65. Held minus empty is about 1 unit (0.024 rad) at an 82.8 mm radius, about 2 mm of jaw travel. Inconsistent with a 31.7 mm brick. Not resolved.

## Not done
- world-frame pad placement at the grasp pose; stall-matched gripper force; 10/10 holds; reading match.

## DC calls: about 20 this pass. Quota near exhausted (85%+). No remote-devices tool used.
