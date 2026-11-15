PARTIAL Sat Nov 14 22:02:52 UTC 2026 3 visually verified successful episodes (7750 frames, 258.3 s; 3 cameras AV1 30 fps); 38 candidate episodes excluded with reasons; checks 27 PASS 0 WARN 0 FAIL

Dataset: /home/intel/intel-junction-2026/data/ours-lerobot/v1. Working files: /home/intel/intel-junction-2026/work/DATA-LEROBOT/E1.
Included: any_1740/ep_0001 (orange on blue); any_1740/ep_0015 (red on blue); any_1740/ep_0017 (yellow on green).
Excluded: 38 (joint or label data 1, loop not a success 14, loop success not visually verified 1, not finished 3, visual check 18, visual placed, data check failed 1).
Visually placed but excluded for a data reason: any_1731/ep_0003.
Loop successes not visually verified: any_2055/ep_0001, any_2127/ep_0008.
Checks: 27 PASS, 0 WARN, 0 FAIL (validation.json).
Open: close clock not determined (0 ms); wrist offset conflict (P2 R4 vs P3-VLA-CLOCK); needs_review on 2 of 3.
Requests to Opus: decide the wrist offset for training; review the visual entries for the loop-only successes listed above before they can be included.
Visual check of 5 frames (see data_card.md, Visual check): the blue plate already holds bricks before placement in ep_0001 and ep_0015; release frames need a full-resolution check.
