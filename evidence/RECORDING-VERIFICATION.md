# Recording verification — 14 November 2026

Dataset `6d6bdce3-b382-478d-9697-fb9e573b97af` finalized through the supported session stop endpoint. Two older saved episodes plus five newly flushed episodes = seven total, 4,366 frames at 30 FPS. Episode lengths: 673, 580, 544, 704, 706, 647, 512 frames (17.1–23.5 seconds). All share task `move the brick to the inspection spot`.

Parquet row count matches metadata; all six-dimensional action/state values finite. All four video containers fully decoded via PyAV at 640×480: per camera, file-000 has 1,253 frames and file-001 has 3,113 frames, matching episode totals. This verifies storage integrity, not physical task quality or camera framing.

Submitted ACT smoke training on local xpu:0: one epoch, batch 8, float32, two loader workers, no validation split, OpenVINO export requested. Job `5f3a6be6-067e-473f-900a-311e2d609c8d`; snapshot `531fb33a-72e2-42d6-9d34-678223e4d00f`. Status running at submission verification; model initialized with 51.6M parameters. No autonomous movement initiated. Training completion and export require follow-up.
