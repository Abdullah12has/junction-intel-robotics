# E2 laptop log (P3-LABEL-STUDIO, stage studio-labels-v2), 2026-11-14T20:56Z

- 20:44Z mirror snapshot: tools/census_e2.py read all meta and data parquet files in 11 folders; 52 rows; containment by exact action-window match; out/census_e2.json.
- Montages, full-res frames and zoom grids for 8a8bb05e:0 to :5 (tools/e2_montage.py, e2_fullres.py, e2_zoomgrid.py); per-brick notes in tools/build_e2.py.
- Pixel checks for the 5 new copies (e60f4087:0 to :4): MAE 0.0 at frames 60 and n/2 on all cameras; out/e2_pixel_sanity.json.
- tools/build_e2.py wrote labels_episodes.jsonl, mismatches.tsv, datasets.json, metrics.json; placeholder utc stamps replaced with 2026-11-14T20:56Z by tools/finalize_e2.py.
- tools/make_report_e2.py asserted E1 prefix equality, info totals against row counts (11 folders), and trust counts (23 training, 23 evaluation, 5 calibration, 1 nothing).
- tools/finalize_e2.py assembled trust_report.md from typed sections (out/_partA.md, out/_partB.md) and out/_tables_new.md, out/_tables_entries.md.
- Joint sanity for the 11 new rows: lag 3 frames, in-bounds 11 of 11, frame index contiguous 11 of 11.
- Not done in this run: sandbox copy of deliverables; E1 montages re-viewed; mismatches.tsv row kinds tallied; video-span check.
- Remote-devices connector failed (503) during the run; Desktop Commander was used throughout.
