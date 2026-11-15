import hashlib, re, os

E2 = '/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/E2/'
OUT = E2 + 'out/'
RUN = '2026-11-14T20:56Z'

def rd(p):
    with open(p, encoding='utf-8') as f:
        return f.read()

def wr(p, s):
    with open(p, 'w', encoding='utf-8') as f:
        f.write(s)

def rep(s, old, new, count=1):
    n = s.count(old)
    assert n == count, (old[:60], n)
    return s.replace(old, new)

A = rd(OUT + '_partA.md')
A = rep(A, 'run 2026-11-14T20:58Z', 'run 2026-11-14T20:56Z')
A = rep(A, '564.6 s in 6 episodes', '564.8 s in 6 episodes')
A = rep(A, 'e60f4087 488.2 s in 5 episodes', 'e60f4087 488.4 s in 5 episodes')

B = rd(OUT + '_partB.md')
B = rep(B, 'run 2026-11-14T20:58Z', 'run 2026-11-14T20:56Z')
B = rep(B, '"pixel check" is frame_check_vs_rep.frame_verdict; "own frames" means the representative was viewed by eye.',
        "Pixel results are in each row's frame_check_vs_rep field.")
B = re.sub(r'- Joint sanity for the 11 new rows:.*?\n',
           '- Joint sanity for the 11 new rows (checked in this run): action lags observation.state by 3 frames in all 11 (lag_best 3); MAE 0.706 to 0.866; action_range_in_bounds 11 of 11; frame_index contiguous 11 of 11. Shoulder_lift and elbow/wrist extremes were not re-checked in this run.\n',
           B, count=1, flags=re.S)
B = re.sub(r'- Growth: .*?\n',
           '- The census lists 52 rows: 8a8bb05e (16943 rows in 6 episodes) and e60f4087 (14651 rows in 5 episodes) are the new folders since the E1 census (19:28Z, 9 folders). Other folders were not re-read for row counts in this run.\n',
           B, count=1, flags=re.S)
SEC7 = (
    '## 7. E1 mismatch re-check (carried from E1; not re-viewed in this run)\n'
    'The 12 E1 mismatch entries were confirmed against frames in the E1 run (E1 trust_report.md). In this E2 run their rows were checked only as unchanged: the first 41 rows of labels_episodes.jsonl are byte-identical to E1, and the census and metrics agree. Montages were not re-viewed in E2, so the per-entry visual notes are not repeated here.\n'
    'Copies of P6 and P8 (18468a09:7, 18468a09:8, 51100357:0, 800d7dc2:2, ba9eafba:0, ba9eafba:1) share action rows with their representative, so their mismatch follows that representative.\n'
    'Curated and V1 mismatch rows were not re-checked in this run (curated labels are not used as truth).\n\n'
)
B = re.sub(r'## 7\. E1 mismatch re-check.*?(?=## 8\.)', lambda m: SEC7, B, count=1, flags=re.S)
B = rep(B, '10. 7e99a9e4 unchanged: 8405 rows.',
        '10. 7e99a9e4 unchanged in the census: 1 row (P17); its row count was not re-read in this run.')
B = rep(B, '11. mismatches.tsv: 41 E1 rows plus 12 new rows (11 OD-10 and 1 not_placed).',
        '11. mismatches.tsv: 53 data rows (the E1 file had 41); row kinds were not tallied in this run. Entry counts are in metrics.json: 23 mismatched, 20 OD-10 wrong, 4 task text.')
B = rep(B, '14. utc fields hold the time the files were written (20:53Z); snapshot_utc is the 20:44Z listing.',
        '14. utc fields set to 2026-11-14T20:56Z (run stamp); snapshot_utc is the 20:44Z listing.')

tn = rd(OUT + '_tables_new.md')
te = rd(OUT + '_tables_entries.md')
report = A.rstrip('\n') + '\n\n' + tn.strip('\n') + '\n\n' + te.strip('\n') + '\n\n' + B.lstrip('\n')
wr(E2 + 'trust_report.md', report)

STATUS = (
    f'PARTIAL {RUN} E2 P3-LABEL-STUDIO (stage studio-labels-v2)\n'
    'Snapshot: mirror census at 20:44Z: 11 Studio folders, 52 episode rows, 23 physical episodes (E1: 9 folders, 41 rows, 17 physical).\n'
    'Done on laptop (R/work/P3-LABEL-STUDIO/E2/): labels_episodes.jsonl (52 rows; first 41 identical to E1), datasets.json, mismatches.tsv, metrics.json, trust_report.md (v2), STATUS.md, laptop-log.md. E1 folder unchanged.\n'
    'Not done: copy of the E2 deliverables to sandbox work/P3-LABEL-STUDIO/E2 (sandbox holds only this STATUS); re-census before handback; video-span check for new rows; curated sets not re-run; E1 montages not re-viewed.\n'
    'Next: transfer deliverables to sandbox with SHA-256 checks, re-census the mirror, then hand back.\n'
)
wr(E2 + 'STATUS.md', STATUS)

LOG = (
    f'# E2 laptop log (P3-LABEL-STUDIO, stage studio-labels-v2), {RUN}\n\n'
    '- 20:44Z mirror snapshot: tools/census_e2.py read all meta and data parquet files in 11 folders; 52 rows; containment by exact action-window match; out/census_e2.json.\n'
    '- Montages, full-res frames and zoom grids for 8a8bb05e:0 to :5 (tools/e2_montage.py, e2_fullres.py, e2_zoomgrid.py); per-brick notes in tools/build_e2.py.\n'
    '- Pixel checks for the 5 new copies (e60f4087:0 to :4): MAE 0.0 at frames 60 and n/2 on all cameras; out/e2_pixel_sanity.json.\n'
    '- tools/build_e2.py wrote labels_episodes.jsonl, mismatches.tsv, datasets.json, metrics.json; placeholder utc stamps replaced with 2026-11-14T20:56Z by tools/finalize_e2.py.\n'
    '- tools/make_report_e2.py asserted E1 prefix equality, info totals against row counts (11 folders), and trust counts (23 training, 23 evaluation, 5 calibration, 1 nothing).\n'
    '- tools/finalize_e2.py assembled trust_report.md from typed sections (out/_partA.md, out/_partB.md) and out/_tables_new.md, out/_tables_entries.md.\n'
    '- Joint sanity for the 11 new rows: lag 3 frames, in-bounds 11 of 11, frame index contiguous 11 of 11.\n'
    '- Not done in this run: sandbox copy of deliverables; E1 montages re-viewed; mismatches.tsv row kinds tallied; video-span check.\n'
    '- Remote-devices connector failed (503) during the run; Desktop Commander was used throughout.\n'
)
wr(E2 + 'laptop-log.md', LOG)

for name in ['trust_report.md', 'STATUS.md', 'laptop-log.md', 'datasets.json', 'metrics.json', 'mismatches.tsv', 'labels_episodes.jsonl']:
    p = E2 + name
    b = open(p, 'rb').read()
    print(name, len(b), hashlib.sha256(b).hexdigest()[:16])
