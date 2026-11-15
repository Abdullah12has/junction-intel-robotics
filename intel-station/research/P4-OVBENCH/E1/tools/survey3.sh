#!/bin/bash
# E1 P4-OVBENCH read-only survey 3: loop imports, inspector defaults, cycle timings, artefact manifests.
R=/home/intel/intel-junction-2026
echo "== UTC $(date -u +%FT%TZ)"
echo "== imports in loop modules"
for f in sort_any.py pick_any.py grasp_auto.py handdet.py; do echo "-- $f"; grep -n -E '^(import |from )' "$R/app/tools/$f" | cut -c1-160; done
echo "== top-level statements (import side effects) in pick_any.py, grasp_auto.py, handdet.py"
for f in pick_any.py grasp_auto.py handdet.py; do echo "-- $f"; grep -n -v -E '^(def |class |import |from |#|\s|$|\)|\]|\}|""")' "$R/app/tools/$f" | cut -c1-160 | head -50; done
echo "== pick_any.py defs"
grep -n -E '^\s*(def|class) ' "$R/app/tools/pick_any.py" | cut -c1-160 | head -90
echo "== Inspector construction and model paths in loop code"
grep -rn -E 'Inspector\(|model_dir|det2_f2|anomalib-env|inspector' "$R/app/tools" --include='*.py' | cut -c1-220 | head -40
echo "== det2_f2 inspector defs, model and preprocessing lines"
grep -n -E '^\s*(def|class) |MODEL|model_dir|\.xml|resize|INPUT|IMAGENET|mean|std|SIZE|size|normal' "$R/app/anomaly/det2_f2/inspector.py" | cut -c1-200 | head -120
echo "== det2_f2 bench_ov.py (first 80 lines)"
head -n 80 "$R/app/anomaly/det2_f2/bench_ov.py"
echo "== sort_any.py cycle() and cycle timing writer (lines 476-580)"
sed -n '476,580p' "$R/app/tools/sort_any.py"
echo "== cycles.jsonl timings"
for f in "$R/logs/sort/any_1731/cycles.jsonl" "$R/logs/sort/any_1740/cycles.jsonl"; do echo "-- $f"; echo "null-timings lines: $(grep -c '"timings": null' "$f")  total: $(wc -l < "$f")"; grep -v '"timings": null' "$f" | head -n 1 | cut -c1-1500; done
echo "== loop.log head (any_1731)"
wc -l "$R/logs/sort/any_1731/loop.log"; head -n 25 "$R/logs/sort/any_1731/loop.log" | cut -c1-220
echo "== 389da0fd manifest"
head -c 1800 "$R/work/P2-B-OPT/E1/models/389da0fd/manifest.json"; echo
echo "== pre_close threshold.json and metrics.json (head)"
cat "$R/app/anomaly/anom3/models/v000/pre_close/threshold.json"; echo
head -c 1000 "$R/app/anomaly/anom3/models/v000/pre_close/metrics.json"; echo
echo "== GPU sysfs frequency or busy (find -L, depth 6)"
find -L /sys/class/drm/card1/device -maxdepth 6 \( -name 'act_freq' -o -name 'cur_freq' -o -name 'idle_residency_ms' -o -name '*busy*' \) -type f 2>/dev/null | sort | head -20
echo "== END survey3 $(date -u +%FT%TZ)"
