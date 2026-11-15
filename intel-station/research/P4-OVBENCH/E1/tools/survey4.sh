#!/bin/bash
# E1 P4-OVBENCH read-only survey 4: which OpenVINO workloads the live loop calls, import side effects, pure functions to time.
R=/home/intel/intel-junction-2026
T=$R/app/tools
D=$R/app/anomaly/det2_f2
echo "== UTC $(date -u +%FT%TZ)"
echo "== openvino / compile usage in loop code and det2_f2 (excluding bench and tests)"
grep -n -E 'openvino|OpenVINO|compile_model|read_model|\.xml|Core\(' $T/*.py $D/*.py 2>/dev/null | grep -v -E 'bench_ov|tests/' | cut -c1-200 | head -60
echo "== DET2./DET. calls in loop modules (counts)"
grep -h -o -E '(DET2|DET|sl|pa|ga|kin)\.[A-Za-z_]+\(' $T/sort_any.py $T/sort_loop.py 2>/dev/null | sort | uniq -c | sort -rn | head -60
echo "== imports in det2_f2/inspector.py, sort_loop.py, kin.py, rs_height.py, paths.py"
for f in $D/inspector.py $T/sort_loop.py $T/kin.py $T/rs_height.py $T/paths.py; do echo "-- $f"; grep -n -E '^(import |from )' "$f" 2>&1 | cut -c1-160; done
echo "== top-level statements (side effects) in those modules"
for f in $D/inspector.py $T/sort_loop.py $T/kin.py $T/rs_height.py $T/paths.py; do echo "-- $f"; grep -n -v -E '^(def |class |import |from |#|\s|$|\)|\]|\}|""")' "$f" 2>&1 | cut -c1-160 | head -40; done
echo "== pick_any.py frame, move_rel, m_base, servo (101-178)"
sed -n '101,178p' $T/pick_any.py
echo "== sort_any.py scan_cal (47-53)"
sed -n '47,53p' $T/sort_any.py
echo "== sort_any.py grasp_reachable and candidates (150-204)"
sed -n '150,204p' $T/sort_any.py
echo "== sort_any.py close_pre_verdict (289-335)"
sed -n '289,335p' $T/sort_any.py
echo "== END survey4 $(date -u +%FT%TZ)"
