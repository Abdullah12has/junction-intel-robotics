#!/bin/bash
# S2: config and closure evidence, guarded perception rerun, candidate patch on a COPY of app/ (R/app untouched).
R=/home/intel/intel-junction-2026
E=$R/work/P4-OVBENCH/E1
PY=$R/envs/data-env/bin/python
OUT=$E/logs/s2_result.txt
{
echo "== S2 start $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
echo "== anom3 infer.py lines 40-70"
awk 'NR>=40 && NR<=70 {print NR": "$0}' $R/app/anomaly/anom3/infer.py | cut -c1-200
echo "== policy device lines (loop_v2 yaml and policy_cycle)"
grep -n -i "device\|npu\|gpu\|openvino" $R/app/loop_v2/app/loop/policy_rules.yaml | cut -c1-220 | head -n 30
grep -n -i "device" $R/app/loop_v2/app/loop/policy_cycle.py | cut -c1-220 | head -n 10
echo "== ovrun runner model paths"
grep -n "xml\|EXPORT\|W =\|MODEL\|device\|DEVICE" $R/app/ovrun/runner.py | cut -c1-220 | head -n 30
echo "== static scan: det2_f2 inspector (loop DET2 import) and its closure"
$PY -I $E/tools/scan_imports.py $R/app $R/app/anomaly/det2_f2/inspector.py 2>&1
echo "== dynamic import or eval in closure files"
grep -n "importlib\|__import__\|exec(\|eval(" $R/app/tools/grasp_auto.py $R/app/tools/rs_height.py $R/app/tools/paths.py $R/app/tools/pick_any.py $R/app/tools/sort_any.py $R/app/tools/kin.py $R/app/tools/sort_loop.py $R/app/tools/handdet.py $R/app/anomaly/det2_f2/*.py $R/app/pick_v2/kinematics.py 2>&1 | cut -c1-200 | head -n 20
} > $OUT 2>&1
{
echo "== guarded perception rerun (pass-1 output was not import-checked; kept as UNTRUSTED)"
mkdir -p $E/perception/pass1_UNTRUSTED
for f in perception_timing.json perception_timing.tsv cycle_phases.tsv; do
  [ -f $E/perception/$f ] && mv $E/perception/$f $E/perception/pass1_UNTRUSTED/$f
done
rm -f $E/logs/guard_hits.txt
for i in $(seq 1 45); do
  L=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l="$L" 'BEGIN{exit !(l+0 < 14)}'; then break; fi
  sleep 20
done
echo "gate: load $(cut -d' ' -f1-3 /proc/loadavg) at $(date -u +%FT%TZ)"
$PY -I $E/tools/perception_guarded.py > $E/logs/perception_guarded.log 2>&1
echo "guarded perception exit=$? at $(date -u +%FT%TZ)"
echo "guard hits (blocked hardware or socket attempts): $( [ -f $E/logs/guard_hits.txt ] && wc -l < $E/logs/guard_hits.txt || echo 0 )"
[ -f $E/logs/guard_hits.txt ] && head -n 20 $E/logs/guard_hits.txt
echo "-- guarded perception log tail"
tail -n 30 $E/logs/perception_guarded.log | cut -c1-220
echo "-- perception_timing.tsv (guarded)"
cat $E/perception/perception_timing.tsv
echo "-- cycle_phases.tsv (guarded)"
cat $E/perception/cycle_phases.tsv
} >> $OUT 2>&1
{
echo "== bench.tsv (aggregate_v2 at PASS2 end)"
cat $E/bench.tsv
echo "== metrics.json"
cat $E/metrics.json
} >> $OUT 2>&1
{
echo "== candidate patch on a COPY of app (R/app untouched)"
rm -rf $E/app_copy
cp -a $R/app $E/app_copy && echo "copied app to $E/app_copy"
$PY -I $E/tools/make_patch.py $R/app $E/app_copy $E/device_map.patch
cd $R && patch -p1 --dry-run -i $E/device_map.patch; echo "dry-run exit=$?"
echo "-- device_map.patch"
cat $E/device_map.patch
echo "== S2 DONE $(date -u +%FT%TZ)"
} >> $OUT 2>&1
