#!/bin/bash
# S3: parity detail from raw npz (read only), then the guarded perception rerun (gated at 1-min load below 14).
R=/home/intel/intel-junction-2026
E=$R/work/P4-OVBENCH/E1
PY=$R/envs/data-env/bin/python
OUT=$E/logs/s3_result.txt
{
echo "== S3 start $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
[ -f $E/device_map.patch ] && mv $E/device_map.patch $E/device_map_NPU_candidate_REJECTED.patch
rm -rf $E/app_copy
echo "NPU candidate renamed to device_map_NPU_candidate_REJECTED.patch; app_copy removed"
echo "== pre_close threshold.json"
cat $R/app/anomaly/anom3/models/v000/pre_close/threshold.json
echo
echo "== parity detail (raw npz, read only)"
$PY -I $E/tools/parity_detail.py 2>&1
echo "== S3 parity done $(date -u +%FT%TZ)"
for i in $(seq 1 45); do
  L=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l="$L" 'BEGIN{exit !(l+0 < 14)}'; then break; fi
  sleep 20
done
echo "== guarded perception rerun: gate load $(cut -d' ' -f1-3 /proc/loadavg) at $(date -u +%FT%TZ)"
rm -f $E/logs/guard_hits.txt
$PY -I $E/tools/perception_guarded.py > $E/logs/perception_guarded.log 2>&1
echo "guarded perception exit=$? at $(date -u +%FT%TZ)"
echo "guard hits: $( [ -f $E/logs/guard_hits.txt ] && wc -l < $E/logs/guard_hits.txt || echo 0 )"
[ -f $E/logs/guard_hits.txt ] && head -n 20 $E/logs/guard_hits.txt
echo "-- guarded perception log tail"
tail -n 25 $E/logs/perception_guarded.log | cut -c1-220
echo "-- perception_timing.tsv"
cat $E/perception/perception_timing.tsv 2>&1
echo "-- cycle_phases.tsv"
cat $E/perception/cycle_phases.tsv 2>&1
echo "== S3 DONE $(date -u +%FT%TZ)"
} > $OUT 2>&1
