#!/bin/bash
# S1: status, config lines and static import scan. Stops only E1's own perception run (never the motion server).
R=/home/intel/intel-junction-2026
E=$R/work/P4-OVBENCH/E1
MOTION=3522921
OUT=$E/logs/s1_result.txt
PY=$R/envs/data-env/bin/python
INC="--include=*.py --include=*.yaml --include=*.yml --include=*.json --include=*.toml --include=*.md --include=*.sh --include=*.txt --include=*.cfg --include=*.ini"
{
echo "== S1 start $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
echo "== E1 processes (motion server listed only, not touched)"
ps -o pid,etime,stat,args -p 3589015,3668599 2>/dev/null | cut -c1-200
pgrep -af "run_ovbench|run_pass2|bench_ov|aggregate|perception_timing" | cut -c1-200
ps -o pid,etime,args -p $MOTION 2>/dev/null | cut -c1-160
} > $OUT 2>&1
for p in $(pgrep -f "perception_timing.py"); do
  if [ "$p" != "$MOTION" ]; then
    kill -TERM "$p" 2>/dev/null && echo "STOPPED E1 perception_timing PID $p before the import check" >> $OUT
  fi
done
{
echo "== E1 tools"
ls $E/tools 2>&1 | tr '\n' ' '
echo
echo "== perception and bench outputs"
ls -la --time-style=+%FT%TZ $E/perception 2>&1 | head -n 30
ls -la --time-style=+%FT%TZ $E/bench.tsv $E/metrics.json 2>&1
echo "raw files: $(ls $E/raw 2>/dev/null | wc -l)"
echo "== ovbench_run.log tail"
tail -n 30 $E/logs/ovbench_run.log 2>&1 | cut -c1-220
echo "== pass2_run.log tail"
tail -n 15 $E/logs/pass2_run.log 2>&1 | cut -c1-220
echo "== app size"
timeout 120 du -sh $R/app 2>&1
echo "== anom3 infer.py: DEVICES and batch or shape lines"
grep -n "DEVICES\|compile_model\|device\|reshape\|shape\|stack\|batch" $R/app/anomaly/anom3/infer.py 2>&1 | cut -c1-200 | head -n 60
echo "== DEVICES anywhere in app"
grep -rn $INC "DEVICES" $R/app 2>/dev/null | cut -c1-200 | head -n 20
echo "== app files importing openvino"
grep -rl $INC "openvino" $R/app 2>/dev/null | head -n 60
echo "== ACT IR references"
grep -rn $INC "P2-B-OVRUN\|389da0fd\|P2-B-OPT\|act\.xml\|act_fp16" $R/app 2>/dev/null | cut -c1-220 | head -n 40
echo "== scratch and excluded IR references"
grep -rn $INC "P2-B-ANOM3\|scratch\|superseded_cache_bug\|P2-B-DET2" $R/app 2>/dev/null | cut -c1-220 | head -n 40
echo "== device settings in config files under app"
grep -rn -i $INC "device" $R/app 2>/dev/null | cut -c1-220 | head -n 60
echo "== perception_timing.py import setup"
grep -n "sys.path\|^import\|^from" $E/tools/perception_timing.py 2>&1 | cut -c1-200 | head -n 40
echo "== static import scan (AST only; nothing imported or run)"
$PY -I $E/tools/scan_imports.py $R/app grasp_auto,rs_height,paths,pick_any,sort_any,kin,sort_loop,handdet,inspector 2>&1
echo "== S1 DONE $(date -u +%FT%TZ)"
} >> $OUT 2>&1
