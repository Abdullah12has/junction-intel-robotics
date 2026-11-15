#!/bin/bash
# S8 phase 1b (read only): threshold rule and calibration list format, threshold writer, physicalai precision defaults.
R=/home/intel/intel-junction-2026
E=$R/work/P4-OVBENCH/E1
A=$R/app/anomaly/anom3
P=$A/models/v000/pre_close
{
echo "== utc $(date -u +%FT%TZ)"
echo "== common.py 100-160 (numbered)"
awk 'NR>=100 && NR<=160 {printf "%d: %s\n", NR, $0}' $A/common.py | cut -c1-200
echo "== def lines in common.py"
grep -n "^def " $A/common.py | head -n 30
echo "== eval_list.tsv: line count, head"
wc -l < $P/eval_list.tsv
head -n 5 $P/eval_list.tsv | cut -c1-240
echo "== train_list.txt head"
head -n 3 $P/train_list.txt | cut -c1-240
echo "== disagreements.tsv"
cat $P/disagreements.tsv | cut -c1-240
echo "== writers of threshold dict (targeted)"
timeout 40 grep -rln "\"threshold\"" $R/app/anomaly --include=*.py 2>/dev/null | head -n 10
timeout 40 grep -rn "threshold.json\|\"threshold\":" $A --include=*.py 2>/dev/null | cut -c1-200 | head -n 12
ls -la $A | head -n 40 | cut -c1-160
echo "== physicalai package location(s)"
ls -d $R/envs/*/lib/python3*/site-packages/physicalai 2>/dev/null
echo "== precision settings in physicalai (openvino backend)"
for d in $(ls -d $R/envs/*/lib/python3*/site-packages/physicalai 2>/dev/null); do timeout 40 grep -rn "PRECISION\|precision\|compile_model" $d 2>/dev/null | cut -c1-220 | head -n 25; done
echo "== phase1b done $(date -u +%FT%TZ)"
} > $E/logs/s8_phase1b.txt 2>&1
