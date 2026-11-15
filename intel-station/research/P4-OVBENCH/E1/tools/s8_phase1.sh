#!/bin/bash
# S8 phase 1 (read only): anom3 infer.py, pre_close threshold provenance, runner precision settings, bench snapshot loader.
R=/home/intel/intel-junction-2026
E=$R/work/P4-OVBENCH/E1
{
echo "== utc $(date -u +%FT%TZ)"
echo "== infer.py (numbered)"
awk '{printf "%d: %s\n", NR, $0}' $R/app/anomaly/anom3/infer.py
echo "== pre_close threshold.json"
cat $R/app/anomaly/anom3/models/v000/pre_close/threshold.json; echo
echo "== pre_close metrics.json"
cat $R/app/anomaly/anom3/models/v000/pre_close/metrics.json; echo
echo "== pre_close folder"
ls -la $R/app/anomaly/anom3/models/v000/pre_close/
echo "== writers of threshold/calibration (anom3 and P2-B-ANOM3 .py/.sh/.md)"
timeout 40 grep -rn "threshold.json\|balanced_accuracy\|auroc\|n_good\|scoring_device" $R/app/anomaly $R/work/P2-B-ANOM3 --include=*.py --include=*.sh --include=*.md 2>/dev/null | cut -c1-220 | head -n 40
echo "== InferenceModel: class and imports"
timeout 40 grep -rln "class InferenceModel" $R/app 2>/dev/null | head -n 5
timeout 40 grep -n "import\|InferenceModel" $R/app/ovrun/runner.py 2>/dev/null | cut -c1-200 | head -n 30
echo "== precision, compile and config lines (ovrun, app/common, anomaly/anom3)"
timeout 40 grep -rn "PRECISION\|INFERENCE_\|hint\|compile_model\|core.compile\|set_property" $R/app/ovrun $R/app/common $R/app/anomaly/anom3 2>/dev/null | cut -c1-220 | head -n 40
echo "== runner.py device/model/backend lines"
awk '/device|DEVICE|model_dir|EXPORT|InferenceModel|backend/ {printf "%d: %s\n", NR, $0}' $R/app/ovrun/runner.py | cut -c1-200 | head -n 40
echo "== replay.py EXPORT lines"
awk '/EXPORT|P2-B-OVRUN/ {printf "%d: %s\n", NR, $0}' $R/app/ovrun/replay.py | cut -c1-200 | head -n 10
echo "== bench snapshot loader (bench_ov_v2.py 96-128)"
awk 'NR>=96 && NR<=128 {printf "%d: %s\n", NR, $0}' $E/tools/bench_ov_v2.py | cut -c1-200
echo "== pre_close raw files"
ls -la $E/raw | grep anom_pre_close | cut -c1-160
echo "== phase1 done $(date -u +%FT%TZ)"
} > $E/logs/s8_phase1.txt 2>&1
