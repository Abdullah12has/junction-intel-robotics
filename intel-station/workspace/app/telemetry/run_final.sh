#!/usr/bin/env bash
# P2-B-OPT final job: (1) harness bench_ov.py, then (2) the Robotics AI Suite recipe
# (object-detection.yaml L38-47: explicit device, -hint latency, -t 10, npu_busy_time_us delta)
# applied to the same IR files.
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # workspace root, from this script's location
W=$R/work/P2-B-OPT/E1
PY=$R/envs/anomalib-env/bin/python
BA=$R/envs/anomalib-env/bin/benchmark_app
NPU_BUSY=/sys/class/accel/accel0/device/npu_busy_time_us
OUT=$W/benchmark_app
mkdir -p "$OUT"
cd "$W" || exit 1
echo "== harness start $(date -u +%FT%TZ)"
nice -n 5 "$PY" -I "$R/app/telemetry/bench_ov.py"
echo "== harness exit $? $(date -u +%FT%TZ)"
echo "== recipe start $(date -u +%FT%TZ)"
: > "$OUT/summary.txt"
for pair in act_fp32=$W/models/389da0fd/act.xml act_fp16=$W/models/389da0fd_fp16/act.xml patchcore_b2=$R/models/anomaly/patchcore/weights/openvino/model.xml padim_b1=$R/models/anomaly/padim/weights/openvino_b1/model.xml; do
  name=${pair%%=*}
  M=${pair#*=}
  for D in CPU GPU NPU; do
    B0=$(cat "$NPU_BUSY" 2>/dev/null || echo 0)
    T0=$(date +%s.%N)
    nice -n 5 "$BA" -m "$M" -d "$D" -hint latency -t 10 > "$OUT/${name}_${D}.log" 2>&1
    RC=$?
    T1=$(date +%s.%N)
    B1=$(cat "$NPU_BUSY" 2>/dev/null || echo 0)
    WALL=$(awk "BEGIN{printf \"%.3f\", $T1-$T0}")
    echo "== $name $D rc=$RC wall_s=$WALL npu_busy_time_us_delta=$((B1-B0))" >> "$OUT/summary.txt"
    grep -E "EXECUTION_DEVICES|Median:|Throughput:" "$OUT/${name}_${D}.log" >> "$OUT/summary.txt"
    sleep 3
  done
done
echo "== recipe end $(date -u +%FT%TZ)"
