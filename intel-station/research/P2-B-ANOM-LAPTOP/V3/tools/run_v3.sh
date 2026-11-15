#!/bin/bash
# V3 verifier runner (P2-B-ANOM-LAPTOP): re-runs E1's app/anomaly/infer.py unchanged on the exported IRs, CPU then GPU then NPU.
# Writes only inside the V3 laptop folder; caches stay inside the project through envs/env.sh.
source /home/intel/intel-junction-2026/envs/env.sh
export PYTHONDONTWRITEBYTECODE=1
PROJ=/home/intel/intel-junction-2026
APP=$PROJ/app/anomaly
DATA=$PROJ/data/anomaly/synthetic
RUNS=$PROJ/work/P2-B-ANOM-LAPTOP/V3/runs
mkdir -p "$RUNS"
PADIM_B1=$PROJ/models/anomaly/padim/weights/openvino_b1/model.xml
PADIM_DYN=$PROJ/models/anomaly/padim/weights/openvino/model.xml
PATCH_B2=$PROJ/models/anomaly/patchcore/weights/openvino/model.xml
run() {
  local name=$1 model=$2 ir=$3 batch=$4 dev=$5
  echo "== infer $name on $dev (batch $batch) =="
  timeout 300 "$PY" -I "$APP/infer.py" --model "$model" --ir "$ir" --batch "$batch" --device "$dev" --data "$DATA" --out "$RUNS/infer-$name-$dev.json" > "$RUNS/infer-$name-$dev.log" 2>&1
  echo "rc=$?"
}
for dev in CPU GPU NPU; do
  run padim-static-b1 padim "$PADIM_B1" 1 "$dev"
  run patchcore-b2 patchcore "$PATCH_B2" 2 "$dev"
  run padim-dyn-b1 padim "$PADIM_DYN" 1 "$dev"
done
echo RUNNER_DONE
