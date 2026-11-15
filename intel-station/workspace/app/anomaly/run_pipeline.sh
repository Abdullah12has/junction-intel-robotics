#!/bin/bash
# E1 pipeline on the laptop (P2-B-ANOM-LAPTOP), final procedure:
# dataset; train PaDiM and PatchCore on XPU (CPU fallback, reason recorded); FP16 OpenVINO export;
# batch-1 static copy of the PaDiM IR (static_ir.py, checked against the source on CPU);
# raw OpenVINO inference on GPU, CPU and NPU (PaDiM batch 1, PatchCore batch 2 as exported); collect results.json.
# Every step's output goes to logs/runs/. Run from the project folder with the env from envs/env.sh.
source /home/intel/intel-junction-2026/envs/env.sh
cd "$PROJ" || exit 1
APP=$PROJ/app/anomaly
DATA=$PROJ/data/anomaly/synthetic
RUNS=$PROJ/logs/runs
mkdir -p "$RUNS"

echo "== dataset =="
if [ ! -f "$DATA/manifest.json" ]; then
  $PY -I "$APP/make_synthetic.py" --out "$DATA" --seed 0 > "$RUNS/make_synthetic.log" 2>&1
  echo "make_synthetic rc=$?"
fi

for m in padim patchcore; do
  RUN=$PROJ/models/anomaly/$m/run
  echo "== train $m on xpu =="
  $PY -I "$APP/train.py" --data "$DATA" --model $m --accelerator xpu --out "$RUN" > "$RUNS/train-$m-xpu.log" 2>&1
  rc=$?
  echo "train $m xpu rc=$rc"
  if [ $rc -ne 0 ]; then
    reason="xpu rc=$rc: $(grep -E 'Error|Exception|SystemExit|is False' "$RUNS/train-$m-xpu.log" | tail -n 1)"
    echo "fallback to cpu, reason: $reason"
    $PY -I "$APP/train.py" --data "$DATA" --model $m --accelerator cpu --out "$RUN" --fallback_reason "$reason" > "$RUNS/train-$m-cpu.log" 2>&1
    echo "train $m cpu rc=$?"
  fi
  echo "== export $m (FP16 IR) =="
  $PY -I "$APP/export.py" --model $m --run "$RUN" --out "$PROJ/models/anomaly/$m" > "$RUNS/export-$m.log" 2>&1
  echo "export $m rc=$?"
done

echo "== static batch-1 copy of padim IR =="
$PY -I "$APP/static_ir.py" --src "$PROJ/models/anomaly/padim/weights/openvino/model.xml" --dst "$PROJ/models/anomaly/padim/weights/openvino_b1" --image "$DATA/test/defect/defect_000.png" > "$RUNS/static-padim.log" 2>&1
echo "static padim rc=$?"
echo "== static batch-1 attempt for patchcore (expected to fail: batch 2 is fixed in the graph) =="
$PY -I "$APP/static_ir.py" --src "$PROJ/models/anomaly/patchcore/weights/openvino/model.xml" --dst "$PROJ/models/anomaly/patchcore/weights/openvino_b1" --image "$DATA/test/defect/defect_000.png" > "$RUNS/static-patchcore.log" 2>&1
echo "static patchcore rc=$?"

for dev in GPU CPU NPU; do
  echo "== infer padim batch1 on $dev =="
  timeout 600 $PY -I "$APP/infer.py" --model padim --ir "$PROJ/models/anomaly/padim/weights/openvino_b1/model.xml" --batch 1 --device $dev --data "$DATA" --out "$RUNS/infer-padim-$dev.json" > "$RUNS/infer-padim-$dev.log" 2>&1
  echo "infer padim $dev rc=$?"
  echo "== infer patchcore batch2 on $dev =="
  timeout 600 $PY -I "$APP/infer.py" --model patchcore --ir "$PROJ/models/anomaly/patchcore/weights/openvino/model.xml" --batch 2 --device $dev --data "$DATA" --out "$RUNS/infer-patchcore-$dev.json" > "$RUNS/infer-patchcore-$dev.log" 2>&1
  echo "infer patchcore $dev rc=$?"
done

$PY -I "$APP/collect.py" --project "$PROJ" --out "$PROJ/logs/results.json" > "$RUNS/collect.log" 2>&1
echo "collect rc=$?"
echo PIPELINE_DONE
