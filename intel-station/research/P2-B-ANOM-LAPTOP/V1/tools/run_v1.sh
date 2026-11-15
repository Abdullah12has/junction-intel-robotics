#!/bin/bash
# V1 re-run (P2-B-ANOM-LAPTOP): E1's infer.py on the exported FP16 IRs (all 20 test images, CPU/GPU/NPU), torch-side
# CPU scores for the parity check, then the comparison summary. Writes only inside the V1 laptop folder.
source /home/intel/intel-junction-2026/envs/env.sh
V1=$PROJ/work/P2-B-ANOM-LAPTOP/V1
APP=$PROJ/app/anomaly
DATA=$PROJ/data/anomaly/synthetic
cd "$PROJ" || exit 1
echo "== sha256 of E1 scripts on laptop =="
sha256sum $APP/infer.py $APP/train.py $APP/export.py $APP/static_ir.py
for dev in CPU GPU NPU; do
  echo "== infer padim batch1 on $dev =="
  timeout 600 $PY -I $APP/infer.py --model padim --ir $PROJ/models/anomaly/padim/weights/openvino_b1/model.xml --batch 1 --device $dev --data $DATA --out $V1/runs/infer-padim-$dev.json > $V1/logs/infer-padim-$dev.log 2>&1
  echo "infer padim $dev rc=$?"
  echo "== infer patchcore batch2 on $dev =="
  timeout 600 $PY -I $APP/infer.py --model patchcore --ir $PROJ/models/anomaly/patchcore/weights/openvino/model.xml --batch 2 --device $dev --data $DATA --out $V1/runs/infer-patchcore-$dev.json > $V1/logs/infer-patchcore-$dev.log 2>&1
  echo "infer patchcore $dev rc=$?"
done
echo "== torch CPU scores =="
timeout 900 $PY -I $V1/tools/torch_predict.py --model padim --ckpt $PROJ/models/anomaly/padim/run/Padim/synthetic/v0/weights/lightning/model.ckpt --data $DATA --out $V1/runs/torch-padim.json > $V1/logs/torch-padim.log 2>&1
echo "torch padim rc=$?"
timeout 900 $PY -I $V1/tools/torch_predict.py --model patchcore --ckpt $PROJ/models/anomaly/patchcore/run/Patchcore/synthetic/v0/weights/lightning/model.ckpt --data $DATA --out $V1/runs/torch-patchcore.json > $V1/logs/torch-patchcore.log 2>&1
echo "torch patchcore rc=$?"
echo "== compare =="
python3 -I $V1/tools/compare.py --runs $V1/runs --data $DATA --out $V1/runs/v1-summary.json
echo "compare rc=$?"
echo RUN_DONE
