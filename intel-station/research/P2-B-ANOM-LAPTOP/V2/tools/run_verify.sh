#!/bin/bash
# V2 verifier run for P2-B-ANOM-LAPTOP: version check, hashes, dataset check, E1 infer.py on the exported FP16 IRs
# (CPU, GPU, NPU), PyTorch-vs-IR parity on 5 images, summary. No installs, no training. Writes only in the V2 folder.
source /home/intel/intel-junction-2026/envs/env.sh
V2=/home/intel/intel-junction-2026/work/P2-B-ANOM-LAPTOP/V2
DATA=$PROJ/data/anomaly/synthetic
PADIM_IR=$PROJ/models/anomaly/padim/weights/openvino_b1/model.xml
PATCH_IR=$PROJ/models/anomaly/patchcore/weights/openvino/model.xml
mkdir -p "$V2/logs" "$V2/runs"
cd "$V2" || exit 1

echo "== version check =="
$PY -I -c "import anomalib, torch, openvino as ov; print(anomalib.__version__, torch.__version__, torch.xpu.is_available(), ov.__version__, ov.Core().available_devices)" 2>&1
echo "rc=$?"

echo "== sha256 laptop scripts and IRs =="
(cd "$PROJ" && sha256sum app/anomaly/*.py app/anomaly/run_pipeline.sh models/anomaly/padim/weights/openvino/model.xml models/anomaly/padim/weights/openvino/model.bin models/anomaly/patchcore/weights/openvino/model.xml models/anomaly/patchcore/weights/openvino/model.bin 2>&1)

echo "== dataset =="
ls "$DATA"
for d in train/good test/good test/defect; do echo "$d $(ls "$DATA/$d" | wc -l)"; done
$PY -I -c "import glob; from PIL import Image; fs=sorted(glob.glob('$DATA/**/*.png', recursive=True)); print('png_files', len(fs), 'sizes', sorted({Image.open(f).size for f in fs}), 'modes', sorted({Image.open(f).mode for f in fs}))" 2>&1

echo "== infer (E1 infer.py, same arguments as run_pipeline.sh) =="
for dev in CPU GPU NPU; do
  echo "== infer padim batch1 $dev =="
  timeout 600 $PY -I $PROJ/app/anomaly/infer.py --model padim --ir $PADIM_IR --batch 1 --device $dev --data $DATA --out $V2/runs/infer-padim-$dev.json > $V2/runs/infer-padim-$dev.log 2>&1
  echo "rc=$?"
  echo "== infer patchcore batch2 $dev =="
  timeout 600 $PY -I $PROJ/app/anomaly/infer.py --model patchcore --ir $PATCH_IR --batch 2 --device $dev --data $DATA --out $V2/runs/infer-patchcore-$dev.json > $V2/runs/infer-patchcore-$dev.log 2>&1
  echo "rc=$?"
done

echo "== parity padim (PyTorch CPU vs IR on CPU, GPU, NPU) =="
timeout 900 $PY -I $V2/tools/parity.py --model padim --ckpt $PROJ/models/anomaly/padim/run/Padim/synthetic/v0/weights/lightning/model.ckpt --ir $PADIM_IR --batch 1 --data $DATA --out $V2/runs/parity-padim.json --devices CPU,GPU,NPU > $V2/runs/parity-padim.log 2>&1
echo "rc=$?"
echo "== parity patchcore (PyTorch CPU vs IR on CPU, GPU, NPU) =="
timeout 900 $PY -I $V2/tools/parity.py --model patchcore --ckpt $PROJ/models/anomaly/patchcore/run/Patchcore/synthetic/v0/weights/lightning/model.ckpt --ir $PATCH_IR --batch 2 --data $DATA --out $V2/runs/parity-patchcore.json --devices CPU,GPU,NPU > $V2/runs/parity-patchcore.log 2>&1
echo "rc=$?"

echo "== summary =="
$PY -I $V2/tools/summarize.py $V2/runs $V2/runs/summary.json > $V2/runs/summary.log 2>&1
echo "rc=$?"
echo VERIFY_ALL_DONE
