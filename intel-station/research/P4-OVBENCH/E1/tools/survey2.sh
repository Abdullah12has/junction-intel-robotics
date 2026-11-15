#!/bin/bash
# E1 P4-OVBENCH read-only survey 2: NPU/GPU counters (find -L), artefact dirs, loop code map, OpenVINO probe.
R=/home/intel/intel-junction-2026
echo "== UTC $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"
echo "== NPU sysfs (find -L, depth 1 files)"
for f in $(find -L /sys/class/accel/accel0/device -maxdepth 1 -type f 2>/dev/null | sort); do echo "$f = $(timeout 2 cat "$f" 2>&1 | head -c 120 | tr '\n' ' ')"; done
echo "== DRM card1 freq/energy/power (find -L)"
for f in $(find -L /sys/class/drm/card1/device -maxdepth 3 \( -name '*freq*' -o -name '*energy*' -o -name '*power*' \) -type f 2>/dev/null | sort | head -60); do echo "$f = $(timeout 2 cat "$f" 2>&1 | head -c 120 | tr '\n' ' ')"; done
echo "== intel_gpu_top non-root probe"
timeout 8 intel_gpu_top -J -s 500 -n 1 2>&1 | head -5
echo "== perf power probe non-root"
timeout 8 perf stat -a -e power/energy-pkg/ -- sleep 0.5 2>&1 | tail -5
echo "== artefact dirs"
for d in "$R/app/anomaly/anom3/models/v000/pre_close" "$R/models/anomaly/patchcore/weights/openvino" "$R/models/anomaly/padim/weights/openvino" "$R/models/anomaly/padim/weights/openvino_b1" "$R/work/P2-B-OPT/E1/models/389da0fd_fp16" "$R/work/P2-B-OPT/E1/models/389da0fd" "$R/work/P2-B-OVRUN/E1/models/389da0fd/openvino" "$R/work/P2-B-DET2/E3/anom/openvino_cpu/weights/openvino"; do echo "-- $d"; ls -la "$d" 2>&1 | head -12; done
echo "== metadata heads"
for f in "$R/app/anomaly/anom3/models/v000/pre_close/metadata.json" "$R/models/anomaly/patchcore/weights/openvino/metadata.json"; do echo "-- $f"; head -c 900 "$f" 2>&1; echo; done
echo "== VLA artefacts"
ls -la "$R/app/vla" 2>&1 | head -12
find "$R/models" "$R/work" "$R/app" -maxdepth 4 \( -iname '*vla*' -o -iname '*smol*' \) 2>/dev/null | head -30
echo "== device settings (loop, anomaly code)"
grep -rn -E "AUTO|'(CPU|GPU|NPU)'|\"(CPU|GPU|NPU)\"" "$R/app/tools" "$R/app/anomaly" --include='*.py' --include='*.yaml' --include='*.json' 2>/dev/null | cut -c1-200 | head -60
echo "== envs referenced by loop code"
grep -rn "envs/" "$R/app/tools" "$R/app/anomaly" --include='*.py' --include='*.sh' 2>/dev/null | cut -c1-200 | head -20
echo "== sort_any.py defs"
grep -n -E '^\s*(def|class) ' "$R/app/tools/sort_any.py" | head -120
echo "== sort_any.py top-level statements (import side effects)"
grep -n -v -E '^(def |class |import |from |#|\s|$|\)|\]|\})' "$R/app/tools/sort_any.py" | cut -c1-160 | head -80
echo "== handdet.py defs"
grep -n -E '^\s*(def|class) ' "$R/app/tools/handdet.py" | head -40
echo "== key names in sort_any.py"
grep -n -E 'close_pre_verdict|grasp_reachable|cycles\.jsonl|\.xml|openvino|OpenVINO|blobs|handdet|servo' "$R/app/tools/sort_any.py" | cut -c1-200 | head -100
echo "== cycles.jsonl head and sizes"
head -n 2 "$R/logs/sort/any_1731/cycles.jsonl" | cut -c1-1200
wc -l "$R/logs/sort/any_1731/cycles.jsonl" "$R/logs/sort/any_1740/cycles.jsonl"
echo "== any_1731 listing (first 40)"
ls "$R/logs/sort/any_1731" | head -40
echo "== OpenVINO probe (ovrun-env, anomalib-env)"
for E in ovrun-env anomalib-env; do echo "-- $E"; timeout 90 "$R/envs/$E/bin/python" -I -c "import openvino as ov, cv2, numpy as np, sys; c=ov.Core(); print('py', sys.version.split()[0], 'ov', ov.__version__, 'cv2', cv2.__version__, 'np', np.__version__); print('devices', c.available_devices); [print(d, c.get_property(d, 'FULL_DEVICE_NAME')) for d in c.available_devices]" 2>&1 | tail -8; done
echo "== END survey2 $(date -u +%FT%TZ)"
