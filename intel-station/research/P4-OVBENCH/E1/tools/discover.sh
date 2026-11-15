#!/bin/bash
# E1 P4-OVBENCH read-only discovery. Writes nothing outside the E1 folder.
R=/home/intel/intel-junction-2026
echo "== UTC $(date -u +%FT%TZ)"
echo "== id: $(id)"
echo "== Groups in /proc/self/status: $(grep '^Groups' /proc/self/status)"
echo "== kernel: $(uname -srm)"
echo "== os: $(grep PRETTY_NAME /etc/os-release)"
echo "== uptime: $(uptime)"
echo "== loadavg: $(cat /proc/loadavg)"
echo "== nproc: $(nproc)"
echo "== lscpu"
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|NUMA node0' | cut -c1-160
echo "== free"
free -m
echo "== disk R"
df -h "$R" | tail -1
echo "== dev nodes"
ls -l /dev/accel /dev/dri 2>&1
echo "== accel0 sysfs (depth 1 files)"
for f in $(find /sys/class/accel/accel0/device -maxdepth 1 -type f 2>/dev/null | sort); do echo "$f = $(timeout 2 cat "$f" 2>&1 | head -c 160 | tr '\n' ' ')"; done
echo "== drm sysfs freq/energy/power"
for f in $(find /sys/class/drm/card*/device -maxdepth 2 \( -name '*freq*' -o -name '*energy*' -o -name '*power*' \) -type f 2>/dev/null | sort); do echo "$f = $(timeout 2 cat "$f" 2>&1 | head -c 160 | tr '\n' ' ')"; done
echo "== hwmon"
for h in /sys/class/hwmon/hwmon*; do echo "$h name=$(cat $h/name 2>/dev/null)"; for f in $(ls $h 2>/dev/null | grep -E '^(energy|power)'); do echo "  $f = $(timeout 2 cat $h/$f 2>&1 | head -c 120)"; done; done
echo "== powercap"
ls -l /sys/class/powercap 2>&1
for f in /sys/class/powercap/*/energy_uj; do echo "$f = $(timeout 2 cat "$f" 2>&1)"; done
echo "== perf_event_paranoid = $(cat /proc/sys/kernel/perf_event_paranoid 2>&1)"
echo "== tools: $(command -v perf intel_gpu_top xpu-smi benchmark_app 2>/dev/null | tr '\n' ' ')"
echo "== device holders"
for d in /dev/accel/accel0 /dev/dri/renderD128; do echo "-- $d"; timeout 5 fuser -v "$d" 2>&1 | head -10; done
echo "== processes (top CPU; args cut; secrets filtered)"
ps -eo pid,user,pcpu,pmem,etime,args --sort=-pcpu | cut -c1-180 | grep -v -i -E 'token|secret|passw|api.?key' | head -30
echo "== envs"
ls -la "$R/envs" 2>&1 | head -40
find "$R/envs" -maxdepth 6 \( -type d -name openvino -o -type d -name cv2 -o -type f -name benchmark_app \) 2>/dev/null | head -40
find "$R/envs" -maxdepth 4 -path '*/bin/python3*' -type f 2>/dev/null | head -20
echo "== IR and onnx artefacts (app, models, work)"
find "$R/app" "$R/models" "$R/work" \( -name '*.xml' -o -name '*.onnx' \) -size +2k 2>/dev/null | sort | head -400
echo "== loop perception code"
grep -rl --include='*.py' -E 'close_pre_verdict|grasp_reachable|def blobs|handdet' "$R/app" "$R/docs" 2>/dev/null | sort | head -80
echo "== cycles.jsonl"
find "$R/logs" -name 'cycles.jsonl' 2>/dev/null | sort | head -60
echo "== END discover $(date -u +%FT%TZ)"
