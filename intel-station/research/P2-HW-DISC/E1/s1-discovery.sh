#!/bin/bash
# P2-HW-DISC E1 read-only discovery. Writes only inside /home/intel/intel-junction-2026/work/P2-HW-DISC/E1/
export PYTHONDONTWRITEBYTECODE=1
run() {
  printf '\n$ %s\n' "$1"
  sh -c "$1" 2>&1
  printf '[exit %s]\n' "$?"
}
echo "### START laptop UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "### STEP 1 system"
run 'uname -a'
run 'cat /etc/os-release'
run 'lscpu | head -20'
run 'free -h'
run 'df -h /home/intel'
run 'python3 --version'
run 'which uv git ffmpeg v4l2-ctl docker node npm'
run 'uv --version'
echo "### STEP 2 accelerators"
run "lspci -nn | grep -iE 'vga|display|processing|npu|accel'"
run 'ls -l /dev/dri /dev/accel 2>&1'
run "lsmod | grep -iE 'intel_vpu|xe|i915'"
run "cd /home/intel/intel-junction-2026/work/P2-HW-DISC/E1 && timeout 60 python3 -c 'import openvino as ov; print(ov.__version__, ov.Core().available_devices)' 2>&1"
echo "### STEP 3 running Studio (read-only; no UI port contact)"
run "ss -ltnp | grep -E ':3000|:7860'"
for p in $(ss -ltnp 2>/dev/null | grep -E ':3000|:7860' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
  run "ps -o pid,user,etime,cmd -p $p"
  run "readlink /proc/$p/cwd"
done
run 'curl -s -m 10 http://localhost:7860/api/health; echo'
echo "### STEP 4 USB and serial (listing only; serial ports not opened)"
run 'lsusb'
run 'ls -l /dev/serial/by-id /dev/ttyACM* /dev/ttyUSB* 2>&1'
run 'id'
echo "### STEP 5 cameras (restricted by orchestrator update: no video node opened or probed)"
run 'v4l2-ctl --list-devices'
run 'ls -l /dev/v4l/by-id 2>&1'
echo "SKIPPED: v4l2-ctl -d /dev/videoN --list-formats-ext (opens video nodes; excluded by orchestrator update)"
echo "SKIPPED: fuser -v /dev/video* busy check (video nodes excluded by orchestrator update)"
echo "### STEP 6 still capture: SKIPPED entirely (orchestrator update)"
echo "### END laptop UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
