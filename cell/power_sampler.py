#!/usr/bin/env python3
"""power_sampler.py OUT.csv [LABEL]  - 2 Hz: RAPL pkg/psys W, pkg temp, NPU busy %, iGPU busy %, battery W.
Label changes live via:  echo npu_int8 > /tmp/pwr_label   (falls back to argv[2])."""
import csv, glob, os, sys, time
S, LABEL_FILE, DT = os.environ.get("SYSROOT", "/sys"), "/tmp/pwr_label", 0.5

def rd(p, conv=int):
    try:
        with open(p) as f: return conv(f.read().strip())
    except (OSError, ValueError): return None

def first(pattern, want=None):  # first glob match whose sibling 'name'/'type' == want
    for p in sorted(glob.glob(S + pattern)):
        if want is None or rd(os.path.join(os.path.dirname(p), "name" if "powercap" in p else "type"), str) == want: return p
rapl = {k: first("/class/powercap/intel-rapl:*/energy_uj", k) for k in ("package-0", "psys")}
wrap = {k: rd(os.path.join(os.path.dirname(p), "max_energy_range_uj")) for k, p in rapl.items() if p}
temp = first("/class/thermal/thermal_zone*/temp", "x86_pkg_temp")
npu = first("/class/accel/accel*/device/npu_busy_time_us")
gpu = first("/class/drm/card*/device/tile0/gt0/gtidle/idle_residency_ms")  # xe driver, render/compute GT
bat = first("/class/power_supply/BAT*/power_now")
for k, p in rapl.items():
    if p and rd(p) is None: sys.exit(f"{p} unreadable: sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj")
def snap(): return time.monotonic(), {k: rd(p) for k, p in rapl.items() if p}, rd(npu) if npu else None, rd(gpu) if gpu else None
with open(sys.argv[1], "a", newline="") as f:
    w = csv.writer(f)
    if f.tell() == 0: w.writerow(["unix_s", "label", "pkg_W", "psys_W", "pkg_C", "npu_busy_pct", "gpu_busy_pct", "bat_W"])
    t0, e0, n0, g0 = snap()
    while True:
        time.sleep(DT)
        t1, e1, n1, g1 = snap(); dt = t1 - t0
        W = {k: ((e1[k] - e0[k]) % (wrap[k] or 2**64)) / dt / 1e6 for k in e1 if e1[k] is not None and e0.get(k) is not None}
        npu_pct = None if n1 is None or n0 is None else 100 * (n1 - n0) / (dt * 1e6)
        gpu_pct = None if g1 is None or g0 is None else max(0.0, 100 - 100 * (g1 - g0) / (dt * 1e3))
        bw = rd(bat) / 1e6 if bat and rd(bat) is not None else None
        label = rd(LABEL_FILE, str) or (sys.argv[2] if len(sys.argv) > 2 else "")
        fmt = lambda v: "" if v is None else f"{v:.2f}"
        w.writerow([f"{time.time():.3f}", label, fmt(W.get("package-0")), fmt(W.get("psys")),
                    fmt(rd(temp) / 1000 if temp and rd(temp) else None), fmt(npu_pct), fmt(gpu_pct), fmt(bw)])
        f.flush(); t0, e0, n0, g0 = t1, e1, n1, g1
