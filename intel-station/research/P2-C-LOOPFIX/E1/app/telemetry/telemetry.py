#!/usr/bin/env python3
"""Light 1 Hz telemetry sampler and reader for the HP OmniBook Ultra 14 (no root).

Sources are sysfs and procfs files readable by the unprivileged user. RAPL package
energy is used only when energy_uj is readable; on this laptop it is root-only, so
pkg_w stays null and the reader reports energy as null.

Usage: python3 -I telemetry.py run | once | latest | rolling [SECONDS] | window T0 T1
"""
import glob
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

R = "/home/intel/intel-junction-2026"
LOG_DIR = os.path.join(R, "logs", "telemetry")
PID_FILE = os.path.join(LOG_DIR, "telemetry.pid")
PERIOD_S = 1.0


def read(path):
    if not path:
        return None
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, ValueError):
        return None


def read_int(path):
    v = read(path)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def discover():
    src = {"pkg_temp": None, "core_pkg": None, "core_inputs": [], "nvme": None}
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        if read(z + "/type") == "x86_pkg_temp":
            src["pkg_temp"] = z + "/temp"
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = read(h + "/name")
        if name == "coretemp":
            for lab in sorted(glob.glob(h + "/temp*_label")):
                label = read(lab) or ""
                inp = lab[: -len("_label")] + "_input"
                if label.startswith("Package id"):
                    src["core_pkg"] = inp
                elif label.startswith("Core"):
                    src["core_inputs"].append(inp)
        elif name == "nvme" and os.path.exists(h + "/temp1_input"):
            src["nvme"] = h + "/temp1_input"
    gt0 = sorted(glob.glob("/sys/class/drm/card*/device/tile*/gt0/freq0"))
    gt1 = sorted(glob.glob("/sys/class/drm/card*/device/tile*/gt1/freq0"))
    src["gpu_act"] = gt0[0] + "/act_freq" if gt0 else None
    src["gpu_max"] = gt0[0] + "/max_freq" if gt0 else None
    src["gpu_media_act"] = gt1[0] + "/act_freq" if gt1 else None
    npu = "/sys/class/accel/accel0/device"
    src["npu_busy_us"] = npu + "/npu_busy_time_us"
    src["npu_freq"] = npu + "/npu_current_frequency_mhz"
    bats = sorted(glob.glob("/sys/class/power_supply/BAT*"))
    src["bat"] = bats[0] if bats else None
    acs = sorted(glob.glob("/sys/class/power_supply/ADP*")) + sorted(glob.glob("/sys/class/power_supply/AC*"))
    src["ac"] = acs[0] if acs else None
    src["rapl_energy"] = None
    src["rapl_max"] = None
    for z in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
        if read(z + "/name") == "package-0":
            src["rapl_energy"] = z + "/energy_uj"
            src["rapl_max"] = z + "/max_energy_range_uj"
    return src


def read_cpu():
    with open("/proc/stat") as f:
        vals = [int(x) for x in f.readline().split()[1:9]]
    idle = vals[3] + vals[4]
    total = sum(vals)
    return total - idle, total


def read_mem():
    kv = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, rest = line.partition(":")
            kv[k] = int(rest.split()[0])
    return round(100.0 * (1 - kv["MemAvailable"] / kv["MemTotal"]), 2), round(kv["MemAvailable"] / 1024.0, 1)


def c_of(path):
    v = read_int(path)
    return round(v / 1000.0, 2) if v is not None else None


def sample(src, prev):
    now = time.time()
    s = {"t": iso(now), "epoch": round(now, 3)}
    pkg_w = None
    e_now = read_int(src["rapl_energy"])
    if e_now is not None:
        if "rapl" in prev:
            e_prev, t_prev = prev["rapl"]
            de = e_now - e_prev
            if de < 0:
                de += read_int(src["rapl_max"]) or 0
            if now > t_prev:
                pkg_w = round(de / 1e6 / (now - t_prev), 3)
        prev["rapl"] = (e_now, now)
    s["pkg_w"] = pkg_w
    temp = {"pkg": c_of(src["pkg_temp"]), "coretemp_pkg": c_of(src["core_pkg"]), "nvme": c_of(src["nvme"])}
    cores = [c_of(p) for p in src["core_inputs"]]
    cores = [c for c in cores if c is not None]
    temp["core_max"] = max(cores) if cores else None
    s["temp_c"] = temp
    busy, total = read_cpu()
    s["cpu_util"] = round((busy - prev["cpu"][0]) / max(1, total - prev["cpu"][1]), 4) if "cpu" in prev else None
    prev["cpu"] = (busy, total)
    act = read_int(src["gpu_act"])
    mx = read_int(src["gpu_max"])
    s["gpu"] = {
        "freq_mhz": act,
        "max_mhz": mx,
        "freq_ratio": round(act / mx, 4) if act is not None and mx else None,
        "media_mhz": read_int(src["gpu_media_act"]),
    }
    nb = read_int(src["npu_busy_us"])
    npu_busy = None
    if nb is not None:
        if "npu" in prev and now > prev["npu"][1]:
            npu_busy = round((nb - prev["npu"][0]) / ((now - prev["npu"][1]) * 1e6), 4)
        prev["npu"] = (nb, now)
    s["npu"] = {"busy": npu_busy, "freq_mhz": read_int(src["npu_freq"])}
    used, avail = read_mem()
    s["mem"] = {"used_pct": used, "avail_mb": avail}
    b = src["bat"]
    pn = read_int(b + "/power_now") if b else None
    s["bat"] = {
        "status": read(b + "/status") if b else None,
        "power_w": round(pn / 1e6, 3) if pn is not None else None,
        "capacity": read_int(b + "/capacity") if b else None,
    }
    s["ac_online"] = read_int(src["ac"] + "/online") if src["ac"] else None
    s["src"] = "pkg_w:" + ("rapl" if e_now is not None else "none") + ";temp:x86_pkg+coretemp;cpu:proc_stat;gpu:xe_freq_ratio;npu:accel0_busy_us"
    return s


def run():
    os.makedirs(LOG_DIR, exist_ok=True)
    src = discover()
    prev = {}
    stop = {"flag": False}

    def on_signal(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    day, fh = None, None
    next_t = time.monotonic()
    try:
        while not stop["flag"]:
            d = datetime.now(timezone.utc).strftime("%Y%m%d")
            if d != day:
                if fh:
                    fh.close()
                fh = open(os.path.join(LOG_DIR, d + ".jsonl"), "a", buffering=1)
                day = d
            fh.write(json.dumps(sample(src, prev), separators=(",", ":")) + "\n")
            next_t += PERIOD_S
            delay = next_t - time.monotonic()
            if delay < 0:
                next_t = time.monotonic()
                delay = 0
            time.sleep(delay)
    finally:
        if fh:
            fh.close()
        if read(PID_FILE) == str(os.getpid()):
            os.remove(PID_FILE)


def _rows_between(t0, t1):
    rows = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                e = r.get("epoch")
                if e is not None and t0 <= e <= t1:
                    rows.append(r)
    return rows


def _rows_last(seconds):
    now = time.time()
    rows = []
    files = sorted(glob.glob(os.path.join(LOG_DIR, "*.jsonl")))
    for path in reversed(files[-2:]):
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - 400000))
            lines = f.read().decode("utf-8", "replace").splitlines()
        for line in reversed(lines):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            e = r.get("epoch")
            if e is None:
                continue
            if e < now - seconds:
                return sorted(rows, key=lambda x: x["epoch"])
            rows.append(r)
    return sorted(rows, key=lambda x: x["epoch"])


def _dig(r, *keys):
    for k in keys:
        r = r.get(k) if isinstance(r, dict) else None
    return r


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def stats(rows, t0, t1):
    pk = [_dig(r, "pkg_w") for r in rows]
    pkg_t = [_dig(r, "temp_c", "pkg") for r in rows]
    pkg_t = [v for v in pkg_t if v is not None]
    span = round(t1 - t0, 3)
    pkg_mean = _mean(pk)
    bat_mean = _mean([_dig(r, "bat", "power_w") for r in rows])
    return {
        "t0": round(t0, 3),
        "t1": round(t1, 3),
        "span_s": span,
        "samples": len(rows),
        "pkg_w_mean": pkg_mean,
        "energy_j_pkg": round(pkg_mean * span, 2) if pkg_mean is not None else None,
        "pkg_temp_c_mean": _mean(pkg_t),
        "pkg_temp_c_max": max(pkg_t) if pkg_t else None,
        "core_max_c_mean": _mean([_dig(r, "temp_c", "core_max") for r in rows]),
        "cpu_util_mean": _mean([_dig(r, "cpu_util") for r in rows]),
        "gpu_freq_mhz_mean": _mean([_dig(r, "gpu", "freq_mhz") for r in rows]),
        "gpu_freq_ratio_mean": _mean([_dig(r, "gpu", "freq_ratio") for r in rows]),
        "npu_busy_mean": _mean([_dig(r, "npu", "busy") for r in rows]),
        "npu_freq_mhz_mean": _mean([_dig(r, "npu", "freq_mhz") for r in rows]),
        "bat_power_w_mean": bat_mean,
        "energy_j_bat": round(bat_mean * span, 2) if bat_mean is not None else None,
        "ac_online_values": sorted({_dig(r, "ac_online") for r in rows if _dig(r, "ac_online") is not None}),
    }


def latest():
    rows = _rows_last(120)
    if not rows:
        return {"status": "no samples"}
    last = rows[-1]
    last["age_s"] = round(time.time() - last["epoch"], 3)
    return last


def rolling(seconds=60.0):
    now = time.time()
    return stats(_rows_last(seconds), now - seconds, now)


def window(t0, t1):
    return stats(_rows_between(t0, t1), t0, t1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "latest"
    if cmd == "run":
        run()
    elif cmd == "once":
        print(json.dumps(sample(discover(), {}), indent=1))
    elif cmd == "latest":
        print(json.dumps(latest(), indent=1))
    elif cmd == "rolling":
        print(json.dumps(rolling(float(sys.argv[2]) if len(sys.argv) > 2 else 60.0), indent=1))
    elif cmd == "window":
        print(json.dumps(window(float(sys.argv[2]), float(sys.argv[3])), indent=1))
    else:
        print("usage: telemetry.py run | once | latest | rolling [SECONDS] | window T0 T1")
