#!/usr/bin/env python3
# VLA-PREP E1 read-only inventory (one output file: work/VLA-PREP/E1/inventory.json).
# Opens no camera, no serial port, no Studio socket, and nothing under the Studio data folder.
import csv, glob, hashlib, json, os, re, shutil, subprocess, sys, time

R = "/home/intel/intel-junction-2026"
W = R + "/work/VLA-PREP/E1"
OUT = W + "/inventory.json"
REDACT = [("/home/intel/junction-cell", "<friend>"), ("/home/intel/junction-handoff", "<friend>"),
          ("/home/intel/.local/share/physicalai", "<friend>"), ("/home/intel/Documents", "<home-docs>")]
DATASETS = ["a4ba6ede-092c-450f-8c76-559f66991dbd", "d2fd0a80-5364-4f38-bc81-d82d348c5a06",
            "ba9eafba-a9c3-47a0-b941-34c20a2bbf55", "e2eae6c8-fe98-4cfb-82eb-e5e59c7cc8af",
            "18468a09-4e62-4aa6-9c68-a785d5f0c3b9", "6d6bdce3-b382-478d-9697-fb9e573b97af",
            "51100357-35c1-4e37-a558-21a96d477651", "800d7dc2-17b0-49d1-b062-f39cf3acdbcc"]


def utc(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if t is None else t))


def redact(s):
    for a, b in REDACT:
        s = s.replace(a, b)
    return s


def run(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout + p.stderr
    except Exception as e:
        return "ERR " + repr(e)[:200]


def safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        return {"error": repr(e)[:300]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def grep_file(path, pattern, limit=40, width=200):
    rx = re.compile(pattern, re.I)
    out = []
    try:
        with open(path, errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if rx.search(line):
                    out.append("%d: %s" % (i, redact(line.rstrip())[:width]))
                    if len(out) >= limit:
                        break
    except OSError as e:
        out.append("ERR " + repr(e)[:200])
    return out


def grep_tree(root, pattern, limit=40, width=200, exts=(".py", ".sh", ".md", ".yaml", ".yml", ".txt", ".json")):
    if not os.path.isdir(root):
        return ["MISSING " + os.path.relpath(root, R)]
    rx = re.compile(pattern, re.I)
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ("__pycache__", ".git", "node_modules", ".venv", "frames", "crops")]
        for f in sorted(fn):
            p = os.path.join(dp, f)
            if not f.endswith(exts):
                continue
            try:
                if os.path.getsize(p) > 2000000:
                    continue
                with open(p, errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            out.append("%s:%d: %s" % (os.path.relpath(p, R), i, redact(line.rstrip())[:width]))
                            if len(out) >= limit:
                                return out
            except OSError:
                continue
    return out


def sec_studio_cmd():
    p = R + "/work/opus-verify/studio_backend_before_stop.txt"
    if not os.path.exists(p):
        return "MISSING"
    with open(p, errors="replace") as fh:
        return redact(fh.read())[:4000]


def sec_processes():
    rx = re.compile(r"intel-junction-2026|physical|uvicorn|python|ffmpeg|node|vite|sort_any|motion|hand_watch|"
                    r"wrist|record|monitor|telemetry|sync_|watch_|dashboard|loop", re.I)
    out = []
    for line in run(["ps", "-eo", "pid=,ppid=,etimes=,user=,args="], 10).splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5 and parts[3] == "intel" and rx.search(parts[4]):
            out.append({"pid": int(parts[0]), "ppid": int(parts[1]), "etimes_s": int(parts[2]),
                        "args": redact(parts[4])[:220]})
    return out


def sec_ports():
    return [redact(line)[:180] for line in run(["ss", "-ltnp"], 10).splitlines()[:60]]


def sec_devices():
    byid = "/dev/serial/by-id"
    return {
        "by_id": {n: os.path.realpath(os.path.join(byid, n)) for n in sorted(os.listdir(byid))}
        if os.path.isdir(byid) else {},
        "video": {os.path.basename(p): os.path.realpath(p) for p in sorted(glob.glob("/dev/video*"))},
        "ttyACM": sorted(glob.glob("/dev/ttyACM*")),
        "accel0": os.path.exists("/dev/accel/accel0"),
        "dri": sorted(os.listdir("/dev/dri")) if os.path.isdir("/dev/dri") else [],
    }


def sec_holders(devs):
    targets = set(devs["video"].values()) | set(devs["by_id"].values()) | set(devs["ttyACM"])
    hold = []
    for pid in sorted(p for p in os.listdir("/proc") if p.isdigit()):
        fd_dir = "/proc/%s/fd" % pid
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                tgt = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if tgt in targets:
                try:
                    comm = open("/proc/%s/comm" % pid).read().strip()
                except OSError:
                    comm = "?"
                hold.append({"pid": int(pid), "comm": comm, "device": tgt})
                break
    return hold


def sec_run_dir():
    flags, names = {}, []
    for p in sorted(glob.glob(R + "/run/*")):
        n = os.path.basename(p)
        if os.path.isfile(p) and (n.endswith(".pid") or n == "HASSAN_PRESENT"):
            with open(p, errors="replace") as fh:
                flags[n] = fh.read(80).strip()
        else:
            names.append(n)
    return {"flags_and_pid_files": flags, "other_names": names[:100]}


def sec_docs():
    return {
        "PICK-ANYWHERE.md": grep_file(R + "/docs/PICK-ANYWHERE.md",
                                      r"stop|resume|kill|pid|torque|hand_watch|recorder|start|port|sort_any", 60),
        "motion_src": grep_tree(R + "/app/motion", r"TORQUE|by-id|MAX_SPEED|MAX_ACC|max_speed|accel", 30),
    }


def sec_so101():
    root = R + "/repos/physicalai/src"
    if not os.path.isdir(root):
        return ["MISSING repos/physicalai/src"]
    out = []
    for dp, dn, fn in os.walk(root):
        for f in sorted(fn):
            p = os.path.join(dp, f)
            if not f.endswith(".py") or "so101" not in p.lower():
                continue
            with open(p, errors="replace") as fh:
                lines = fh.read().splitlines()
            for i, line in enumerate(lines):
                if re.search(r"def (dis)?connect\b|disable_torque|enable_torque|TORQUE_ENABLE", line):
                    out.append("%s:%d: %s" % (os.path.relpath(p, R), i + 1, redact(line.strip())[:160]))
                    if re.search(r"def disconnect\b", line):
                        out += ["    " + redact(x.rstrip())[:160] for x in lines[i + 1:i + 16]]
                    if len(out) >= 80:
                        return out
    return out or ["no so101 connect/disconnect/torque lines found"]


def sec_task_usage():
    base = R + "/repos/physicalai/src/physicalai/policies"
    return {"act": grep_tree(base + "/act", r"\btask\b", 12),
            "smolvla": grep_tree(base + "/smolvla", r"\btask\b", 12)}


def sec_envs():
    code = ("import sys,importlib.util as u;print(sys.version.split()[0]);"
            "print('openvino',bool(u.find_spec('openvino')));print('physicalai',bool(u.find_spec('physicalai')))")
    out = {}
    for py in sorted(glob.glob(R + "/envs/*/bin/python3")):
        out[os.path.relpath(py, R)] = run([py, "-I", "-c", code], 30).strip()[:300]
    return out


def sec_git():
    return {n: run(["git", "-C", R + "/" + n, "rev-parse", "--short", "HEAD"], 10).strip()[:40]
            for n in ["repos/physicalai", "app"]}


def sec_exports():
    found = []
    for root in [R + "/work", R + "/models", R + "/app"]:
        if not os.path.isdir(root):
            continue
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in ("frames", "crops", "logs", "raw", "node_modules", ".git",
                                                 "__pycache__", "P2-SAVE-KB")]
            if "manifest.json" in fn and "openvino" in dp:
                found.append(dp)
    out = []
    for dp in sorted(set(found)):
        e = {"dir": os.path.relpath(dp, R), "files": {}}
        for f in sorted(os.listdir(dp)):
            p = os.path.join(dp, f)
            if not os.path.isfile(p):
                continue
            e["files"][f] = {"bytes": os.path.getsize(p)}
            if f.endswith((".xml", ".bin")) or f == "manifest.json":
                e["files"][f]["sha256"] = sha256(p)
        try:
            with open(os.path.join(dp, "manifest.json")) as fh:
                man = json.load(fh)
            e["manifest_top_keys"] = list(man.keys())
            e["manifest_model"] = json.dumps(man.get("model", man))[:4000]
        except (OSError, ValueError) as ex:
            e["manifest_error"] = repr(ex)[:200]
        out.append(e)
    return out


def dataset_entry(ds, base, names, tool):
    d = os.path.join(base, ds)
    if not os.path.isdir(d):
        return {"present": False, "prefix_matches": [n for n in names if n.startswith(ds[:8])]}
    e = {"present": True, "meta_files": sorted(os.listdir(d + "/meta")) if os.path.isdir(d + "/meta") else []}
    info = d + "/meta/info.json"
    if os.path.exists(info):
        with open(info) as fh:
            j = json.load(fh)
        for k in ("codebase_version", "fps", "total_episodes", "total_frames", "total_tasks", "robot_type"):
            e[k] = j.get(k)
        e["feature_keys"] = sorted((j.get("features") or {}).keys())
    tp = d + "/meta/tasks.parquet"
    if os.path.exists(tp):
        code = ("import sys,json\ntry:\n import pyarrow.parquet as pq\n"
                " print(json.dumps(pq.read_table(sys.argv[1]).to_pylist(), default=str)[:900])\n"
                "except Exception as x:\n print('ERR', repr(x)[:200])")
        e["tasks_parquet"] = run([tool, "-I", "-c", code, tp], 60).strip()[:900]
    return e


def sec_datasets():
    base = R + "/data/studio-datasets"
    names = sorted(os.listdir(base)) if os.path.isdir(base) else []
    tool = R + "/envs/anomalib-env/bin/python3"
    if not os.path.exists(tool):
        tool = "python3"
    out = {"mirror_entries": len(names), "mirror_names_head": names[:12]}
    for ds in DATASETS:
        out[ds] = safe(dataset_entry, ds, base, names, tool)
    return out


def sec_episode(d):
    if not os.path.isdir(d):
        return "MISSING"
    files = sorted(os.listdir(d))
    e = {"n_files": len(files), "files_head": files[:60]}
    ep = d + "/episode.json"
    if os.path.exists(ep):
        with open(ep) as fh:
            j = json.load(fh)
        e["episode_json_top_keys"] = list(j.keys()) if isinstance(j, dict) else str(type(j))
        e["episode_json_head"] = json.dumps(j)[:1500]
    jc = d + "/joints.csv"
    if os.path.exists(jc):
        with open(jc) as fh:
            rows = list(csv.reader(fh))
        e["joints_header"] = rows[0] if rows else None
        body = rows[1:]
        e["joints_rows"] = len(body)
        if body:
            e["joints_first"] = body[0]
            e["joints_last"] = body[-1]
            try:
                t0, t1 = float(body[0][0]), float(body[-1][0])
                e["t_first_utc"], e["t_last_utc"] = utc(t0), utc(t1)
                e["duration_s"] = round(t1 - t0, 2)
            except ValueError:
                e["t_parse"] = "column 0 is not epoch seconds"
    return e


def sec_sort():
    base = R + "/logs/sort"
    names = sorted(os.listdir(base)) if os.path.isdir(base) else []
    out = {"count": len(names), "last_names": names[-12:]}
    for name in ["any_1740", "any_2055"]:
        out[name] = safe(sec_episode, base + "/" + name)
    return out


def sec_monitor():
    base = R + "/logs/monitor"
    names = sorted(os.listdir(base)) if os.path.isdir(base) else []
    out = []
    for n in names:
        if not n.endswith(".mkv") or n.split("_", 1)[-1] < "20261114T160000Z.mkv":
            continue
        p = base + "/" + n
        probe = run(["ffprobe", "-v", "error", "-of", "json", "-show_entries",
                     "format=start_time,duration:stream=codec_name,width,height,r_frame_rate", p], 30)
        try:
            pj = json.loads(probe)
        except ValueError:
            pj = {"raw": probe[:200]}
        out.append({"file": n, "bytes": os.path.getsize(p), "mtime_utc": utc(os.path.getmtime(p)), "probe": pj})
    return out[:40]


def sec_cam_map():
    pat = r"243622061421|243722064316|video8|video14|video16|USB2\.0_CAM1|usb2_0_cam1"
    out = []
    for sub in ["tools", "motion", "loop", "ovrun", "anomaly"]:
        out += grep_tree(R + "/app/" + sub, pat, 12)
    return out[:60]


def sec_loop_log():
    files = glob.glob(R + "/logs/sort_any_any_*.out")
    if not files:
        return "none"
    newest = max(files, key=os.path.getmtime)
    with open(newest, errors="replace") as fh:
        tail = fh.read().splitlines()[-4:]
    return {"file": os.path.basename(newest), "mtime_utc": utc(os.path.getmtime(newest)),
            "tail": [redact(x)[:200] for x in tail]}


def main():
    t0 = time.time()
    devs = safe(sec_devices)
    res = {
        "utc": utc(), "script_sha256": sha256(__file__), "nproc": os.cpu_count(),
        "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe"),
        "studio_backend_before_stop": safe(sec_studio_cmd),
        "processes": safe(sec_processes),
        "ports": safe(sec_ports),
        "devices": devs,
        "device_holders": safe(sec_holders, devs) if isinstance(devs, dict) and "video" in devs else "n/a",
        "run_dir": safe(sec_run_dir),
        "docs": safe(sec_docs),
        "so101_source": safe(sec_so101),
        "task_usage_policy_source": safe(sec_task_usage),
        "envs": safe(sec_envs),
        "git": safe(sec_git),
        "exports": safe(sec_exports),
        "datasets_mirror": safe(sec_datasets),
        "sort_episodes": safe(sec_sort),
        "monitor_mkv": safe(sec_monitor),
        "camera_mapping_grep": safe(sec_cam_map),
        "loop_log": safe(sec_loop_log),
    }
    res["elapsed_s"] = round(time.time() - t0, 2)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    os.replace(tmp, OUT)
    print("WROTE", OUT, os.path.getsize(OUT), "bytes; sha256", sha256(OUT), "; elapsed", res["elapsed_s"], "s")


if __name__ == "__main__":
    main()
