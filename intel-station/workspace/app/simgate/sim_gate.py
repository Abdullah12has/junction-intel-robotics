#!/usr/bin/env python3
"""Sim gate: GO/NO-GO per use for one version of the team's simulation data.

Reads sim data as data only (json, npz, mp4, text of source files). Never imports
or runs anything from the sim folder (R6 / untrusted-source rule).
Usage: python3 -I sim_gate.py --version-root DIR [--convert]
Writes DIR-independent results to data/sim/gate/<version>.json and one line to logs/sim_gate.log.
"""
import argparse, ast, glob, hashlib, json, math, os, re, subprocess, time
import xml.etree.ElementTree as ET
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

R = "/home/intel/intel-junction-2026"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TP = 2 * math.pi
TOL = {"pan_deg": 3.0, "lift_elbow_units": 3.0, "gripper_ratio": 0.10, "gripper_offset_units": 1.0,
       "rest_frac_min": 0.02, "wrist_pinned_max": 0.01, "brightness_rel": 0.15, "cam_px": 10.0,
       "release_min": 0.98, "episodes_min": 0.95, "geom_cm": 2.0, "fk_cm": 2.5}
REAL_SETS = {"blue": [7, 11, 12], "green": [8, 10, 14, 15]}   # real plate releases (explicit; see verifier notes)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_text(p):
    return open(p, encoding="utf-8", errors="replace").read() if os.path.isfile(p) else ""


# ---------- provenance: manifest of unknown shape, verified by hashing the files it lists
def collect_pairs(obj, out):
    if isinstance(obj, dict):
        if isinstance(obj.get("sha256"), str) and any(k in obj for k in ("path", "file", "name")):
            out.append((obj.get("path") or obj.get("file") or obj.get("name"), obj["sha256"].lower()))
            return
        for k, v in obj.items():
            if isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F]{64}", v):
                out.append((k, v.lower()))
            else:
                collect_pairs(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_pairs(v, out)


def provenance(root):
    man = os.path.join(root, "SHA256.json")
    if not os.path.isfile(man):
        return {"manifest": None, "entries": 0, "verified": False, "note": "no manifest"}
    pairs = []
    collect_pairs(json.load(open(man)), pairs)
    ok = bad = miss = 0
    bad_list = []
    for p, h in pairs:
        fp = os.path.join(root, p) if p else None
        if not fp or not os.path.isfile(fp):
            miss += 1
            continue
        if sha256_file(fp) == h:
            ok += 1
        else:
            bad += 1
            bad_list.append(p)
    return {"manifest": man, "entries": len(pairs), "ok": ok, "mismatch": bad, "missing": miss,
            "mismatch_examples": bad_list[:5],
            "verified": len(pairs) > 0 and bad == 0 and miss == 0,
            "note": "the manifest comes from the same source as the data: a match proves only consistency"}


# ---------- MJCF forward kinematics (text parse of the pinned so101.xml; no sim code run)
def quat_mat(q):
    q = np.asarray(q, float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]])


def axis_mat(a, ang):
    a = np.asarray(a, float); a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = math.cos(ang), math.sin(ang); C = 1 - c
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


def tf(p, Rm):
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = p
    return T


def pq4(el):
    pos = np.array([float(v) for v in el.get("pos", "0 0 0").split()])
    quat = np.array([float(v) for v in el.get("quat", "1 0 0 0").split()])
    return pos, quat


def load_mjcf(path):
    root = ET.parse(path).getroot()
    base = root.find(".//body[@name='base']")
    lim = {j.get("name"): [float(v) for v in j.get("range").split()]
           for j in root.iter("joint") if j.get("range") and j.get("name")}
    return base, lim


def fk_sites(base_el, q):
    out = {}
    def walk(el, parent):
        pos, quat = pq4(el)
        T = parent @ tf(pos, quat_mat(quat))
        for j in el.findall("joint"):
            nm = j.get("name")
            if nm in q:
                ax = [float(v) for v in j.get("axis", "0 0 1").split()]
                T = T @ tf(np.zeros(3), axis_mat(ax, q[nm]))
        for s in el.findall("site"):
            sp, sq = pq4(s)
            out[s.get("name")] = T @ tf(sp, quat_mat(sq))
        for c in el.findall("body"):
            walk(c, T)
    walk(base_el, np.eye(4))
    return out


def station_T(st_txt):
    m1 = re.search(r'base\.set\("quat",\s*"([^"]+)"\)', st_txt)
    m2 = re.search(r'base\.set\("pos",\s*"([^"]+)"\)', st_txt)
    if not (m1 and m2):
        return None
    q = [float(v) for v in m1.group(1).split()]
    p = [float(v) for v in m2.group(1).split()]
    return tf(p, quat_mat(q))


# ---------- media (ffmpeg decode to rgb24, 640x480, 1 fps)
def frames_rgb(path, dur=None, maxn=None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-vf", "fps=1,scale=640:480", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    a = np.frombuffer(p.stdout, np.uint8).reshape(-1, 480, 640, 3)
    return a[:maxn] if maxn else a


def luma_stats(a):
    y = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    return {"frames": int(len(a)), "mean_luma": round(float(y.mean()), 2),
            "p2": round(float(np.percentile(y, 2)), 1), "p98": round(float(np.percentile(y, 98)), 1)}


def green_centroid(a):
    r = a[..., 0].astype(np.int16); g = a[..., 1].astype(np.int16); b = a[..., 2].astype(np.int16)
    m = (g > r + 35) & (g > b + 25) & (g > 50)
    pts = []
    for i in range(len(m)):
        ys, xs = np.nonzero(m[i])
        if len(xs) >= 200:
            pts.append((float(xs.mean()), float(ys.mean()), float(len(xs))))
    if not pts:
        return None
    arr = np.array(pts)
    return {"frames_with_plate": len(pts), "x_px": round(float(np.median(arr[:, 0])), 1),
            "y_px": round(float(np.median(arr[:, 1])), 1), "mask_px_median": float(np.median(arr[:, 2]))}


# ---------- real dataset (6d6bdce3): state in Studio normalized units
def load_real(real):
    S, E, F = [], [], []
    for f in sorted(glob.glob(real + "/data/chunk-*/file-*.parquet")):
        tb = pq.read_table(f, columns=["observation.state", "episode_index", "frame_index"]).to_pydict()
        S += tb["observation.state"]; E += tb["episode_index"]; F += tb["frame_index"]
    return np.array(S, float), np.array(E), np.array(F)


def real_release(S, E, F, e):
    m = np.where(E == e)[0]
    m = m[np.argsort(F[m])]
    st = S[m]; g = st[:, 5]
    tmin = int(np.argmin(g))
    r = g[tmin + 5:] - g[tmin:-5]
    k = int(np.argmax(r))
    return st[tmin + k, :5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version-root", required=True)
    ap.add_argument("--out-dir", default=R + "/data/sim/gate")
    ap.add_argument("--log", default=R + "/logs/sim_gate.log")
    ap.add_argument("--real", default=R + "/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af")
    ap.add_argument("--robots", default=R + "/app/motion/project_robots.json")
    ap.add_argument("--mjcf", default=R + "/app/simgate/ref/so101.xml")
    ap.add_argument("--baseline-raw", default=R + "/data/sim/2026-11-14/dataset-150-raw")
    ap.add_argument("--convert", action="store_true", help="write a Studio-units parquet copy if a training use is GO")
    ap.add_argument("--max-attempts", type=int, default=150)
    args = ap.parse_args()

    root = args.version_root.rstrip("/")
    res = {"utc": utc(), "root": root, "errors": [], "checks": {}}
    raws = sorted(glob.glob(root + "/dataset-*-raw"))
    raw = raws[0] if raws else root
    attempts = sorted(glob.glob(raw + "/attempt-*"))[:args.max_attempts]
    sig = hashlib.sha1()
    for d in attempts:
        for f in ("metadata.json", "trajectory.npz", "trace.json", "overhead.mp4", "wrist.mp4"):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                sig.update(("%s/%s:%d" % (os.path.basename(d), f, os.path.getsize(p))).encode())
    version = "%s-%s" % (os.path.basename(root), sig.hexdigest()[:10])
    res["version"] = version
    res["provenance"] = provenance(root)

    # source text (parsed, not executed)
    st_txt = read_text(raw + "/source/station_scene.py")
    sv_txt = read_text(raw + "/source/sorting_env.py")
    rc_path = raw + "/recording_context.json"
    rc = json.load(open(rc_path)) if os.path.isfile(rc_path) else {}
    base_rc = json.load(open(args.baseline_raw + "/recording_context.json")) if os.path.isfile(args.baseline_raw + "/recording_context.json") else {}
    res["source"] = {
        "self_hash_ok": {k: (os.path.isfile(raw + "/source/" + k) and sha256_file(raw + "/source/" + k) == h)
                         for k, h in rc.get("source_hashes", {}).items()},
        "changed_vs_2026-11-14": [k for k, h in base_rc.get("source_hashes", {}).items()
                                  if rc.get("source_hashes", {}).get(k) != h],
        "tape_or_defect_word_in_source": bool(re.search(r"tape", sv_txt + st_txt, re.I)),
    }
    try:
        m = re.search(r"BOWL_CENTERS\s*=\s*(\{.*?\})", st_txt, re.S)
        bowl = ast.literal_eval(m.group(1))
        res["source"]["plate_centres_world"] = bowl
    except Exception as ex:
        bowl = {}; res["errors"].append("BOWL_CENTERS: %s" % ex)
    Tw = station_T(st_txt)
    res["source"]["station_base_T_world"] = None if Tw is None else Tw.round(6).tolist()
    m = re.search(r"BLOCK_TYPES\s*=\s*(\[.*?\])", sv_txt, re.S)
    res["source"]["block_types"] = sorted(set(ast.literal_eval(m.group(1)))) if m else None

    # reference model (MJCF, pinned Menagerie so101.xml) and calibration
    base_el, lim = load_mjcf(args.mjcf)
    res["reference"] = {"mjcf_base_pos_quat": [pq4(base_el)[0].tolist(), pq4(base_el)[1].tolist()],
                        "wrist_flex_limit_rad": lim["wrist_flex"][1], "elbow_limit_rad": lim["elbow_flex"][1],
                        "gripper_range_rad": lim["gripper"]}
    rob = json.load(open(args.robots))
    fol = [r for r in rob if r.get("type") == "SO101_Follower"][0]
    cal = fol["payload"]["calibration"]
    rmin = np.array([cal[j]["range_min"] for j in JOINTS], float)
    rmax = np.array([cal[j]["range_max"] for j in JOINTS], float)
    span = rmax - rmin
    K = (200 * 4095 / (TP * span))[:5]          # zero at the recorded range midpoint, sign +1
    K_grip_enc = 100 * 4095 / (TP * span[5])    # encoder-linear gripper scale (assumption)
    res["reference"]["K_mid_per_rad"] = K.round(3).tolist()

    # real data
    S, E, F = load_real(args.real)
    rest_mask = (S[:, 1] <= -90) & (S[:, 2] >= 90)
    rest_ref = np.median(S[rest_mask, :5], axis=0)
    real_rel = {}
    for p, eps in REAL_SETS.items():
        real_rel[p] = np.array([real_release(S, E, F, e) for e in eps])
    real_pan = {p: np.median(real_rel[p][:, 0]) for p in real_rel}
    real_pan_sd = {p: float(np.std(real_rel[p][:, 0], ddof=1)) for p in real_rel}
    real_gr_open = float(np.median([S[E == e, 5].max() for e in np.unique(E)]))
    real_gr_closed = float(np.median([S[E == e, 5].min() for e in np.unique(E)]))
    res["real"] = {"rest_ref_norm": rest_ref.round(2).tolist(), "release_pan_norm_median": {k: round(float(v), 2) for k, v in real_pan.items()},
                   "release_pan_sd": {k: round(v, 2) for k, v in real_pan_sd.items()},
                   "gripper_open_units": round(real_gr_open, 2), "gripper_closed_units": round(real_gr_closed, 2)}

    # sim episodes (data only)
    eps = []
    for d in attempts:
        if not os.path.isfile(d + "/metadata.json"):
            continue
        md = json.load(open(d + "/metadata.json"))
        z = np.load(d + "/trajectory.npz")
        st = z["state"].astype(float)
        tr = json.load(open(d + "/trace.json"))
        obj = np.array([t["object_xyz"] for t in tr], float)
        n = min(len(st), len(obj))
        eps.append({"name": os.path.basename(d), "dir": d, "md": md, "st": st[:n], "obj": obj[:n]})
    units = (eps[0]["md"].get("action_units") if eps else None) or ""
    res["units_field"] = units
    rad = "radian" in units
    norm_ok = "normalized" in units
    inv = {"episodes": len(eps)}
    if eps:
        inv["accepted"] = sum(bool(e["md"].get("accepted")) for e in eps)
        inv["success"] = sum(bool(e["md"].get("success")) for e in eps)
        inv["grasp_verified"] = sum(bool(e["md"].get("physical_grasp_verified")) for e in eps)
        inv["task_text_matches"] = sum(e["md"].get("task") == "move the %s brick to the %s plate" % (e["md"].get("target_color"), e["md"].get("destination")) for e in eps)
        bt = [e["md"].get("brick_type") for e in eps]
        dest = [e["md"].get("destination") for e in eps]
        inv["brick_type_counts"] = {k: bt.count(k) for k in sorted(set(map(str, bt)))}
        inv["destination_counts"] = {k: dest.count(k) for k in sorted(set(map(str, dest)))}
        inv["od10_contradictions"] = int(sum((b == "2x2" and d == "green") or (b != "2x2" and d == "blue") for b, d in zip(bt, dest)))
        errs = []
        for e in eps:
            fx = e["md"]["final_object_xyz"]
            c = bowl.get(e["md"]["destination"], (np.nan, np.nan))
            errs.append(math.hypot(fx[0] - c[0], fx[1] - c[1]))
        inv["final_xy_err_m_max"] = round(float(np.nanmax(errs)), 4) if errs else None
    res["inventory"] = inv

    # units, gripper, rest, wrist (sim, mapped to Studio normalized units with the validated convention)
    try:
        allq = np.concatenate([e["st"][:, :5] for e in eps])
        allg = np.concatenate([e["st"][:, 5] for e in eps])
        if rad:
            n5 = allq * K
            q5 = allq
        elif norm_ok:
            n5 = allq
            q5 = allq / K
        else:
            n5 = None; q5 = None
        if n5 is None:
            res["errors"].append("units not recognised: '%s'" % units)
        else:
            rest_frac = float(np.mean((n5[:, 1] <= -90) & (n5[:, 2] >= 90)))
            near_rest = float(np.mean((np.abs(n5[:, 1] - rest_ref[1]) <= 10) & (np.abs(n5[:, 2] - rest_ref[2]) <= 10)))
            wrist_pinned = float(np.mean(q5[:, 3] >= lim["wrist_flex"][1] - 0.002))
            res["checks"]["units"] = {"sim_min_norm": n5.min(axis=0).round(2).tolist(), "sim_max_norm": n5.max(axis=0).round(2).tolist(),
                                      "real_min_norm": S[:, :5].min(axis=0).round(2).tolist(), "real_max_norm": S[:, :5].max(axis=0).round(2).tolist(),
                                      "rest_like_frac_sim": round(rest_frac, 4), "near_real_rest_frac_sim": round(near_rest, 4),
                                      "wrist_pinned_frac_sim": round(wrist_pinned, 4)}
    except Exception as ex:
        res["errors"].append("units: %s" % ex); n5 = q5 = None; allg = None

    # gripper scale against real (open-minus-closed span, and the closed offset)
    try:
        sim_open = float(np.median([e["st"][:, 5].max() for e in eps]))
        sim_closed = []
        for e in eps:
            lift = np.where(e["obj"][:, 2] > 0.03)[0]
            if len(lift):
                sim_closed.append(float(e["st"][int(lift[0]), 5]))
        sim_closed = float(np.median(sim_closed)) if sim_closed else float("nan")
        scale = 1.0 if norm_ok else K_grip_enc
        sim_span = (sim_open - sim_closed) * scale
        real_span = real_gr_open - real_gr_closed
        ratio = sim_span / real_span if real_span else float("nan")
        offset = sim_closed * scale - real_gr_closed
        res["checks"]["gripper"] = {"sim_open": round(sim_open, 4), "sim_closed_on_brick": round(sim_closed, 4), "scale_used": round(scale, 3),
                                    "span_ratio_sim_over_real": round(ratio, 3), "closed_offset_units": round(offset, 2)}
    except Exception as ex:
        res["errors"].append("gripper: %s" % ex)

    # releases, geometry (MJCF FK) and episode quality
    try:
        rows = []
        for e in eps:
            md, st, obj = e["md"], e["st"], e["obj"]
            init = np.array(md["initial_object_xyz"][:2], float)
            fx, fy = md["final_object_xyz"][0], md["final_object_xyz"][1]
            dfin = np.hypot(obj[:, 0] - fx, obj[:, 1] - fy)
            mov = np.hypot(obj[:, 0] - init[0], obj[:, 1] - init[1])
            idx = np.where((dfin < 0.015) & (mov > 0.05))[0]
            if not len(idx):
                continue
            ta = int(idx[0]); g = st[:, 5]
            if len(g) < ta + 7:
                continue
            r = g[ta + 5:] - g[ta:-5]
            thr = max(0.05, 0.1 * float(g.max() - g.min()))
            k = int(np.argmax(r))
            if r[k] < thr:
                continue
            trel = ta + k
            q_rel = st[trel, :5] if rad else st[trel, :5] / K
            tcp = fk_sites(base_el, dict(zip(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"], q_rel)))["gripperframe"][:3, 3]
            pan_n = float((st[trel, 0] * K[0]) if rad else st[trel, 0])
            rows.append({"plate": md["destination"], "pan_n": pan_n, "tcp": tcp, "ok_side": (pan_n > 0) == (md["destination"] == "blue"),
                         "wrist": float(st[trel, 3])})
        inv["release_detected"] = len(rows)
        if rows:
            inv["release_plate_match"] = round(float(np.mean([r["ok_side"] for r in rows])), 4)
        Tinv = np.linalg.inv(Tw) if Tw is not None else None
        geo = {}
        for p in ["blue", "green"]:
            R_ = [r for r in rows if r["plate"] == p]
            if not R_ or Tinv is None or p not in bowl:
                continue
            pc = (Tinv @ np.array([bowl[p][0], bowl[p][1], 0.004, 1.0]))[:3]
            d_sim = [math.hypot(r["tcp"][0] - pc[0], r["tcp"][1] - pc[1]) * 100 for r in R_]
            geo.setdefault("sim_tcp_to_plate_cm_median", {})[p] = round(float(np.median(d_sim)), 2)
            geo.setdefault("plate_centre_base_m", {})[p] = pc.round(4).tolist()
        # real plates in the same base frame (FK of the real release poses, zero at range midpoint)
        if Tinv is not None:
            for p in REAL_SETS:
                P = []
                for qn in real_rel[p]:
                    q = dict(zip(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"], qn / K))
                    P.append(fk_sites(base_el, q)["gripperframe"][:3, 3])
                P = np.array(P)
                mean = P.mean(axis=0)
                pc = np.array(geo["plate_centre_base_m"][p]) if "plate_centre_base_m" in geo and p in geo["plate_centre_base_m"] else None
                geo.setdefault("real_tcp_mean_base_m", {})[p] = mean.round(4).tolist()
                geo.setdefault("real_tcp_sd_m", {})[p] = P.std(axis=0, ddof=1).round(4).tolist() if len(P) > 1 else None
                if pc is not None:
                    geo.setdefault("real_to_sim_plate_cm", {})[p] = round(float(math.hypot(mean[0] - pc[0], mean[1] - pc[1]) * 100), 2)
        res["checks"]["geometry"] = geo
        res["checks"]["release"] = {"pan_residual_deg": {}, "real_pan_sd_units": {k: round(v, 2) for k, v in real_pan_sd.items()}}
        if n5 is not None:
            for p in REAL_SETS:
                P = [r["pan_n"] for r in rows if r["plate"] == p]
                if P:
                    res["checks"]["release"]["pan_residual_deg"][p] = round(float((np.median(P) - real_pan[p]) / K[0] * 180 / math.pi), 2)
                    res["checks"]["release"].setdefault("sim_pan_norm_median", {})[p] = round(float(np.median(P)), 2)
    except Exception as ex:
        res["errors"].append("release/geometry: %s" % ex)

    # cameras: brightness and green-plate position (approximate), sim vs real
    try:
        cams = {}
        for label, fname in [("sim_overhead", "overhead.mp4"), ("sim_close", "wrist.mp4")]:
            bufs = []
            for e in eps[:5]:
                a = frames_rgb(e["dir"] + "/" + fname)
                if a is not None:
                    bufs.append(a)
            if bufs:
                a = np.concatenate(bufs)
                cams[label] = {**luma_stats(a), "green": green_centroid(a[:30])}
        for label, cam in [("real_top", "243622061421"), ("real_close", "243722064316")]:
            path = os.path.join(args.real, "videos", "observation.images.realsense " + cam, "chunk-000", "file-000.mp4")
            a = frames_rgb(path, dur=30)
            if a is not None:
                cams[label] = {**luma_stats(a), "green": green_centroid(a)}
        res["checks"]["cameras"] = cams
    except Exception as ex:
        res["errors"].append("cameras: %s" % ex)

    # ---------- decisions
    c = res["checks"]
    def ok_units():
        u = c.get("units", {}); r = c.get("release", {}).get("pan_residual_deg", {})
        if len(r) < 2:
            return False, "pan residual not computable"
        pan_ok = all(abs(v) <= TOL["pan_deg"] for v in r.values())
        return pan_ok, "pan residual deg %s (tol %.1f)" % (r, TOL["pan_deg"])
    results = {}
    results["provenance_ok"] = bool(res["provenance"].get("verified")) and not res["source"]["changed_vs_2026-11-14"]
    results["units_ok"], results["units_note"] = ok_units()
    gr = c.get("gripper", {})
    results["gripper_ok"] = bool(gr) and abs(gr.get("span_ratio_sim_over_real", 9) - 1) <= TOL["gripper_ratio"] and abs(gr.get("closed_offset_units", 9)) <= TOL["gripper_offset_units"]
    un = c.get("units", {})
    results["rest_ok"] = un.get("near_real_rest_frac_sim", 0) >= TOL["rest_frac_min"]
    results["wrist_ok"] = un.get("wrist_pinned_frac_sim", 1) <= TOL["wrist_pinned_max"]
    cam = c.get("cameras", {})
    def bright(lbl_sim, lbl_real):
        if lbl_sim in cam and lbl_real in cam:
            return abs(cam[lbl_sim]["mean_luma"] / cam[lbl_real]["mean_luma"] - 1) <= TOL["brightness_rel"]
        return False
    results["brightness_ok"] = bright("sim_overhead", "real_top") and bright("sim_close", "real_close")
    def cam_ok(sim_l, real_l):
        if sim_l in cam and real_l in cam and cam[sim_l].get("green") and cam[real_l].get("green"):
            dx = cam[sim_l]["green"]["x_px"] - cam[real_l]["green"]["x_px"]
            dy = cam[sim_l]["green"]["y_px"] - cam[real_l]["green"]["y_px"]
            return math.hypot(dx, dy) <= TOL["cam_px"], round(math.hypot(dx, dy), 1)
        return False, None
    results["top_camera_ok"], res["checks"]["top_green_offset_px"] = cam_ok("sim_overhead", "real_top")
    results["close_camera_ok"], res["checks"]["close_green_offset_px"] = cam_ok("sim_close", "real_close")
    inv = res["inventory"]
    results["release_ok"] = inv.get("release_plate_match", 0) >= TOL["release_min"] and (inv.get("final_xy_err_m_max") or 9) <= 0.015
    results["episodes_ok"] = inv.get("episodes", 0) > 0 and inv.get("grasp_verified", 0) / max(inv.get("episodes", 1), 1) >= TOL["episodes_min"] and inv.get("success", 0) / max(inv.get("episodes", 1), 1) >= TOL["episodes_min"]
    gm = c.get("geometry", {}).get("real_to_sim_plate_cm", {})
    results["geometry_plates_ok"] = len(gm) == 2 and all(v <= TOL["geom_cm"] for v in gm.values())
    fk = c.get("geometry", {}).get("sim_tcp_to_plate_cm_median", {})
    results["fk_cross_check_ok"] = len(fk) == 2 and all(v <= TOL["fk_cm"] for v in fk.values())
    results["od10_clean"] = inv.get("od10_contradictions", 1) == 0
    results["tape_class_in_sim"] = bool(res["source"]["tape_or_defect_word_in_source"])
    uses = {
        "a_smolvla_cotraining": ["provenance_ok", "units_ok", "gripper_ok", "rest_ok", "wrist_ok", "brightness_ok", "top_camera_ok", "close_camera_ok", "release_ok", "episodes_ok", "geometry_plates_ok"],
        "b_act_pretrain_augment": ["provenance_ok", "units_ok", "gripper_ok", "rest_ok", "wrist_ok", "brightness_ok", "top_camera_ok", "close_camera_ok", "episodes_ok", "geometry_plates_ok"],
        "c_geometry_fk_crosscheck": ["provenance_ok", "fk_cross_check_ok"],
        "d_detector_synthetic": ["provenance_ok", "brightness_ok", "top_camera_ok", "close_camera_ok", "tape_class_in_sim"],
    }
    decisions = {}
    for u, keys in uses.items():
        fails = [k for k in keys if (k == "tape_class_in_sim" and not results[k]) or (k != "tape_class_in_sim" and not results.get(k, False))]
        if u == "d_detector_synthetic":
            fails = [k for k in keys if k != "tape_class_in_sim" and not results.get(k, False)] + ([] if results["tape_class_in_sim"] else ["no tape/defect class in sim"])
        decisions[u] = {"decision": "GO" if not fails else "NO-GO", "failed": fails}
    decisions["e_none"] = {"decision": "GO", "failed": [], "note": "default: keep sim out of training"}
    results = {k: (v if isinstance(v, str) else bool(v)) for k, v in results.items()}
    res["results"] = results
    res["decisions"] = decisions

    # optional converter (only for a GO training use; videos are not transcoded here)
    res["converter"] = {"ran": False}
    train_go = decisions["a_smolvla_cotraining"]["decision"] == "GO" or decisions["b_act_pretrain_augment"]["decision"] == "GO"
    if args.convert and train_go and n5 is not None:
        out = os.path.join(args.out_dir, version + "_lerobot")
        os.makedirs(out + "/data/chunk-000", exist_ok=True); os.makedirs(out + "/meta", exist_ok=True)
        rows_t = []
        for ei, e in enumerate(eps):
            qn = e["st"][:, :5] * K if rad else e["st"][:, :5]
            gn = e["st"][:, 5] * (1.0 if norm_ok else K_grip_enc) + (0.0 if norm_ok else 0.0)
            state = np.concatenate([qn, gn[:, None]], axis=1).astype(np.float32)
            for fi in range(len(state)):
                rows_t.append({"observation.state": state[fi].tolist(), "action": state[fi].tolist(), "timestamp": float(fi / 30.0),
                               "frame_index": fi, "episode_index": ei, "index": len(rows_t), "task_index": 0})
        pq.write_table(pa.Table.from_pylist(rows_t), out + "/data/chunk-000/file-000.parquet")
        pq.write_table(pa.Table.from_pylist([{"task_index": 0, "task": "move the brick to the inspection spot"}]), out + "/meta/tasks.parquet")
        res["converter"] = {"ran": True, "out": out, "note": "parquet only: state/action in normalized units (gripper unvalidated), task text placeholder, no video transcode"}

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, version + ".json")
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    line = "%s version=%s episodes=%d prov=%s units=%s gripper=%s rest=%s wrist=%s bright=%s topcam=%s closecam=%s release=%s episodes=%s geomplates=%s fkcheck=%s | a_smolvla=%s b_act=%s c_geom=%s d_detector=%s e_none=GO | json=%s errors=%d" % (
        res["utc"], version, inv.get("episodes", 0), int(results["provenance_ok"]), int(results["units_ok"]), int(results["gripper_ok"]),
        int(results["rest_ok"]), int(results["wrist_ok"]), int(results["brightness_ok"]), int(results["top_camera_ok"]), int(results["close_camera_ok"]),
        int(results["release_ok"]), int(results["episodes_ok"]), int(results["geometry_plates_ok"]), int(results["fk_cross_check_ok"]),
        decisions["a_smolvla_cotraining"]["decision"], decisions["b_act_pretrain_augment"]["decision"], decisions["c_geometry_fk_crosscheck"]["decision"],
        decisions["d_detector_synthetic"]["decision"], out_json, len(res["errors"]))
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "a") as fh:
        fh.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
