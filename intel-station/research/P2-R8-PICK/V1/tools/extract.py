# V1 verifier: grasp/release events from dataset states, TCP via URDF FK, top-view frames for labelling.
# Normalisation follows physicalai_so101_plugin/_conversion.py (body deg = (tick - mid) * 360/4095, mid = (min+max)/2)
# which reduces to rad = u * (range_max - range_min) / 200 * 2*pi/4095 for body joints u in [-100, 100].
import json, math, os, subprocess
import numpy as np
import pyarrow.parquet as pq
import xml.etree.ElementTree as ET

BASE = "/home/intel/intel-junction-2026"
OUT = BASE + "/work/P2-R8-PICK/V1"
URDF = BASE + "/repos/physicalai/packages/physicalai-so101-plugin/urdf/so101-follower/so101_follower.urdf"
CAL = json.load(open(BASE + "/app/motion/project_robots.json"))[0]["payload"]["calibration"]
DS = {"6d6": BASE + "/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af",
      "a4b": BASE + "/data/studio-datasets/a4ba6ede-092c-450f-8c76-559f66991dbd"}
TOP = "observation.images.realsense 243622061421"
BODY = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TCP = "gripper_frame_link"
FPS = 30.0
TWO_PI_TICK = 2 * math.pi / 4095.0


def load_urdf(path):
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        o, a = j.find("origin"), j.find("axis")
        xyz = o.get("xyz") if o is not None and o.get("xyz") else "0 0 0"
        rpy = o.get("rpy") if o is not None and o.get("rpy") else "0 0 0"
        ax = a.get("xyz") if a is not None and a.get("xyz") else "1 0 0"
        joints[j.get("name")] = {"type": j.get("type"), "parent": j.find("parent").get("link"),
                                 "child": j.find("child").get("link"),
                                 "xyz": [float(v) for v in xyz.split()],
                                 "rpy": [float(v) for v in rpy.split()],
                                 "axis": [float(v) for v in ax.split()]}
    return joints


def chain_to(joints, target):
    by_child = {v["child"]: k for k, v in joints.items()}
    names, link = [], target
    while link in by_child:
        names.append(by_child[link])
        link = joints[by_child[link]]["parent"]
    return list(reversed(names))


def rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_rot(axis, th):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def fk(joints, chain, q):
    T = np.eye(4)
    for n in chain:
        j = joints[n]
        M = np.eye(4)
        M[:3, :3] = rot_rpy(*j["rpy"])
        M[:3, 3] = j["xyz"]
        T = T @ M
        if j["type"] in ("revolute", "continuous") and n in q:
            R = np.eye(4)
            R[:3, :3] = axis_rot(j["axis"], q[n])
            T = T @ R
    return T


def q_of(state):
    out = {}
    for i, n in enumerate(BODY):
        span = CAL[n]["range_max"] - CAL[n]["range_min"]
        out[n] = float(state[i]) * span / 200.0 * TWO_PI_TICK
    return out


def tcp(joints, chain, state):
    return fk(joints, chain, q_of(state))[:3, 3]


def grasp_index(g):
    opened = False
    for t in range(len(g) - 30):
        if g[t] > 8:
            opened = True
        if opened and g[t] < 4 and np.all(g[t:t + 30] < 4):
            return t
    return None


def release_index(g, tc):
    for t in range(tc + 30, len(g)):
        if g[t] > 7:
            return t
    return None


def load(root):
    meta = []
    md = root + "/meta/episodes/chunk-000"
    for f in sorted(os.listdir(md)):
        meta += pq.read_table(md + "/" + f).to_pylist()
    st = {}
    dd = root + "/data/chunk-000"
    for f in sorted(os.listdir(dd)):
        t = pq.read_table(dd + "/" + f, columns=["observation.state", "episode_index", "frame_index"]).to_pydict()
        for s, e, fi in zip(t["observation.state"], t["episode_index"], t["frame_index"]):
            st.setdefault(e, []).append((fi, s))
    eps = {}
    for e, lst in st.items():
        lst.sort(key=lambda x: x[0])
        assert [fi for fi, _ in lst] == list(range(len(lst))), "frame index gap"
        eps[e] = np.array([s for _, s in lst], dtype=float)
    return meta, eps


KNOWN = {  # normalized pan, lift, elbow, wrist_flex, wrist_roll (+gripper) from FACTS-LIVE and gm_fast.yaml
    "home_facts_13:24": [-6.6, -98.1, 99.4, 54.3, -53.1, 25.7],
    "blue_ep7_release_facts": [32.1, 5.5, 25.6, 61.8, -36.8, 7.0],
    "green_ep8_release_facts": [-54.7, 24.5, 1.9, 54.2, -55.0, 7.0],
    "green_ep9_release_facts": [-57.3, 18.5, 7.8, 36.2, -52.7, 7.0],
    "green_ep10_release_facts": [-51.8, 47.7, -49.8, 80.1, -52.4, 7.0],
    "blue_wp0_gm_fast": [31.93, -19.05, 47.91, 49.3, -40.65, 25.0],
    "green_wp0_gm_fast": [-51.83, 53.44, -51.91, 84.82, -45.46, 25.0],
    "blue_wp11_gm_fast_home_end": [-5.21, -99.03, 96.37, 65.92, -44.99, 25.0],
}


def main():
    joints = load_urdf(URDF)
    chain = chain_to(joints, TCP)
    print("chain", chain)
    rows, allst = [], {}
    for tag, root in DS.items():
        meta, eps = load(root)
        allst[tag] = eps
        for m in sorted(meta, key=lambda r: r["episode_index"]):
            e = m["episode_index"]
            S = eps[e]
            g = S[:, 5]
            tc = grasp_index(g)
            rec = {"ds": tag, "ep": e, "len": int(m["length"]), "tasks": m["tasks"],
                   "video_file": int(m["videos/" + TOP + "/file_index"]),
                   "video_from_s": float(m["videos/" + TOP + "/from_timestamp"]),
                   "g0": round(float(g[0]), 2), "grasp_t": tc}
            if tc is not None:
                tr = release_index(g, tc)
                rec["closure_level"] = round(float(np.median(g[tc + 5:tc + 30])), 2)
                rec["release_t"] = tr
                rec["state_grasp"] = S[tc].tolist()
                rec["tcp_grasp_m"] = tcp(joints, chain, S[tc]).tolist()
                if tr is not None:
                    rec["state_release"] = S[tr].tolist()
                    rec["tcp_release_m"] = tcp(joints, chain, S[tr]).tolist()
            rec["tcp_start_m"] = tcp(joints, chain, S[0]).tolist()
            rec["state_start"] = S[0].tolist()
            rows.append(rec)
    # duplicate check: curated set a4b vs episodes of 6d6
    for r in rows:
        r["dup_of"] = None
        if r["ds"] == "a4b":
            a = allst["a4b"][r["ep"]]
            for e6, b in allst["6d6"].items():
                if b.shape == a.shape and np.allclose(a, b, atol=1e-4):
                    r["dup_of"] = "6d6:ep%d" % e6
    known = {}
    for k, u in KNOWN.items():
        known[k] = {"normalized": u, "tcp_m": tcp(joints, chain, u[:5]).tolist()}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"rows": rows, "known_poses": known, "chain": chain},
              open(OUT + "/episodes.json", "w"), indent=1)
    for r in rows:
        tg = r.get("tcp_grasp_m")
        print(r["ds"], "ep%02d" % r["ep"], "len", r["len"], "g0", r["g0"], "grasp_t", r["grasp_t"],
              "close", r.get("closure_level"), "rel_t", r.get("release_t"),
              "tcpG_mm", None if tg is None else [round(v * 1000, 1) for v in tg], "dup", r["dup_of"])
    for k, v in known.items():
        print("KNOWN", k, [round(x * 1000, 1) for x in v["tcp_m"]])
    return rows


def grab_frames(rows):
    os.makedirs(OUT + "/frames", exist_ok=True)
    for r in rows:
        if r.get("grasp_t") is None:
            continue
        vid = "%s/videos/%s/chunk-000/file-%03d.mp4" % (DS[r["ds"]], TOP, r["video_file"])
        for lab, t in (("start", 0), ("grasp", r["grasp_t"]), ("lift", r["grasp_t"] + 30)):
            ts = r["video_from_s"] + t / FPS
            out = "%s/frames/%s_ep%02d_%s.png" % (OUT, r["ds"], r["ep"], lab)
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "%.4f" % ts,
                            "-i", vid, "-frames:v", "1", out], check=True)
    print("frames written")


if __name__ == "__main__":
    rows = main()
    grab_frames(rows)
