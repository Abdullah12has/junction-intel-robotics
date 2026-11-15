#!/usr/bin/env python3
"""Self-check (R12): re-reads sources directly and compares with compare.json / metrics.json / gate JSON. Data only.
Usage: python3 -I selfcheck.py OUT"""
import ast, glob, hashlib, json, re, sys, time
import xml.etree.ElementTree as ET
import numpy as np
R = "/home/intel/intel-junction-2026"
OUT = sys.argv[1].rstrip("/")
SIM = R + "/data/sim/live/2026-11-14/dataset-150-raw"
res = []


def chk(name, ok, detail=""):
    res.append({"fact": name, "pass": bool(ok), "detail": str(detail)[:160]})


C = json.load(open(OUT + "/compare.json"))
G = json.load(open(OUT + "/gate/2026-11-14-95fa40dbf3.json"))
M = json.load(open(OUT + "/metrics.json"))["metrics"]
st = open(SIM + "/source/station_scene.py", encoding="utf-8").read()
bowl = ast.literal_eval(re.search(r"BOWL_CENTERS\s*=\s*(\{.*?\})", st, re.S).group(1))
chk("bowl centres in station_scene.py equal compare.json", all(abs(bowl[k][i] - C["bowl_centres_world_m"][k][i]) < 1e-9 for k in bowl for i in (0, 1)), bowl)
chk("station base pos/quat as recorded", "0.0388353 0.0388353 0" in st and "0.7071067812 0 0 -0.7071067812" in st)
chk("overhead look-at as recorded", "camera_look_at(overhead, [-0.025, -0.46, 0.61], [-0.025, -0.22, 0])" in st)
chk("overhead fovy 40", 'overhead.set("fovy", "40")' in st)
chk("wrist camera is world-fixed (appended to world)", "world.append(wrist)" in st and "camera_look_at(wrist, [0.025, -0.45, 0.22]" in st)
chk("friend's own comment: geometry visually fitted", "Geometry is visually fitted" in st)
md = [json.load(open(d + "/metadata.json")) for d in sorted(glob.glob(SIM + "/attempt-*"))]
chk("150 attempts with metadata", len(md) == 150, len(md))
chk("accepted, success, grasp verified for all", all(m["accepted"] and m["success"] and m["physical_grasp_verified"] for m in md))
chk("action_units = absolute_joint_radians for all", all(m["action_units"] == "absolute_joint_radians" for m in md))
od = sum((m["brick_type"] == "2x2" and m["destination"] == "green") or (m["brick_type"] != "2x2" and m["destination"] == "blue") for m in md)
chk("OD-10 contradictions recount = 73", od == 73, od)
chk("gate decisions as recorded (a NO-GO, c GO)", G["decisions"]["a_smolvla_cotraining"]["decision"] == "NO-GO" and G["decisions"]["c_geometry_fk_crosscheck"]["decision"] == "GO")
chk("manifest 762 entries, all verified", G["provenance"]["entries"] == 762 and G["provenance"]["verified"])
h = hashlib.sha256()
with open(SIM + "/../dataset-150-lerobot.tar.gz", "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
chk("tar sha256 = 36fbe792...93780b", h.hexdigest() == "36fbe79242afa86ed29247fd32327041780e0937e586b1c014962bc74b93780b", h.hexdigest()[:16])
rob = json.load(open(R + "/app/motion/project_robots.json"))
cal = [r for r in rob if r.get("type") == "SO101_Follower"][0]["payload"]["calibration"]
K0 = 200 * 4095 / (2 * np.pi * (cal["shoulder_pan"]["range_max"] - cal["shoulder_pan"]["range_min"]))
chk("K pan recomputed = compare K[0]", abs(K0 - C["joints"]["K_units_per_rad"][0]) < 1e-2, round(K0, 3))
mj = ET.parse(R + "/app/simgate/ref/so101.xml").getroot()
wl = [float(v) for v in [j for j in mj.iter("joint") if j.get("name") == "wrist_flex"][0].get("range").split()][1]
chk("MJCF wrist_flex limit 1.658 rad -> 91.62 units", abs(wl * C["joints"]["K_units_per_rad"][3] - 91.62) < 0.02, round(wl, 6))
info = json.load(open(R + "/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af/meta/info.json"))
chk("real Studio dataset: 16 episodes at 30 fps", info["total_episodes"] == 16 and info["fps"] == 30, (info["total_episodes"], info["fps"]))
used = [r for r in C["motion"]["real_episode_meta"] if r["used"]]
chk("real joints.csv: 15 used, 2 short files skipped", len(used) == 15 and len(C["motion"]["real_episode_meta"]) == 17, len(used))
rr = json.load(open(OUT + "/real_release_check.json"))["episodes"]
g_eps = [r for r in rr if r["episode"] in (8, 10, 14, 15)]
b_eps = [r for r in rr if r["episode"] in (7, 11, 12)]
chk("release pans: green eps 8,10,14,15 near -52..-55; blue 7,11,12 near +32..+34", all(r["pan_norm"] < -50 for r in g_eps) and all(30 < r["pan_norm"] < 35 for r in b_eps))
chk("metrics.json equals compare.json for top blue px offset", abs(M["top_plate_offset_px_real_minus_sim"]["blue"][0] - C["plates"]["top"]["real_minus_sim_px"]["blue"][0]) < 1e-9)
chk("image-based sim plate spacing within 3 mm of truth", abs(M["top_plate_spacing_mm"]["sim_image_via_model"] - M["top_plate_spacing_mm"]["sim_world_truth_from_bowl_centres"]) < 3.0, M["top_plate_spacing_mm"])
chk("live mirror identical to 2026-11-14 (compare.json)", C["version_identity"]["live_equals_2026_11_14"] is True)
chk("twin status recorded in compare.json", "twin" in C and "status" in C["twin"], C.get("twin", {}).get("status", ""))
n_ok = sum(r["pass"] for r in res)
out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "passed": n_ok, "total": len(res), "checks": res}
with open(OUT + "/selfcheck.json", "w") as fh:
    json.dump(out, fh, indent=1)
for r in res:
    print("PASS" if r["pass"] else "FAIL", "|", r["fact"], "|", r["detail"])
print("passed", n_ok, "of", len(res))
