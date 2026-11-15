import glob
import json
import os
import subprocess

import cv2
import numpy as np
import pyarrow.parquet as pq

R = "/home/intel/intel-junction-2026"
D = R + "/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
REP = R + "/data/studio-datasets-report/V2"
FR = R + "/work/P2-D-DATA/V2/frames"
OUT = R + "/work/P2-D-DATA/V2/out"
os.makedirs(FR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
FPS = 30.0
CAM = {"TOP": "realsense 243622061421", "CLOSE": "realsense 243722064316"}
meta = {}
for f in sorted(glob.glob(D + "/meta/episodes/chunk-*/file-*.parquet")):
    for r in pq.read_table(f).to_pylist():
        meta[int(r["episode_index"])] = r
S_l, E_l, F_l = [], [], []
for f in sorted(glob.glob(D + "/data/chunk-*/file-*.parquet")):
    t = pq.read_table(f, columns=["observation.state", "frame_index", "episode_index"]).to_pydict()
    S_l.append(np.array(t["observation.state"], dtype=float))
    E_l.append(np.array(t["episode_index"]))
    F_l.append(np.array(t["frame_index"]))
S = np.vstack(S_l)
E = np.concatenate(E_l)
F = np.concatenate(F_l)
order = np.lexsort((F, E))
S, E, F = S[order], E[order], F[order]


def vinfo(e, tag):
    m = meta[e]
    pre = "videos/observation.images." + CAM[tag] + "/"
    p = D + "/videos/observation.images." + CAM[tag] + "/chunk-%03d/file-%03d.mp4" % (
        m[pre + "chunk_index"], m[pre + "file_index"])
    return p, float(m[pre + "from_timestamp"]), float(m[pre + "to_timestamp"])


def grab(e, local, tag, name):
    p, ft, _ = vinfo(e, tag)
    out = FR + "/" + name + ".png"
    rc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "%.4f" % (ft + local / FPS),
                         "-i", p, "-frames:v", "1", out], capture_output=True).returncode
    return out if rc == 0 and os.path.exists(out) else None


def runs(mask, minlen):
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= minlen:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


res = {}
# 1 plate centroids on the arm-home reference frame (ep0 f0, top view)
ref = cv2.imread(FR + "/ref_ep0_f0_top.png")
hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)


def biggest(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    return [round(M["m10"] / M["m00"], 1), round(M["m01"] / M["m00"], 1), int(cv2.contourArea(c))]


blue = biggest(cv2.inRange(hsv, (85, 90, 90), (110, 255, 255)))
green = biggest(cv2.inRange(hsv, (45, 60, 50), (85, 255, 255)))
res["plate_centroids_ref_top_640x480"] = {"blue_xy_area": blue, "green_xy_area": green,
                                           "w2_claim_blue": [242, 150], "w2_claim_green": [578, 182]}

# 2 marker check of W3 placed-brick points on the put-down frames (eps 3, 5, 6)
pts = [(348, 123, "P1 348,123"), (340, 72, "P2 340,72"), (391, 216, "P3 391,216")]
tiles = []
for name in ["ep3_spot_top", "ep5_spot_top", "ep6_spot_top"]:
    im = cv2.imread(FR + "/" + name + ".png")
    for (x, y, lab) in pts:
        cv2.circle(im, (x, y), 14, (0, 0, 255), 2)
        cv2.putText(im, lab, (x + 16, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.drawMarker(im, (int(blue[0]), int(blue[1])), (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.drawMarker(im, (int(green[0]), int(green[1])), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(im, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    tiles.append(cv2.resize(im, (480, 360)))
cv2.imwrite(REP + "/marker_check_w3_spot_points.png", np.hstack(tiles))

# 3 start-of-episode and hold checks for eps 8-10 (brick in jaws?)
cells = [("ep8 f5 TOP", grab(8, 5, "TOP", "ep8_f5_top")), ("ep8 f5 CLOSE", grab(8, 5, "CLOSE", "ep8_f5_close")),
         ("ep9 f5 TOP", grab(9, 5, "TOP", "ep9_f5_top")), ("ep9 f5 CLOSE", grab(9, 5, "CLOSE", "ep9_f5_close")),
         ("ep10 f5 TOP", grab(10, 5, "TOP", "ep10_f5_top")), ("ep10 f5 CLOSE", grab(10, 5, "CLOSE", "ep10_f5_close")),
         ("ep9 f150 CLOSE", grab(9, 150, "CLOSE", "ep9_f150_close")), ("ep9 f187 CLOSE", grab(9, 187, "CLOSE", "ep9_f187_close"))]
tiles = []
for lab, p in cells:
    im = cv2.imread(p) if p else np.full((480, 640, 3), (0, 0, 80), np.uint8)
    cv2.rectangle(im, (0, 0), (330, 34), (0, 0, 0), -1)
    cv2.putText(im, lab, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    tiles.append(cv2.resize(im, (320, 240)))
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 8, 4)]
cv2.imwrite(REP + "/start_and_hold_check_eps8_10.png", np.vstack(rows))
res["start_check_grabs"] = [c[0] + ":" + str(c[1] is not None) for c in cells]

# 4 decodability of first and last frame per episode and camera, plus container durations
dec = {}
for e in sorted(meta):
    L = int(meta[e]["length"])
    for tag in CAM:
        p, ft, to = vinfo(e, tag)
        r1 = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "%.4f" % ft, "-i", p,
                             "-frames:v", "1", "-f", "null", "-"], capture_output=True).returncode
        r2 = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "%.4f" % (ft + (L - 1) / FPS),
                             "-i", p, "-frames:v", "1", "-f", "null", "-"], capture_output=True).returncode
        dec["%d-%s" % (e, tag)] = {"file": os.path.basename(p), "from": round(ft, 3), "to": round(to, 3),
                                   "len": L, "first_rc": r1, "last_rc": r2}
res["decode"] = dec
dur = {}
for p in sorted(glob.glob(D + "/videos/*/chunk-*/file-*.mp4")):
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                        "default=nw=1:nk=1", p], capture_output=True, text=True).stdout.strip()
    dur[p.split("/videos/")[1]] = o
res["container_durations_s"] = dur

# 5 gripper closure onset/plateau validation and idle-speed floor
val = []
idle = {}
for e in sorted(meta):
    idx = np.where(E == e)[0]
    s = S[idx]
    g = s[:, 5]
    N = len(idx)
    spd = np.zeros(N)
    spd[1:] = np.max(np.abs(np.diff(s[:, :5], axis=0)), axis=1)
    idle[e] = {"first30_p50": round(float(np.median(spd[1:31])), 3), "first30_p95": round(float(np.percentile(spd[1:31], 95)), 3),
               "last30_p50": round(float(np.median(spd[-30:])), 3), "last30_p95": round(float(np.percentile(spd[-30:], 95)), 3)}
    for (c0, c1) in runs(g < 4.0, 10):
        if c0 == 0:
            continue
        k = c0
        while k > 0 and g[k - 1] > g[k] + 0.3:
            k -= 1
        cm = float(np.median(g[c0 + 5:c1 - 5] if c1 - c0 > 12 else g[c0:c1]))
        pl = next((t for t in range(k, c1) if g[t] <= cm + 0.3), None)
        drop10 = (g[k] - g[min(k + 10, N - 1)]) / 10.0
        pre10 = float(np.max(np.abs(np.diff(g[max(0, k - 10):k + 1])))) if k > 0 else None
        val.append({"ep": e, "onset": int(k), "plateau": int(pl) if pl is not None else None,
                    "close_frames": int(pl - k) if pl is not None else None,
                    "drop_per_frame_10f": round(float(drop10), 3), "pre_onset_max_step": round(pre10, 3) if pre10 is not None else None,
                    "closed_med": round(cm, 3), "open_peak_2s": round(float(g[max(0, k - 60):k + 1].max()), 2),
                    "grasp_pose": s[pl, :5].round(2).tolist() if pl is not None else None})
    for (c0, c1) in runs(g < 4.0, 10):
        if c1 < N and c1 + 10 < N:
            pass
res["gripper_closure_validation"] = val
res["idle_speed_per_episode"] = idle
rel = []
for e in sorted(meta):
    idx = np.where(E == e)[0]
    g = S[idx, 5]
    N = len(idx)
    for (c0, c1) in runs(g < 4.0, 10):
        if c1 < N and c1 + 10 < N:
            rel.append({"ep": e, "release": int(c1), "rise_per_frame_10f": round(float((g[c1 + 10] - g[c1]) / 10.0), 3),
                        "g_at_release": round(float(g[c1]), 3), "g_peak_1s": round(float(g[c1:c1 + 30].max()), 2)})
res["release_validation"] = rel
with open(OUT + "/partC.json", "w") as fh:
    json.dump(res, fh, indent=1, default=float)
print("PLATES", blue, green)
print("GRABS", res["start_check_grabs"])
bad = [k for k, v in dec.items() if v["first_rc"] != 0 or v["last_rc"] != 0]
print("DECODE_BAD", bad, "of", len(dec))
for k in sorted(dur):
    print("DUR", k, dur[k])
for v in val:
    print("CLOSE", v)
for r in rel:
    print("REL", r)
for e in sorted(idle):
    print("IDLE", e, idle[e])
print("DONE_PARTC")
