import glob, json
import numpy as np
import pandas as pd
from PIL import Image

W = "/home/intel/intel-junction-2026/work/P2-D-DATA/W2/results3"
SN = "/home/intel/intel-junction-2026/work/P2-D-DATA/W2/snapshot3/6d6bdce3-b382-478d-9697-fb9e573b97af"
FR = f"{W}/frames/6d6bdce3-b382-478d-9697-fb9e573b97af"
TOP = "243622061421"
FPS = 30
H = int(round(0.7 * FPS))  # 21 frames = 0.7 s before the event
PLATE = {"blue": (242.0, 150.5), "green": (578.0, 182.0)}

segs = json.load(open(f"{W}/segments.json"))
calib = json.load(open(f"{W}/calib.json"))

parq = sorted(glob.glob(f"{SN}/data/chunk-000/*.parquet"))
df = pd.concat([pd.read_parquet(p) for p in parq])
states = {}
for ep, g in df.groupby("episode_index"):
    g = g.sort_values("frame_index")
    states[int(ep)] = np.stack(g["observation.state"].to_numpy())[:, :6]


def load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.int32)


def brick_px(a, b, thr=60):
    d = np.abs(a - b).sum(axis=2) > thr
    m = np.zeros_like(d); m[60:380, 120:520] = True
    ys, xs = np.nonzero(d & m)
    if xs.size < 80:
        return None
    return {"x": round(float(np.median(xs)), 1), "y": round(float(np.median(ys)), 1),
            "sx": round(float(xs.std()), 1), "sy": round(float(ys.std()), 1), "px": int(xs.size)}


events = []
for s in segs:
    ep = s["episode"]; sg = s["segments"]
    if not (ep in states):
        continue
    st = states[ep]
    g = sg.get("grasp_closed_frame")
    if g is not None and ep != 9:  # ep9 closed on nothing
        png_s = f"{FR}/ep{ep}_{TOP}_start.png"
        png_g = f"{FR}/ep{ep}_{TOP}_grasp.png"
        px = None
        try:
            px = brick_px(load(png_s), load(png_g))
        except Exception:
            pass
        events.append({"ep": ep, "kind": "grasp", "frame": g, "pose": st[g][:5].round(2).tolist(),
                       "hover": st[max(0, g - H)][:5].round(2).tolist(), "px": px})
    r = sg.get("release_frame")
    if r is not None:
        if ep in (3, 4, 5, 6):
            px = None  # put-down on mat, placed-brick centroid already in calib od9
            pl = calib["od9"]["inspection_spot"]["placed_brick_top_view_px"]["per_episode"][str(ep)]
            px = {"x": pl["cx"], "y": pl["cy"], "src": "placed-brick diff (mat put-down)"}
            plate = "mat (put-down)"
        else:
            plate = "blue" if ep in (7, 8) else "green"
            px = {"x": PLATE[plate][0], "y": PLATE[plate][1], "src": "plate centroid"}
        events.append({"ep": ep, "kind": "release", "frame": r, "plate": plate,
                       "pose": st[r][:5].round(2).tolist(), "hover": st[max(0, r - H)][:5].round(2).tolist(),
                       "px": px})

grasps = [e for e in events if e["kind"] == "grasp" and e["px"] is not None]
X = np.array([[e["px"]["x"], e["px"]["y"]] for e in grasps])
Yg = np.array([e["pose"] for e in grasps])
Yh = np.array([e["hover"] for e in grasps])


def idw(Xtr, Ytr, xq, p=2.0):
    d = np.linalg.norm(Xtr - xq, axis=1) + 1e-6
    w = 1 / d ** p
    return (w[:, None] * Ytr).sum(0) / w.sum()


def poly(Xtr, Ytr, xq):
    def feat(P):
        x, y = P[:, 0] / 640, P[:, 1] / 480
        return np.stack([np.ones_like(x), x, y], axis=1)
    A = feat(Xtr)
    coef, *_ = np.linalg.lstsq(A, Ytr, rcond=None)
    return (feat(np.atleast_2d(xq)) @ coef)[0]


def loo(fn, X, Y):
    errs = []
    for i in range(len(X)):
        m = np.arange(len(X)) != i
        errs.append(fn(X[m], Y[m], X[i]) - Y[i])
    E = np.abs(np.array(errs))
    return {"mae_per_joint_deg": E.mean(0).round(2).tolist(), "mae_mean_deg": round(float(E.mean()), 2)}


res = {}
for name, Y in (("grasp_pose", Yg), ("hover_pose", Yh)):
    res[name] = {
        "n_points": int(len(X)),
        "idw_p2_loo": loo(lambda a, b, q: idw(a, b, q), X, Y),
        "linear_loo": loo(lambda a, b, q: poly(a, b, q), X, Y),
    }
better = {k: ("idw_p2" if v["idw_p2_loo"]["mae_mean_deg"] <= v["linear_loo"]["mae_mean_deg"] else "linear")
          for k, v in res.items()}
pm = {
    "method": ("grasp events: pixel = median |start - grasp| > 60 in mat box (top view 243622061421); "
               "joints = observation.state[:5] at grasp frame; hover = state 0.7 s (21 frames) before event. "
               "Release events: pixel = plate centroid or placed-brick centroid (eps 3-6). ep9 excluded (closed on nothing); eps 8 and 10 have no grasp event."),
    "events": events,
    "fit_grasp_and_hover": res,
    "better_model": better,
    "caveat": "pixel from frame differencing includes arm motion; only 8 grasp points; LOO error is the honest validation figure.",
}
calib["pick_map"] = pm
json.dump(calib, open(f"{W}/calib.json", "w"), indent=1)
print(json.dumps({"n_events": len(events), "grasp_points": len(X), "fit": res, "better": better}, indent=1))
for e in events:
    print(e["ep"], e["kind"], e.get("plate", ""), e["frame"], e["px"], e["pose"])
