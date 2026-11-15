"""Paint a fixed cross on the cube each episode moved, in the top camera video of a copied LeRobot v3 dataset.

Run with envs/lerobot-env (pyav with dav1d and libsvtav1):
  python -I app/data/paint_cross.py SRC_DATASET_DIR OUT_DIR --analyze      first and last frames, chosen cube, sheet
  python -I app/data/paint_cross.py SRC_DATASET_DIR OUT_DIR --paint        copy SRC to OUT and repaint the top video
  --overrides FILE   JSON {"episode": candidate number from review_*.jpg | [u, v] | "skip"} for unresolved episodes

The moved cube is the cube found by topdet in the episode's first frames whose spot is bare mat in its last frames.
Episodes with no moved cube or more than one are reported and left unpainted; --paint refuses to run while any
episode is unresolved unless --allow-unresolved is given. The cross is drawn with marker.Marker.paint's geometry
from app/config/marker_top.json, so training frames match the live marker. Only the top video files and the top
camera's entry in meta/stats.json change; every other file is a byte copy.
"""
import argparse
import glob
import json
import os
import shutil
import sys
from fractions import Fraction

import av
import cv2
import numpy as np
import pandas as pd

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP, "tools"))
import topdet  # noqa: E402

TOP = "observation.images.realsense 243622061421"
EDGE_FRAMES = 3
SPOT_R = 9
MAT_V = 90


def load_cfg(path):
    with open(path) as f:
        return json.load(f)


def episodes(root):
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index").reset_index(drop=True)


def video_files(root, info, key, eps):
    out = {}
    for (c, f), g in eps.groupby([f"videos/{key}/chunk_index", f"videos/{key}/file_index"]):
        path = os.path.join(root, info["video_path"].format(video_key=key, chunk_index=int(c), file_index=int(f)))
        spans = [(int(r.episode_index), int(round(r[f"videos/{key}/from_timestamp"] * info["fps"])),
                  int(round(r[f"videos/{key}/to_timestamp"] * info["fps"]))) for _, r in g.iterrows()]
        out[path] = spans
    return out


def frames(path, fps):
    with av.open(path) as c:
        for fr in c.decode(video=0):
            yield int(round(fr.time * fps)), fr.to_ndarray(format="bgr24")


def ep_of(spans, idx):
    for e, a, b in spans:
        if a <= idx < b:
            return e
    return None


def paint(img, u, v, c):
    out = img.copy()
    u, v, h = int(round(u)), int(round(v)), int(c["half"])
    segs = ((u - h, v - h, u + h, v + h), (u - h, v + h, u + h, v - h))
    for x0, y0, x1, y1 in segs:
        cv2.line(out, (x0, y0), (x1, y1), (0, 0, 0), c["width"] + 2 * c["outline"], cv2.LINE_AA)
    for x0, y0, x1, y1 in segs:
        cv2.line(out, (x0, y0), (x1, y1), tuple(int(x) for x in c["bgr"]), c["width"], cv2.LINE_AA)
    return out


def bare(img, u, v):
    V = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
    m = np.zeros(V.shape, np.uint8)
    cv2.circle(m, (int(round(u)), int(round(v))), SPOT_R, 1, -1)
    return float((V[m.astype(bool)] < MAT_V).mean())


def plate_hull(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = topdet.PLATE(hsv[..., 0], hsv[..., 1], hsv[..., 2]).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    hull = np.zeros(m.shape, np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 6000:
            pts = cv2.findNonZero((lab == i).astype(np.uint8))
            cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    return cv2.erode(hull, np.ones((5, 5), np.uint8)).astype(bool)


def plate_blobs(img, hull, area=(300, 3600)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out = []
    for name, fn in topdet.CLASSES.items():
        fg = (fn(H, S, V) & hull).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        k, lab, st, cen = cv2.connectedComponentsWithStats(fg)
        for i in range(1, k):
            if area[0] <= st[i, cv2.CC_STAT_AREA] <= area[1]:
                out.append((name, float(cen[i][0]), float(cen[i][1])))
    return out


def new_on_plate(first, last):
    hull = plate_hull(first)
    a, b = plate_blobs(first, hull), plate_blobs(last, hull)
    return [x for x in b if not any(y[0] == x[0] and np.hypot(y[1] - x[1], y[2] - x[2]) < 15 for y in a)]


def analyze(root, info, cfg):
    eps = episodes(root)
    fps = info["fps"]
    edges = {}
    for path, spans in video_files(root, info, TOP, eps).items():
        bounds = {e: (a, b) for e, a, b in spans}
        for idx, img in frames(path, fps):
            e = ep_of(spans, idx)
            if e is None:
                continue
            a, b = bounds[e]
            d = edges.setdefault(e, {"first": [], "last": []})
            if idx < a + EDGE_FRAMES:
                d["first"].append(img)
            if idx >= b - EDGE_FRAMES:
                d["last"].append(img)
    res = {}
    for e in sorted(edges):
        first = np.median(np.stack(edges[e]["first"]), axis=0).astype(np.uint8)
        last = np.median(np.stack(edges[e]["last"]), axis=0).astype(np.uint8)
        cubes = [d for d in topdet.detect(first, cfg["workspace"], cfg["keepout"]) if d["ok"]]
        for d in cubes:
            d["bare_first"] = round(bare(first, d["u"], d["v"]), 2)
            d["bare_last"] = round(bare(last, d["u"], d["v"]), 2)
        moved = [d for d in cubes if d["bare_last"] >= 0.6 and d["bare_first"] <= 0.2]
        placed = sorted({x[0] for x in new_on_plate(first, last)})
        match = [d for d in moved if len(placed) == 1 and d["colour"] == placed[0]]
        if len(moved) == 1:
            status, target = "ok", moved[0]
        elif len(match) == 1:
            status, target = "ok_by_plate", match[0]
        else:
            status, target = ("none_moved" if not moved else "several_moved"), None
        res[e] = {"status": status, "target": target, "n_cubes": len(cubes), "placed": placed,
                  "moved": [(d["colour"], d["u"], d["v"]) for d in moved], "cubes": cubes, "first": first,
                  "last": last}
    return res


def apply_overrides(res, path):
    with open(path) as f:
        ov = json.load(f)
    for k, v in ov.items():
        r = res[int(k)]
        if v == "skip":
            r["status"], r["target"] = "skip", None
        elif isinstance(v, int):
            r["status"], r["target"] = "override", dict(r["cubes"][v])
        else:
            r["status"], r["target"] = "override", {"u": float(v[0]), "v": float(v[1]), "colour": None}


def review(res, rep_dir, per_image=5):
    bad = [e for e, r in sorted(res.items()) if r["status"] not in ("ok", "ok_by_plate", "override", "skip")]
    paths = []
    for n in range(0, len(bad), per_image):
        rows = []
        for e in bad[n:n + per_image]:
            r = res[e]
            a = r["first"].copy()
            for i, d in enumerate(r["cubes"]):
                col = (0, 0, 255) if (d["colour"], d["u"], d["v"]) in [tuple(m) for m in r["moved"]] else (0, 255, 0)
                cv2.circle(a, (int(d["u"]), int(d["v"])), 14, col, 2)
                cv2.putText(a, str(i), (int(d["u"]) - 5, int(d["v"]) - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(a, f"ep {e} first ({r['status']}, plate +{'/'.join(r['placed']) or '?'})", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            b = r["last"].copy()
            cv2.putText(b, f"ep {e} last", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            rows.append(cv2.resize(np.hstack([a, b]), (960, 360), interpolation=cv2.INTER_AREA))
        p = os.path.join(rep_dir, f"review_{n // per_image}.jpg")
        cv2.imwrite(p, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(p)
    return paths


def sheet(res, cfg, path, cols=8, w=240):
    h = w * 3 // 4
    tiles = []
    for e, r in sorted(res.items()):
        img = r["first"].copy()
        if r["target"]:
            img = paint(img, r["target"]["u"], r["target"]["v"], cfg["cross"])
        t = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        last = cv2.resize(r["last"], (w // 3, h // 3), interpolation=cv2.INTER_AREA)
        t[h - h // 3:, w - w // 3:] = last
        col = (0, 200, 0) if r["status"] in ("ok", "ok_by_plate", "override") else (0, 0, 255)
        cv2.rectangle(t, (0, 0), (w - 1, 18), (0, 0, 0), -1)
        cv2.putText(t, f"ep {e} {r['status']}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros((h, w, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def repaint(src_root, out_root, info, res, cfg):
    eps = episodes(src_root)
    fps = info["fps"]
    vinfo = info["features"][TOP]["info"]
    acc = {"n": 0, "sum": np.zeros(3), "sq": np.zeros(3), "min": np.full(3, np.inf), "max": np.full(3, -np.inf),
           "samples": []}
    checks = []
    for src, spans in video_files(src_root, info, TOP, eps).items():
        dst = os.path.join(out_root, os.path.relpath(src, src_root))
        tmp = dst + ".painting.mp4"
        n_in = 0
        with av.open(tmp, "w") as oc:
            st = oc.add_stream("libsvtav1", rate=fps)
            st.width, st.height, st.pix_fmt = vinfo["video.width"], vinfo["video.height"], vinfo["video.pix_fmt"]
            st.time_base = Fraction(1, fps)
            st.options = {"g": str(vinfo["video.g"]), "crf": str(vinfo["video.crf"]), "preset": str(vinfo["video.preset"])}
            for idx, img in frames(src, fps):
                n_in += 1
                e = ep_of(spans, idx)
                t = res.get(e, {}).get("target") if e is not None else None
                if t:
                    img = paint(img, t["u"], t["v"], cfg["cross"])
                if idx % 10 == 0:
                    rgb = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)[..., ::-1].reshape(-1, 3) / 255.0
                    acc["n"] += len(rgb)
                    acc["sum"] += rgb.sum(0)
                    acc["sq"] += (rgb ** 2).sum(0)
                    acc["min"] = np.minimum(acc["min"], rgb.min(0))
                    acc["max"] = np.maximum(acc["max"], rgb.max(0))
                    acc["samples"].append(rgb[::37])
                fr = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format=vinfo["video.pix_fmt"])
                fr.pts = idx
                fr.time_base = Fraction(1, fps)
                for p in st.encode(fr):
                    oc.mux(p)
            for p in st.encode():
                oc.mux(p)
        idxs = [i for i, _ in frames(tmp, fps)]
        ok = len(idxs) == n_in and idxs == sorted(idxs) and idxs[0] == spans[0][1]
        checks.append({"file": os.path.relpath(src, src_root), "frames_in": n_in, "frames_out": len(idxs),
                       "first_idx": idxs[0], "last_idx": idxs[-1], "ok": ok})
        if not ok:
            raise SystemExit(f"re-encode check failed for {src}: {checks[-1]}")
        os.replace(tmp, dst)
    mean = acc["sum"] / acc["n"]
    std = np.sqrt(np.maximum(acc["sq"] / acc["n"] - mean ** 2, 0))
    samples = np.concatenate(acc["samples"])
    return {"mean": mean, "std": std, "min": acc["min"], "max": acc["max"], "samples": samples}, checks


def update_stats(out_root, new):
    p = os.path.join(out_root, "meta", "stats.json")
    with open(p) as f:
        stats = json.load(f)
    old = stats[TOP]
    shape = np.asarray(old["mean"]).shape
    changed = []
    for k in list(old):
        if k in ("mean", "std", "min", "max"):
            old[k] = new[k].reshape(shape).tolist()
        elif k.startswith("q") and k[1:].isdigit():
            old[k] = np.quantile(new["samples"], int(k[1:]) / 100.0, axis=0).reshape(shape).tolist()
        else:
            continue
        changed.append(k)
    with open(p, "w") as f:
        json.dump(stats, f, indent=4)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--paint", action="store_true")
    ap.add_argument("--allow-unresolved", action="store_true")
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--config", default=os.path.join(APP, "config", "marker_top.json"))
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    with open(os.path.join(a.src, "meta", "info.json")) as f:
        info = json.load(f)
    if TOP not in info["features"]:
        raise SystemExit(f"{TOP} not in {a.src}")
    os.makedirs(a.out + "_report", exist_ok=True)
    rep_dir = a.out + "_report"
    res = analyze(a.src, info, cfg)
    if a.overrides:
        apply_overrides(res, a.overrides)
    sheet(res, cfg, os.path.join(rep_dir, "sheet.jpg"))
    reviews = review(res, rep_dir)
    summary = {str(e): {k: v for k, v in r.items() if k not in ("first", "last")} for e, r in res.items()}
    with open(os.path.join(rep_dir, "targets.json"), "w") as f:
        json.dump({"src": a.src, "camera": TOP, "cross": cfg["cross"], "episodes": summary}, f, indent=1)
    bad = [e for e, r in res.items() if r["status"] not in ("ok", "ok_by_plate", "override", "skip")]
    print(json.dumps({"episodes": len(res), "ok": sum(r["status"] == "ok" for r in res.values()),
                      "ok_by_plate": [e for e, r in res.items() if r["status"] == "ok_by_plate"],
                      "override": [e for e, r in res.items() if r["status"] == "override"],
                      "skip": [e for e, r in res.items() if r["status"] == "skip"],
                      "unresolved": {str(e): res[e]["status"] for e in bad}, "review": reviews, "report": rep_dir}))
    if not a.paint:
        return
    if bad and not a.allow_unresolved:
        raise SystemExit(f"{len(bad)} unresolved episodes; review {rep_dir}/sheet.jpg or pass --allow-unresolved")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    shutil.copytree(a.src, a.out)
    new, checks = repaint(a.src, a.out, info, res, cfg)
    changed = update_stats(a.out, new)
    with open(os.path.join(rep_dir, "paint.json"), "w") as f:
        json.dump({"checks": checks, "stats_changed": changed}, f, indent=1)
    print(json.dumps({"painted": a.out, "checks": checks, "stats_changed": changed}))


if __name__ == "__main__":
    main()
