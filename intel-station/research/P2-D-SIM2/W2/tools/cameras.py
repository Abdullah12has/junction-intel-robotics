"""Rough camera comparison: one release frame per camera (sim overhead/wrist vs real top/close).

Frames are extracted with ffmpeg (read-only inputs), analysed with PIL/numpy.
Plate pixels are found with simple RGB rules (not calibrated). Outputs: out/camera_metrics.json,
plots/camera_contact_sheet.png (under the laptop W2 folder).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

W = "/home/intel/intel-junction-2026/work/P2-D-SIM2/W2"
OUT = W + "/out"
FR = W + "/out/frames"
os.makedirs(FR, exist_ok=True)
os.makedirs(W + "/plots", exist_ok=True)
SIM = "/home/intel/intel-junction-2026/data/sim/2026-11-14/dataset-150-raw"
DS = "/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
TOP = "realsense 243622061421"
CLOSE = "realsense 243722064316"


def sim_release_idx(att):
    tr = json.load(open(f"{SIM}/{att}/trace.json"))
    return next(i for i, t in enumerate(tr) if t["phase"] == "release")


def ff_frame(src, idx, png):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", f"select=eq(n\\,{idx})", "-vsync", "0", "-frames:v", "1", png]
    subprocess.run(cmd, check=True)


def ff_time(src, ts, png):
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{ts:.4f}", "-i", src, "-frames:v", "1", png]
    subprocess.run(cmd, check=True)


def metrics(png):
    im = np.asarray(Image.open(png).convert("RGB")).astype(float)
    H, Wd, _ = im.shape
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    mx, mn = im.max(axis=2), im.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    blue = (B > 140) & (B > R + 60) & (G > 90) & (sat > 0.45)
    green = (G > 90) & (G > R + 40) & (G > B + 15) & (sat > 0.35)
    def cen(mask):
        if mask.sum() < 50:
            return None
        ys, xs = np.nonzero(mask)
        return [round(float(xs.mean() / Wd), 3), round(float(ys.mean() / H), 3)]
    return {"size": [Wd, H], "mean_rgb": [round(float(v), 1) for v in im.reshape(-1, 3).mean(axis=0)],
            "mean_sat": round(float(sat.mean()), 3), "mean_luma": round(float((0.299*R+0.587*G+0.114*B).mean()), 1),
            "blue_px_frac": round(float(blue.mean()), 4), "green_px_frac": round(float(green.mean()), 4),
            "blue_centroid_xy_norm": cen(blue), "green_centroid_xy_norm": cen(green)}


def real_ep_time(ep, key):
    # episode video timestamps from meta/episodes (LeRobot v3)
    import glob
    import pandas as pd
    tab = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(DS + "/meta/episodes/chunk-000/file-*.parquet"))], ignore_index=True)
    cols = [c for c in tab.columns if key in c and "from_timestamp" in c]
    fcols = [c for c in tab.columns if key in c and "file_index" in c]
    ccols = [c for c in tab.columns if key in c and "chunk_index" in c]
    row = tab.loc[tab["episode_index"] == ep] if "episode_index" in tab.columns else None
    if row is None or not len(row) or not cols:
        raise SystemExit("EP_META_MISSING cols=" + str(list(tab.columns)[:60]))
    return float(row[cols[0]].iloc[0]), int(row[fcols[0]].iloc[0]) if fcols else 0, int(row[ccols[0]].iloc[0]) if ccols else 0


def main():
    res = {}
    # sim: attempt-0000 (red->blue), attempt-0002 (orange->green)
    for att in ("attempt-0000", "attempt-0002"):
        ri = sim_release_idx(att)
        for cam, fn in (("overhead", "overhead.mp4"), ("wrist", "wrist.mp4")):
            png = f"{FR}/sim_{att}_{cam}_rel.png"
            ff_frame(f"{SIM}/{att}/{fn}", ri, png)
            res[f"sim_{att}_{cam}"] = {"release_frame": ri, **metrics(png)}
    # real: episode 7 (blue drop) top and close camera at release frame
    rel_frame = 1214
    for label, key in (("real_top", TOP), ("real_close", CLOSE)):
        try:
            t0, fidx, cidx = real_ep_time(7, key)
        except SystemExit as e:
            print(str(e)[:1500])
            continue
        vid = f"{DS}/videos/observation.images.{key}/chunk-{cidx:03d}/file-{fidx:03d}.mp4"
        png = f"{FR}/{label}_ep7_rel.png"
        ff_time(vid, t0 + rel_frame / 30.0, png)
        res[f"{label}_ep7"] = {"release_frame": rel_frame, "t0": t0, **metrics(png)}
    json.dump(res, open(OUT + "/camera_metrics.json", "w"), indent=1)
    for k, v in res.items():
        print(k, json.dumps(v))
    # contact sheet
    names = sorted(n for n in os.listdir(FR) if n.endswith(".png"))
    ims = [Image.open(os.path.join(FR, n)).convert("RGB").resize((320, 240)) for n in names]
    if ims:
        cols = 4
        rows = (len(ims) + cols - 1) // cols
        sheet = Image.new("RGB", (320 * cols, 240 * rows), (0, 0, 0))
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % cols) * 320, (i // cols) * 240))
        sheet.save(W + "/plots/camera_contact_sheet.png")
        print("SHEET", W + "/plots/camera_contact_sheet.png", "files", sorted(os.listdir(FR)))


if __name__ == "__main__":
    main()
