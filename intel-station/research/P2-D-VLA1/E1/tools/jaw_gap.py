import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/intel/intel-junction-2026/app/vla")
import build_dataset as B  # the builder's own segment clock and frame decoder

RUN = "/home/intel/intel-junction-2026/logs/sort/any_1740"
WRIST = "/home/intel/intel-junction-2026/data/wrist/raw"
PNG = "/home/intel/intel-junction-2026/work/P2-D-VLA1/E1/png/wrist_event"
EP = sys.argv[1]
C = float(sys.argv[2])  # window centre, epoch seconds (the pos5 midpoint from wrist_event_check.py)
TAG = EP.replace("_", "")
os.makedirs(PNG, exist_ok=True)

with open(os.path.join(RUN, EP, "joints.csv"), newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
t = np.array([float(r["t"]) for r in rows])
goal5 = np.array([float(r["goal5"]) for r in rows])
pos5 = np.array([float(r["pos5"]) for r in rows])
w = (t >= C - 1.0) & (t <= C + 1.0)
tj, gj, pj = t[w], goal5[w], pos5[w]


def half_travel(ts, ys):
    """Time at which ys first reaches the midpoint of its first and last value (linear interpolation)."""
    ok = np.isfinite(ys)
    ts, ys = ts[ok], ys[ok]
    v0, v1 = ys[0], ys[-1]
    mid = 0.5 * (v0 + v1)
    s = np.sign(v1 - v0)
    idx = np.where((ys - mid) * s >= 0)[0]
    if len(idx) == 0 or idx[0] == 0:
        return None
    j = idx[0]
    frac = (ys[j - 1] - mid) / (ys[j - 1] - ys[j])
    return ts[j - 1] + frac * (ts[j] - ts[j - 1])


def jaw_gap(img):
    """Pixel gap between the inner edges of the two red fingers in the bottom quarter of a wrist frame."""
    band = img[360:480]
    r, g, bl = band[..., 0], band[..., 1], band[..., 2]
    m = (r > 120) & (g < 100) & (bl < 120) & (r - g > 90)
    col = m.sum(axis=0)
    left = np.where(col[:320] > 3)[0]
    right = np.where(col[320:] > 3)[0]
    if len(left) == 0 or len(right) == 0:
        return np.nan, m
    return float((right[0] + 320) - left[-1]), m


tg, tp = half_travel(tj, gj), half_travel(tj, pj)
print(EP, "window joints %d samples, goal5 %.1f -> %.1f, pos5 %.1f -> %.1f" % (len(tj), gj[0], gj[-1], pj[0], pj[-1]))
print("goal5 half-travel t=%.4f, pos5 half-travel t=%.4f, pos5 - goal5 %.1f ms" % (tg, tp, 1000 * (tp - tg)))

segs = B.list_media(WRIST, "w")
a, b = C - 1.0, C + 1.0
cand = B.candidates(segs, a, b, B.WRIST_SEG_S)
print("wrist segments:", [(os.path.basename(p), round(s, 3)) for p, s, e in cand])
frames = list(B.decode_window(cand, a, b))
ft = np.array([tt for tt, _ in frames])
imgs = [fr.reformat(format="rgb24").to_ndarray().astype(np.int16) for _, fr in frames]
gaps, masks = [], []
for img in imgs:
    gp, m = jaw_gap(img)
    gaps.append(gp)
    masks.append(m)
gaps = np.array(gaps)
gaps_s = np.array([np.nanmedian(gaps[max(0, i - 1):i + 2]) for i in range(len(gaps))])
tgap = half_travel(ft, gaps_s)
print("wrist frames %d, NaN gaps %d, gap start %.1f px, end %.1f px (3-frame median)" % (len(ft), int(np.isnan(gaps).sum()), gaps_s[0], gaps_s[-1]))
print("jaw-gap half-travel t=%.4f" % tgap)
print("jaw-gap - goal5 half-travel: %+.1f ms = %+.2f frames" % (1000 * (tgap - tg), 30 * (tgap - tg)))
print("jaw-gap - pos5 half-travel: %+.1f ms = %+.2f frames" % (1000 * (tgap - tp), 30 * (tgap - tp)))
print("frame table (ms from pos5 half-travel, gap px raw, gap px median3):")
for i in range(len(ft)):
    if abs(ft[i] - tp) <= 0.6:
        print("  %+5d  %6.1f  %6.1f" % (round(1000 * (ft[i] - tp)), gaps[i], gaps_s[i]))

k = int(np.argmin(np.abs(ft - tgap)))
frame_img = Image.fromarray(imgs[k].astype(np.uint8))
mask_img = Image.fromarray((masks[k] * 255).astype(np.uint8)).convert("RGB")
pair = Image.new("RGB", (1280, 480))
pair.paste(frame_img, (0, 0))
pair.paste(mask_img, (640, 0))
dr = ImageDraw.Draw(pair)
dr.text((8, 8), "wrist frame %+d ms from pos5 half-travel, gap %.0f px" % (round(1000 * (ft[k] - tp)), gaps[k]), fill=(255, 255, 0))
dr.text((648, 8), "red-finger mask used for the gap (bottom quarter)", fill=(255, 255, 0))
pair_path = os.path.join(PNG, TAG + "_jaw_mask_frame.png")
pair.save(pair_path)

W, H = 1000, 560
plot = Image.new("RGB", (W, H), (255, 255, 255))
pd_ = ImageDraw.Draw(plot)
x0, x1 = 80, W - 30
lo_t, hi_t = C - 1.0, C + 1.0


def xpix(tv):
    return x0 + (tv - lo_t) / (hi_t - lo_t) * (x1 - x0)


def panel(ytop, ybot, title, series, marks):
    pd_.rectangle([x0, ytop, x1, ybot], outline=(0, 0, 0))
    pd_.text((x0 + 4, ytop + 4), title, fill=(0, 0, 0))
    allv = np.concatenate([np.asarray(v)[np.isfinite(v)] for _, _, v, _ in series])
    ylo, yhi = float(allv.min()) - 1, float(allv.max()) + 1
    for idx, (name, tv, yv, color) in enumerate(series):
        pts = []
        for tt, vv in zip(tv, yv):
            if np.isfinite(vv):
                pts.append((xpix(tt), ybot - (vv - ylo) / (yhi - ylo) * (ybot - ytop - 22)))
        if len(pts) > 1:
            pd_.line(pts, fill=color, width=2)
        pd_.text((x1 - 160, ytop + 4 + 14 * idx), name, fill=color)
    for tv, color in marks:
        if tv is not None and lo_t <= tv <= hi_t:
            pd_.line([(xpix(tv), ytop), (xpix(tv), ybot)], fill=color, width=1)


panel(10, 270, "joints (units): goal5 blue, pos5 red; lines = half-travel times",
      [("goal5", tj, gj, (0, 80, 255)), ("pos5", tj, pj, (220, 0, 0))],
      [(tg, (0, 80, 255)), (tp, (220, 0, 0))])
panel(290, 550, "wrist jaw gap (px, bottom quarter, median of 3 frames); black line = gap half-travel",
      [("gap", ft, gaps_s, (0, 0, 0))], [(tgap, (0, 0, 0)), (tp, (220, 0, 0))])
trace_path = os.path.join(PNG, TAG + "_jaw_gap_trace.png")
plot.save(trace_path)
print("saved", pair_path, trace_path)
