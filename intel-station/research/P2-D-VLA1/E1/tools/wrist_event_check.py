import csv
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/intel/intel-junction-2026/app/vla")
import build_dataset as B  # the builder's own segment clock and frame decoder

RUN = "/home/intel/intel-junction-2026/logs/sort/any_1740"
WRIST = "/home/intel/intel-junction-2026/data/wrist/raw"
PNG = "/home/intel/intel-junction-2026/work/P2-D-VLA1/E1/png/wrist_event"
EP = sys.argv[1]
TAG = EP.replace("_", "")
os.makedirs(PNG, exist_ok=True)

with open(os.path.join(RUN, EP, "joints.csv"), newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
t = np.array([float(r["t"]) for r in rows])
goal5 = np.array([float(r["goal5"]) for r in rows])
pos5 = np.array([float(r["pos5"]) for r in rows])
print(EP, "joint samples", len(t), "median dt ms %.1f" % (1000 * float(np.median(np.diff(t)))))


def biggest_fall(y, lo, hi):
    drop = y[5:] - y[:-5]
    lo, hi = max(lo, 0), min(hi, len(drop))
    return lo + int(np.argmin(drop[lo:hi]))


def midpoint_time(y, start):
    pre = y[max(start - 5, 0)]
    stop = min(start + 60, len(y))
    post = y[start:stop].min()
    mid = 0.5 * (pre + post)
    j = start + int(np.argmax(y[start:stop] <= mid))
    frac = (y[j - 1] - mid) / (y[j - 1] - y[j])
    return t[j - 1] + frac * (t[j] - t[j - 1]), pre, post


g_start = biggest_fall(goal5, 0, len(goal5))
t_goal, g_pre, g_post = midpoint_time(goal5, g_start)
p_start = biggest_fall(pos5, g_start - 10, g_start + 80)
t_pos, p_pre, p_post = midpoint_time(pos5, p_start)
print("goal5 fall starts t=%.3f, goal5 %.1f -> %.1f, midpoint t=%.4f" % (t[g_start], g_pre, g_post, t_goal))
print("pos5 fall starts t=%.3f, pos5 %.1f -> %.1f, midpoint t=%.4f" % (t[p_start], p_pre, p_post, t_pos))
print("goal5 midpoint to pos5 midpoint: %.1f ms" % (1000 * (t_pos - t_goal)))

segs = B.list_media(WRIST, "w")
a, b = t_pos - 1.0, t_pos + 1.0
cand = B.candidates(segs, a, b, B.WRIST_SEG_S)
print("wrist segments used:", [(os.path.basename(p), round(s, 3), "inf" if e == float("inf") else round(e, 3)) for p, s, e in cand])
frames = list(B.decode_window(cand, a, b))
ft = np.array([tt for tt, _ in frames])
gap = np.diff(ft)
print("wrist frames %d, frame interval ms mean %.2f max %.2f" % (len(ft), 1000 * gap.mean(), 1000 * gap.max()))
gray = [fr.reformat(format="gray").to_ndarray().astype(np.int16) for _, fr in frames]
energy = np.array([np.abs(gray[i + 1] - gray[i]).mean() for i in range(len(gray) - 1)])
e_t = ft[1:]
base = (e_t >= t_pos - 1.0) & (e_t <= t_pos - 0.5)
m, s = energy[base].mean(), energy[base].std()
thr = m + 4 * s
after = np.where((e_t >= t_pos - 0.5) & (energy > thr))[0]
ON = t_pos + 99  # outside the plotted window unless an onset is found
if len(after):
    on = e_t[after[0]]
    ON = on
    print("energy onset (> baseline +4 sd, baseline %.3f +- %.3f) at t=%.4f, %+.1f ms, %+.2f frames from pos5 midpoint"
          % (m, s, on, 1000 * (on - t_pos), 30 * (on - t_pos)))
else:
    print("no energy onset above threshold %.3f" % thr)

sel = [i for i in range(len(frames)) if t_pos - 0.5 <= ft[i] <= t_pos + 0.5]
cols, tw, th = 6, 240, 180
rows_n = (len(sel) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw, rows_n * th), (20, 20, 20))
draw = ImageDraw.Draw(sheet)
closest = min(sel, key=lambda i: abs(ft[i] - t_pos))
for k, i in enumerate(sel):
    tile = Image.fromarray(frames[i][1].reformat(width=tw, height=th, format="rgb24").to_ndarray())
    x, y = (k % cols) * tw, (k // cols) * th
    sheet.paste(tile, (x, y))
    stamp = time.strftime("%H:%M:%S", time.gmtime(ft[i])) + (".%03d" % (int(round(ft[i] * 1000)) % 1000))
    draw.rectangle([x, y, x + tw - 1, y + 16], fill=(0, 0, 0))
    draw.text((x + 3, y + 3), "%+d ms  %s UTC" % (round(1000 * (ft[i] - t_pos)), stamp), fill=(255, 255, 0))
    if i == closest:
        draw.rectangle([x, y, x + tw - 1, y + th - 1], outline=(255, 0, 0), width=3)
montage_path = os.path.join(PNG, TAG + "_wrist_montage.png")
sheet.save(montage_path)
print("montage", montage_path, "tiles", len(sel), "offsets ms", [round(1000 * (ft[i] - t_pos)) for i in sel][:3], "...")

W, H = 1000, 640
plot = Image.new("RGB", (W, H), (255, 255, 255))
pd_ = ImageDraw.Draw(plot)
x0, x1 = 80, W - 30
lo_t, hi_t = t_pos - 1.0, t_pos + 1.0


def xpix(tv):
    return x0 + (tv - lo_t) / (hi_t - lo_t) * (x1 - x0)


def panel(ytop, ybot, title, ylo, yhi, series):
    pd_.rectangle([x0, ytop, x1, ybot], outline=(0, 0, 0))
    pd_.text((x0 + 4, ytop + 4), title, fill=(0, 0, 0))
    for idx, (name, tv, yv, color) in enumerate(series):
        pts = []
        for tt, vv in zip(tv, yv):
            if lo_t <= tt <= hi_t:
                py = ybot - (vv - ylo) / (yhi - ylo + 1e-9) * (ybot - ytop - 20)
                pts.append((xpix(tt), py))
        if len(pts) > 1:
            pd_.line(pts, fill=color, width=2)
        pd_.text((x1 - 150, ytop + 4 + 14 * idx), name, fill=color)
    for tv, color in [(t_goal, (150, 150, 150)), (t_pos, (0, 0, 0)), (ON, (0, 170, 0))]:
        if lo_t <= tv <= hi_t:
            pd_.line([(xpix(tv), ytop), (xpix(tv), ybot)], fill=color, width=1)


sel_t = (t >= lo_t) & (t <= hi_t)
panel(10, 300, "joints: goal5 blue, pos5 red; grey line = goal5 midpoint, black = pos5 midpoint",
      min(goal5[sel_t].min(), pos5[sel_t].min()) - 2, max(goal5[sel_t].max(), pos5[sel_t].max()) + 2,
      [("goal5", t, goal5, (0, 80, 255)), ("pos5", t, pos5, (220, 0, 0))])
e_sel = (e_t >= lo_t) & (e_t <= hi_t)
panel(330, 620, "wrist mean abs frame difference (gray); green = energy onset", 0, float(energy[e_sel].max()) + 1,
      [("energy", e_t, energy, (0, 140, 0))])
trace_path = os.path.join(PNG, TAG + "_trace.png")
plot.save(trace_path)
print("trace", trace_path)
