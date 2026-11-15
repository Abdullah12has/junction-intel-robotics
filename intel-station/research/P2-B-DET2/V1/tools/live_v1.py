"""V1 live-frame diagnostics (analysis only; not part of the app). usage: python3 -I -B live_v1.py APP_DIR
Dumps the raw foreground components of the live top and close stills, HSV at the known brick and mat points,
the bottom-rail row profile, the scan_full result, and exposure statistics of a dataset frame for comparison."""
import sys

import cv2
import numpy as np

APP = sys.argv[1]
sys.path.insert(0, APP)
import inspector as I  # noqa: E402

LIVE = "/home/intel/intel-junction-2026/logs/detlive/"
DS = "/home/intel/intel-junction-2026/work/P2-B-DET2/E2/samples/"
top = cv2.imread(LIVE + "top_153426.jpg")
close = cv2.imread(LIVE + "close_153426.jpg")
ds_top = cv2.imread(DS + "6d6bdce3_e09_end_243622061421.png")
ds_close = cv2.imread(DS + "6d6bdce3_e09_end_243722064316.png")
print("shapes live top", top.shape, "close", close.shape, "| dataset top", ds_top.shape, "close", ds_close.shape)


def med_hsv(img, x, y, r=4):
    h, s, v = I._split(img)
    sl = (slice(max(y - r, 0), y + r + 1), slice(max(x - r, 0), x + r + 1))
    return [round(float(np.median(a[sl])), 1) for a in (h, s, v)]


TOP_PTS = {"red_400_218": (400, 218), "white24_420_272": (420, 272), "white22_410_322": (410, 322),
           "yellow_474_262": (474, 262), "mat_470_320": (470, 320), "bluepl_230_160": (230, 160)}
CLOSE_PTS = {"red_370_180": (370, 180), "white24_500_255": (500, 255), "white22_560_380": (560, 380),
             "yellow_580_200": (580, 200), "mat_250_300": (250, 300)}
for k, (x, y) in TOP_PTS.items():
    print("TOP", k, med_hsv(top, x, y))
for k, (x, y) in CLOSE_PTS.items():
    print("CLOSE", k, med_hsv(close, x, y))
print("DS top mat_470_320", med_hsv(ds_top, 470, 320), "DS close mat_250_300", med_hsv(ds_close, 250, 300))

h, s, v = I._split(top)
print("rail rows (x 230..500): y mean_V mean_S")
for y in range(326, 372, 4):
    print("  ", y, round(float(v[y, 230:500].mean()), 1), round(float(s[y, 230:500].mean()), 1))

roi = I._roi(h.shape, I.TOP_ROI)
robot = np.zeros(h.shape, np.uint8)
cv2.fillPoly(robot, [I.ROBOT_POLY], 1)
fg = roi & (v > 70) & (robot == 0)
fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, I.K3).astype(bool)
print("TOP raw fg components (area>=200):")
for m, a, st in I._components(fg, 200):
    print("  area", a, "bbox", st[:4].tolist(), "medHSV", [round(float(np.median(x[m])), 1) for x in (h, s, v)])

hc, sc, vc = I._split(close)
roi_c = I._roi(hc.shape, I.CLOSE_ROI)
fgc = cv2.morphologyEx((roi_c & (((sc > 85) & (vc > 90)) | ((vc > 170) & (sc < 70)))).astype(np.uint8),
                       cv2.MORPH_OPEN, I.K3).astype(bool)
print("CLOSE raw fg components (area>=300):")
for m, a, st in I._components(fgc, 300):
    print("  area", a, "bbox", st[:4].tolist(), "medHSV", [round(float(np.median(x[m])), 1) for x in (hc, sc, vc)])

hd, sd, vd = I._split(ds_close)
mat_d = I._roi(hd.shape, I.CLOSE_ROI) & (vd < 80)
print("DS close mat V median", round(float(np.median(vd[mat_d])), 1), "live close mat V median",
      round(float(np.median(vc[I._roi(hc.shape, I.CLOSE_ROI) & (vc < 80)])), 1))
try:
    res = I.scan_full(top, close)
    print("scan_full hands", res["hands"], "plates", res["plates_rejected"], "partial_close", res["partial_at_border"],
          "rejected_close", res["rejected"])
    for b in res["bricks"]:
        print("  BRICK", b["id"], b["view"], b["colour"], b["hsv_med"], b["shape"], b["aspect"], b["area"],
              b["centre_px"], "merged", b["merged"], b.get("merged_reason"), "split", b.get("split"),
              "tape", b["tape_score"], b["verdict"], b["reasons"])
except I.HandsPresent as exc:
    print("REFUSED", exc)
