# V1 verifier: candidate bricks in the START top view (mat ROI) and a colour-matched presence test in the LIFT frame.
# The grasped brick is the candidate whose colour is gone from its start location 1 s after the grasp.
import json, sys
import cv2
import numpy as np

OUT = "/home/intel/intel-junction-2026/work/P2-R8-PICK/V1"
ROI = (300, 60, 540, 320)  # x0, y0, x1, y1 of the cluster area on the mat (plates and cable tape are outside)


def masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = [c.astype(np.int16) for c in cv2.split(hsv)]
    col = (S > 110) & (V > 90)
    white = (S < 45) & (V > 185)
    yel = (H >= 18) & (H <= 35) & (S > 70) & (V > 120)
    return H, col, white, yel


def components(bgr):
    H, col, white, yel = masks(bgr)
    out = []
    for kind, m in (("col", col), ("white", white), ("yel", yel)):
        mm = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(mm)
        for i in range(1, n):
            x, y, w, h, a = [int(v) for v in st[i]]
            cx, cy = float(cen[i][0]), float(cen[i][1])
            if a < 300 or a > 2600 or max(w, h) > 75:
                continue
            if not (ROI[0] <= cx <= ROI[2] and ROI[1] <= cy <= ROI[3]):
                continue
            sel = lab == i
            out.append({"kind": kind, "cx": round(cx, 1), "cy": round(cy, 1), "area": a, "bbox": [x, y, w, h],
                        "hue": float(np.median(H[sel])), "sel": sel})
    return out


def presence(c, lift_bgr):
    H, col, white, yel = masks(lift_bgr)
    sel = c["sel"]
    if c["kind"] == "white":
        return float(white[sel].mean())
    if c["kind"] == "yel":
        return float(yel[sel].mean())
    return float((col & (np.abs(H - c["hue"]) <= 8))[sel].mean())


def main(name_list):
    res = {}
    for name in name_list:
        s = cv2.imread("%s/frames/%s_start.png" % (OUT, name))
        l = cv2.imread("%s/frames/%s_lift.png" % (OUT, name))
        rows = []
        for c in components(s):
            rows.append({"kind": c["kind"], "cx": c["cx"], "cy": c["cy"], "area": c["area"], "bbox": c["bbox"],
                         "hue": round(c["hue"], 1), "presence_lift": round(presence(c, l), 3)})
        rows.sort(key=lambda r: r["presence_lift"])
        res[name] = rows
        print(name, "roi_candidates", len(rows))
        for r in rows:
            print("   %-5s cx %6.1f cy %6.1f area %4d hue %5.1f presence %.2f" % (r["kind"], r["cx"], r["cy"], r["area"], r["hue"], r["presence_lift"]))
    json.dump(res, open(OUT + "/labels_candidates.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
