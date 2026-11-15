"""Per-file plate detections for the real top and close videos (data only, read-only).

Uses the same HSV thresholds, 5x5 opening and largest-component rule as tools/sim_real_compare.py,
and prints, for each camera and each chunk-000 file, the detection count, the median centroid,
and how many detections lie more than 20 px from that file's median (possible misdetections or
occlusion). Writes nothing. Reproducing the file-000 numbers of the v1 run (42 frames, blue cx
244.27, green cx 578.48) is the check that the detection logic matches.
Run with: envs/data-env/bin/python -I tools/real_plate_perfile.py
"""
import subprocess

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
REAL = R + "/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
CAMS = {"top": "observation.images.realsense 243622061421",
        "close": "observation.images.realsense 243722064316"}
W0, H0 = 640, 480
HSV = {"blue": ((80, 105), 90, 100), "green": ((50, 90), 60, 60)}
MIN_AREA = 4000
OUTLIER_PX = 20.0


def frames(path):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-vf", "fps=1",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = W0 * H0 * 3
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield np.frombuffer(b, np.uint8).reshape(H0, W0, 3).copy()
    finally:
        p.stdout.close()
        p.wait()


def plate(bgr, color):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    (h0, h1), smin, vmin = HSV[color]
    m = cv2.inRange(hsv, (h0, smin, vmin), (h1, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    area = int(st[k, cv2.CC_STAT_AREA])
    if area < MIN_AREA:
        return None
    ys, xs = np.nonzero(lab == k)
    return float(xs.mean()), float(ys.mean()), area


def summarise(dets):
    if not dets:
        return {"n": 0}
    cx = np.array([d[1] for d in dets])
    cy = np.array([d[2] for d in dets])
    ar = np.array([d[3] for d in dets])
    mcx, mcy = float(np.median(cx)), float(np.median(cy))
    far = [d for d in dets if abs(d[1] - mcx) > OUTLIER_PX]
    return {"n": len(dets), "med_cx": round(mcx, 2), "med_cy": round(mcy, 2),
            "med_area": float(np.median(ar)), "cx_sd": round(float(cx.std()), 2),
            "cy_sd": round(float(cy.std()), 2), "n_far_gt_20px": len(far),
            "far_times_s": [d[0] for d in far[:12]],
            "far_cx": [round(d[1], 1) for d in far[:12]],
            "far_area": [d[3] for d in far[:12]]}


for cam, folder in CAMS.items():
    all_det = {"blue": [], "green": []}
    total = 0
    for k in range(4):
        path = "%s/videos/%s/chunk-000/file-00%d.mp4" % (REAL, folder, k)
        det = {"blue": [], "green": []}
        t = 0
        for fr in frames(path):
            for c in ("blue", "green"):
                d = plate(fr, c)
                if d:
                    det[c].append((t, d[0], d[1], d[2]))
            t += 1
        total += t
        for c in ("blue", "green"):
            all_det[c].extend(det[c])
            print(cam, "file-00%d" % k, c, "frames", t, json_like := "", summarise(det[c]))
    for c in ("blue", "green"):
        print(cam, "ALL FILES", c, summarise(all_det[c]))
    print(cam, "total frames", total)
