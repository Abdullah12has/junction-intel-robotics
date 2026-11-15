#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 11b: picked-brick vanish events and plate centres in the top view (read-only on the frames).

Event = one picked brick (lift TCP from kin, out/s11_inventory.json). Before frame: pre_top.jpg of that episode (arm at
the scan pose). After frame: released_top.jpg of the same episode when it exists (arm at the plate, so the gripper
itself can produce false vanish candidates), else pre_top.jpg of the next episode of the same run (arm back at the scan
pose, the cleaner case). Vanish candidates = saturated pixels in the before frame that are absent in the after frame
(dilated 7 px), 150..4000 px. Plate centres = ellipse fits to the blue and green plate outlines in one pre_top frame.
Writes out/s11_events.json and overlays/s11_events_overlay.jpg for viewing.
Run: envs/data-env/bin/python -I tools/s11b_events.py
"""
import json
import os

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-CAMS/E1"
LOG = f"{R}/logs/sort"


def sat_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 1] >= 90) & (hsv[..., 2] >= 90)).astype(np.uint8)


def vanish_candidates(before, after):
    mb, ma = sat_mask(before), sat_mask(after)
    ma_d = cv2.dilate(ma, np.ones((7, 7), np.uint8))
    van = ((mb == 1) & (ma_d == 0)).astype(np.uint8)
    van = cv2.morphologyEx(van, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(van, 8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if 150 <= a <= 4000:
            out.append({"u": round(float(cents[i][0]), 1), "v": round(float(cents[i][1]), 1), "area": a})
    return out


def plate_centre(img, lo, hi, smin, vmin):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= lo) & (hsv[..., 0] <= hi) & (hsv[..., 1] >= smin) & (hsv[..., 2] >= vmin)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if len(c) < 5:
        return None
    (cu, cv_), (w, h), ang = cv2.fitEllipse(c)
    return {"u": round(float(cu), 1), "v": round(float(cv_), 1), "axes_px": [round(float(w), 1), round(float(h), 1)],
            "area_px": int(cv2.contourArea(c))}


def main():
    inv = json.load(open(f"{E1}/out/s11_inventory.json"))
    by_ep = {r["ep"]: r for r in inv}
    events, tiles = [], []
    for r in inv:
        if not (r["lift_tcp_mm"] and r["pre_top"]):
            continue
        ep_dir = os.path.join(LOG, r["ep"])
        before_p = os.path.join(ep_dir, "pre_top.jpg")
        if r["released_top"]:
            after_p, src = os.path.join(ep_dir, "released_top.jpg"), "released_top (same episode)"
        else:
            run, num = r["ep"].split("/")
            nxt = f"{run}/ep_{int(num.split('_')[1]) + 1:04d}"
            if nxt in by_ep and by_ep[nxt]["pre_top"]:
                after_p, src = os.path.join(LOG, nxt, "pre_top.jpg"), f"pre_top of {nxt}"
            else:
                continue
        b, a = cv2.imread(before_p), cv2.imread(after_p)
        if b is None or a is None:
            continue
        cands = vanish_candidates(b, a)
        ev = {"ep": r["ep"], "after_source": src, "lift_tcp_mm_xy": r["lift_tcp_mm"][:2],
              "lift_tcp_z_mm": r["lift_tcp_mm"][2], "colour_from_episode": r["colour"],
              "success": r["success"], "n_candidates": len(cands), "candidates": cands,
              "image_size": [int(b.shape[1]), int(b.shape[0])]}
        events.append(ev)
        tile = b.copy()
        for c in cands:
            cv2.circle(tile, (int(c["u"]), int(c["v"])), 9, (255, 255, 0), 2)
        cv2.putText(tile, f"{r['ep']} n={len(cands)}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(cv2.resize(tile, (320, 240)))

    first_pre = next(os.path.join(LOG, r["ep"], "pre_top.jpg") for r in inv if r["pre_top"])
    fimg = cv2.imread(first_pre)
    plates = {"blue": plate_centre(fimg, 85, 105, 80, 80), "green": plate_centre(fimg, 40, 85, 60, 60)}
    tile = fimg.copy()
    for name, pc in plates.items():
        if pc:
            cv2.drawMarker(tile, (int(pc["u"]), int(pc["v"])), (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
            cv2.putText(tile, name, (int(pc["u"]) + 12, int(pc["v"])), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)
    tiles.insert(0, cv2.resize(tile, (320, 240)))

    while len(tiles) % 4:
        tiles.append(np.zeros((240, 320, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(f"{E1}/overlays/s11_events_overlay.jpg", np.vstack(rows))

    out = {"events": events, "plate_centres_px_first_pre_top": plates,
           "plate_centres_base_mm": {"blue": [187.1, -86.1], "green": [194.6, 218.3]},
           "note": "candidates are unverified until viewed; before/after frames from the same camera; arm at scan pose"}
    json.dump(out, open(f"{E1}/out/s11_events.json", "w"), indent=1)
    print("events", len(events), [(e["ep"], e["after_source"], e["n_candidates"]) for e in events], flush=True)
    print("plates (first pre_top frame)", plates, flush=True)


if __name__ == "__main__":
    main()
