"""Replay closeloc on recorded episodes: which brick does each close view judge, and with what verdict.
usage: closeloc_replay.py RUN [RUN ...]   -> prints one block per episode, writes logs/depth/servo/closeloc_<run>_<ep>.jpg"""
import glob
import json
import os
import sys

import cv2
import numpy as np

R = "/home/intel/intel-junction-2026"
sys.path.insert(0, f"{R}/app/tools")
sys.path.insert(0, f"{R}/app/anomaly/det2_f2")
import closeloc as cl  # noqa: E402
import inspector as DET  # noqa: E402

OLD_SPOT = (380, 262)


def mask_jaws(img):
    """Same rule as sort_loop.mask_jaws: red parts touching the image top are the gripper; blank them."""
    red = cl.red_mask(img)
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    out = img.copy()
    for i in range(1, n):
        if st[i, 1] <= 3 and st[i, 4] > 800:
            out[cv2.dilate((lab == i).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = 0
    return out


def verdict(rec, thr):
    if rec is None:
        return None
    v, reasons = DET.triage(rec, thr)
    return {"verdict": v, "reasons": list(reasons), "colour": rec.get("colour"), "tape": rec.get("tape_score"),
            "centre": [round(c) for c in rec["centre_px"]], "bbox": [int(b) for b in rec["bbox"]]}


def box(img, rec, colour, label):
    if rec is None:
        return
    x, y, w, h = (int(b) for b in rec["bbox"])
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
    cv2.putText(img, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)


thr = DET.load_thresholds()
for run in sys.argv[1:]:
    for ep in sorted(glob.glob(f"{R}/logs/sort/{run}/ep_*")):
        pre = cv2.imread(f"{ep}/pre_close.jpg")
        if pre is None:
            continue
        out = {"ep": os.path.basename(ep)}
        seg_pre = DET.segment_close(mask_jaws(pre))["bricks"]
        out["n_pre"] = len(seg_pre)
        sheet = [pre.copy()]
        st = cv2.imread(f"{ep}/station_close.jpg")
        if st is not None:
            ch = cl.changed_mask(st, pre)
            seg_st = DET.segment_close(mask_jaws(st))["bricks"]
            app, fa, fa2 = cl.pick_changed(seg_st, ch)
            van, fv, fv2 = cl.pick_changed(seg_pre, ch)
            old, dold = cl.nearest(seg_st, OLD_SPOT[0], OLD_SPOT[1], 170)
            out.update({"appeared": verdict(app, thr), "appeared_frac": [fa, fa2],
                        "vanished": verdict(van, thr), "vanished_frac": [fv, fv2],
                        "old_spot_choice": verdict(old, thr), "changed_px": int(ch.sum())})
            vis = st.copy()
            vis[ch > 0] = (0.5 * vis[ch > 0] + [0, 0, 127]).astype(np.uint8)
            box(vis, app, (0, 255, 0), "appeared")
            box(vis, old, (0, 0, 255), "old spot")
            cv2.circle(vis, OLD_SPOT, 6, (0, 0, 255), 2)
            box(sheet[0], van, (255, 128, 0), "vanished")
            sheet.append(vis)
        hv = cv2.imread(f"{ep}/stage1_close.jpg")
        if hv is not None:
            tips = cl.jaw_tips_by_diff(hv, pre)
            out["jaw_tips"] = [[round(a), round(b)] for a, b in tips]
            if len(tips) == 2:
                mx, my = (tips[0][0] + tips[1][0]) / 2, max(tips[0][1], tips[1][1])
                under, d = cl.brick_below(seg_pre, mx, my)
                out["under_jaws"] = verdict(under, thr)
                out["under_jaws_d"] = d
                box(sheet[0], under, (255, 0, 255), "under jaws")
                vis2 = hv.copy()
                for x, y in tips:
                    cv2.circle(vis2, (int(x), int(y)), 7, (255, 0, 255), 2)
                sheet.append(vis2)
        print(json.dumps(out))
        os.makedirs(f"{R}/logs/depth/servo", exist_ok=True)
        cv2.imwrite(f"{R}/logs/depth/servo/closeloc_{run}_{out['ep']}.jpg",
                    np.hstack([cv2.resize(s, (480, 360)) for s in sheet]), [cv2.IMWRITE_JPEG_QUALITY, 80])
