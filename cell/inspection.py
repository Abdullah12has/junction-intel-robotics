"""Inspection camera + Anomalib OpenVINO. calibrate/holdout run on the station 14 Nov with synthetic images;
capture/empty/grab (the camera paths) are still untested.

python cell/inspection.py capture train/good   # SPACE saves a crop, q quits
python cell/inspection.py empty                # save the empty-mark reference
python cell/inspection.py calibrate            # thresholds.json from val scores
python cell/inspection.py holdout              # held-out result; run once per model

The inspection camera is NOT in the Studio environment; this file owns it.
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CAM = "/dev/video16"            # check with v4l2-ctl --list-devices; numbers change
ROI = (420, 180, 360, 360)      # x, y, w, h: measure once, then never move the camera
SIZE = 256
DEVICE = os.environ.get("INSPECT_DEVICE", "NPU")  # explicit; never AUTO. Use GPU until the NPU driver is installed
XML = "export/patchcore/weights/openvino/model.xml"
DATA = Path("data/bricks")
EMPTY_REF, PRESENT_THR = Path("empty_ref.png"), 12.0  # mean abs grey-level diff; tune on real images

_cap = None
_inf = None


def grab():
    """Fresh frame: drop buffered ones so we never classify an image from before the brick arrived."""
    global _cap
    if _cap is None:
        _cap = cv2.VideoCapture(CAM)
    for _ in range(5):
        _cap.grab()
    ok, frame = _cap.read()
    if not ok:
        raise RuntimeError("inspection camera returned no frame")
    return frame


def crop(frame):  # the ONLY crop/resize; used for training images and live, so they cannot differ
    x, y, w, h = ROI
    return cv2.resize(frame[y:y + h, x:x + w], (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def present(c):
    """Is something on the mark? Compares with the empty-mark reference."""
    return float(np.abs(c.astype(np.int16) - cv2.imread(str(EMPTY_REF)).astype(np.int16)).mean()) > PRESENT_THR


def score(c):
    global _inf
    if _inf is None:
        from anomalib.deploy import OpenVINOInferencer
        _inf = OpenVINOInferencer(path=XML, device=DEVICE, config={"PERFORMANCE_HINT": "LATENCY"})
        print("requested", DEVICE, "executing on", _inf.model.get_property("EXECUTION_DEVICES"))
        for _ in range(5):  # warm-up; first NPU compile is slow
            _inf.predict(np.zeros((SIZE, SIZE, 3), np.uint8))  # noqa
    pred = _inf.predict(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))  # model normalises internally; do not do it again
    return float(np.asarray(pred.pred_score).reshape(-1)[0]), np.asarray(pred.anomaly_map)


def classify(c, th):
    """-> (verdict, score, ms, anomaly_map). Ambiguous is treated as defective by the caller: a false reject beats a false accept."""
    t0 = time.perf_counter()
    s, amap = score(c)
    v = "defective" if s > th["hi"] else "good" if s <= th["lo"] else "ambiguous"
    return v, s, (time.perf_counter() - t0) * 1e3, amap


def _folder(sub):
    return [cv2.imread(str(p)) for p in sorted((DATA / sub).glob("*.png"))]


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "capture":
        out = DATA / sys.argv[2]
        out.mkdir(parents=True, exist_ok=True)
        while True:
            c = crop(grab())
            cv2.imshow("crop", c)
            k = cv2.waitKey(30) & 0xFF
            if k == ord(" "):
                cv2.imwrite(str(out / f"{time.time_ns()}.png"), c)
                print("saved", len(list(out.glob("*.png"))))
            elif k == ord("q"):
                break
    elif mode == "empty":
        cv2.imwrite(str(EMPTY_REF), crop(grab()))
    elif mode == "calibrate":
        g = np.array([score(im)[0] for im in _folder("val/good")])
        d = np.array([score(im)[0] for im in _folder("val/defect")])
        if g.max() < d.min():   # clean separation: one cut in the middle, no ambiguous band
            th = {"lo": float((g.max() + d.min()) / 2)}
            th["hi"] = th["lo"]
        else:                   # overlap: everything between the lowest defect and the highest good is ambiguous
            th = {"lo": float(d.min()), "hi": float(g.max())}
        json.dump(th, open("thresholds.json", "w"))
        print(f"good max {g.max():.3f}  defect min {d.min():.3f}  band {th}")
    elif mode == "holdout":
        th = json.load(open("thresholds.json"))
        for sub in ("test_holdout/good", "test_holdout/defect"):
            r = [classify(im, th) for im in _folder(sub)]
            print(sub, {k: sum(x[0] == k for x in r) for k in ("good", "ambiguous", "defective")},
                  "p50 ms", np.median([x[2] for x in r]))
