"""Hands-free capture session for anom3 v002: saves a top/close pair each time the close view changes and then stays
still for 1.5 s with no hands in the top view. Stops after --max captures or --minutes, and always releases both cameras.
Run: nohup nice -n 10 envs/anomalib-env/bin/python -I -B work/P2-B-ANOM3/v002_close/auto_capture.py > LOG 2>&1 &
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import capture as C  # noqa: E402

SHUTTER = "/usr/share/sounds/freedesktop/stereo/camera-shutter.oga"


def open_cam(dev):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        sys.exit("cannot open " + dev)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=15)
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--still", type=float, default=1.5, help="mean abs grey difference that counts as still")
    ap.add_argument("--changed", type=float, default=2.0, help="mean abs grey difference from the last capture")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    busy = subprocess.run(["fuser", "/dev/video8", "/dev/video14"], capture_output=True, text=True).stdout.strip()
    if busy:
        sys.exit("cameras busy: " + busy)
    top_cap, close_cap = open_cam("/dev/video8"), open_cam("/dev/video14")
    print("cameras open; capturing up to %d scenes for %.0f min" % (a.max, a.minutes), flush=True)
    t_end, next_t, saved, stable, prev, last = time.time() + a.minutes * 60, 0.0, 0, 0, None, None
    try:
        for _ in range(30):
            top_cap.grab(), close_cap.grab()
        while time.time() < t_end and saved < a.max:
            top_cap.grab(), close_cap.grab()
            if time.time() < next_t:
                continue
            next_t = time.time() + 0.5
            ok1, top = top_cap.retrieve()
            ok2, close = close_cap.retrieve()
            if not (ok1 and ok2):
                continue
            small = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(close, cv2.COLOR_BGR2GRAY), (160, 120)), (5, 5), 0).astype(np.float32)
            d_prev = float(np.mean(np.abs(small - prev))) if prev is not None else -1.0
            stable = stable + 1 if 0 <= d_prev < a.still else 0
            prev = small
            if a.debug:
                print("d_prev %.2f stable %d d_last %s" % (d_prev, stable, "%.2f" % float(np.mean(np.abs(small - last))) if last is not None else "-"), flush=True)
            if stable < 3 or (last is not None and float(np.mean(np.abs(small - last))) < a.changed):
                continue
            try:
                res = C.I.scan_full(top, close)
            except C.I.HandsPresent as e:
                if a.debug:
                    print("hands:", e, flush=True)
                continue
            C.harvest(C.next_dir(), top, close, res)
            last, saved, stable = small, saved + 1, 0
            player = shutil.which("ffplay")
            if player and os.path.exists(SHUTTER):  # best effort: a shutter sound tells the operator to shuffle
                env = dict(os.environ, XDG_RUNTIME_DIR=os.environ.get("XDG_RUNTIME_DIR", "/run/user/%d" % os.getuid()))
                subprocess.Popen([player, "-nodisp", "-autoexit", "-loglevel", "quiet", SHUTTER], env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        top_cap.release()
        close_cap.release()
        print("done: %d captures; cameras released" % saved, flush=True)


if __name__ == "__main__":
    main()
