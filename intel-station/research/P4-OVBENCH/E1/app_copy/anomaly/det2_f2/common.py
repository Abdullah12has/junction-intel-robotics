"""Shared constants and helpers for the P2-B-DET2 E2 per-brick verdict build (laptop side)."""
import os

import cv2
import numpy as np

ROOT = "/home/intel/intel-junction-2026"
APP = os.path.join(ROOT, "app/anomaly/det2_e2")
WORK = os.path.join(ROOT, "work/P2-B-DET2/E2")
DATA = os.path.join(ROOT, "data/studio-datasets")
TOP = "realsense 243622061421"  # top view (dataset key)
CLOSE = "realsense 243722064316"  # close low view (dataset key)
HOME = np.array([-6.0, -98.2, 96.6, 60.5, -48.3])  # pan, lift, elbow, wrist_flex, wrist_roll (median)
FPS = 30


def skin_fraction(bgr):
    """Fraction of pixels with a skin-like hue, used to flag hands in the frame."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h <= 18) | (h >= 172)) & (s >= 50) & (s <= 170) & (v >= 90)
    return float(mask.mean()), mask
