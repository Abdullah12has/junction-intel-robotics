"""Shared causal marker and train-only augmentation, also usable at deployment."""

import cv2
import numpy as np


class TargetTracker:
    """Fixed appearance, local forward tracking; suppress permanently on ambiguity.

    No future frames, object poses, action labels, or phase annotations are inputs.
    Reinitialization requires a new operator click / new episode.
    """

    def __init__(self, rgb, xy, half=12):
        self.xy = np.asarray(xy, dtype=float)
        self.origin = self.xy.copy()
        self.half = half
        self.lost = False
        x, y = np.rint(self.xy).astype(int)
        self.template = rgb[y - half : y + half + 1, x - half : x + half + 1].copy()
        if self.template.shape != (2 * half + 1, 2 * half + 1, 3):
            raise ValueError("Initial target outside image")

    def update(self, rgb):
        if self.lost:
            return [-1.0, -1.0], 0.0
        x, y = np.rint(self.xy).astype(int)
        r = 18
        h = self.half
        x0 = max(0, x - h - r)
        y0 = max(0, y - h - r)
        x1 = min(rgb.shape[1], x + h + r + 1)
        y1 = min(rgb.shape[0], y + h + r + 1)
        search = rgb[y0:y1, x0:x1]
        score = cv2.matchTemplate(search, self.template, cv2.TM_SQDIFF_NORMED)
        best, _, loc, _ = cv2.minMaxLoc(score)
        xy = np.array([x0 + loc[0] + h, y0 + loc[1] + h], dtype=float)
        if (
            best > 0.13
            or np.linalg.norm(xy - self.xy) > 20
            or np.linalg.norm(xy - self.origin) > 8
        ):
            self.lost = True
            return [-1.0, -1.0], float(1 - best)
        self.xy = xy
        return xy.tolist(), float(1 - best)


def transform(rgb, recipe, rng, overhead=False, xy=None):
    im = rgb.astype(np.float32) / 255
    point = None if xy is None or min(xy) < 0 else np.array(xy, dtype=float)
    h, w = im.shape[:2]
    if recipe >= 1 and rng.random() < 0.5:
        brightness, contrast = rng.uniform(0.9, 1.1, 2)
        mean = im.mean(axis=(0, 1), keepdims=True)
        im = (im - mean) * contrast + mean
        im *= brightness
    if recipe >= 2 and overhead and rng.random() < 0.5:
        # At most 2.5% per side; outer context remains in nearly-full framing.
        # Fixed inspected workspace bounds are preserved. No wrist crop.
        ratio = rng.uniform(0.95, 1.0)
        nw = round(w * ratio)
        nh = round(h * ratio)
        left = int(rng.integers(0, w - nw + 1))
        top = int(rng.integers(0, h - nh + 1))
        # Preserve task ROI including both plates and full target region.
        roi = (80, 35, 635, 355)
        valid = (
            left <= roi[0]
            and top <= roi[1]
            and left + nw >= roi[2]
            and top + nh >= roi[3]
        )
        if valid and (
            point is None
            or (left <= point[0] < left + nw and top <= point[1] < top + nh)
        ):
            im = cv2.resize(im[top : top + nh, left : left + nw], (w, h))
            if point is not None:
                point = (point - [left, top]) * [w / nw, h / nh]
    if recipe >= 3 and rng.random() < 0.2:
        if rng.random() < 0.5:
            im += rng.normal(0, rng.uniform(0, 0.01), im.shape).astype(np.float32)
        else:
            im = cv2.GaussianBlur(im, (3, 3), float(rng.uniform(0.1, 0.5)))
    im = np.ascontiguousarray(np.rint(np.clip(im, 0, 1) * 255).astype(np.uint8))
    # Digital marker is drawn last, as at runtime; augmentation never moves it independently.
    if point is not None:
        x, y = np.rint(point).astype(int)
        for thickness, color in [(4, (0, 0, 0)), (2, (255, 0, 255))]:
            cv2.line(im, (x - 7, y - 7), (x + 7, y + 7), color, thickness, cv2.LINE_AA)
            cv2.line(im, (x - 7, y + 7), (x + 7, y - 7), color, thickness, cv2.LINE_AA)
    return im
