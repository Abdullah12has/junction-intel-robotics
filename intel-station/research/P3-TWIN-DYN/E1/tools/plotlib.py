#!/usr/bin/env python3
"""P3-TWIN-DYN E1: minimal PIL plotting (matplotlib is not installed on the laptop; PIL is)."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {"k": (20, 20, 20), "r": (200, 30, 30), "b": (30, 90, 200), "g": (30, 150, 60),
          "o": (220, 130, 0), "p": (130, 40, 170), "gray": (150, 150, 150), "lgray": (226, 226, 226)}
FONT = ImageFont.load_default()


def nice_ticks(lo, hi, n=5):
    lo, hi = float(lo), float(hi)
    if not hi > lo:
        lo, hi = lo - 0.5, hi + 0.5
    raw = (hi - lo) / n
    mag = 10.0 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    ticks = []
    v = math.ceil(lo / step - 1e-9) * step
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return lo, hi, ticks


class Panel:
    def __init__(self, draw, box, xlim, ylim, title="", xlabel="", ylabel=""):
        self.d = draw
        self.x0, self.y0, self.x1, self.y1 = box
        xlo, xhi, xt = nice_ticks(*xlim)
        ylo, yhi, yt = nice_ticks(*ylim)
        self.xlim, self.ylim = (xlo, xhi), (ylo, yhi)
        for v in xt:
            px = self.X(v)
            self.d.line([(px, self.y0), (px, self.y1)], fill=COLORS["lgray"])
            self.d.text((px - 8, self.y1 + 3), f"{v:g}", fill=COLORS["k"], font=FONT)
        for v in yt:
            py = self.Y(v)
            self.d.line([(self.x0, py), (self.x1, py)], fill=COLORS["lgray"])
            self.d.text((self.x0 - 40, py - 5), f"{v:g}", fill=COLORS["k"], font=FONT)
        self.d.rectangle(box, outline=COLORS["k"], width=1)
        if title:
            self.d.text((self.x0 + 4, self.y0 - 13), title, fill=COLORS["k"], font=FONT)
        if xlabel:
            self.d.text((self.x0 + (self.x1 - self.x0) // 2 - 60, self.y1 + 16), xlabel, fill=COLORS["k"], font=FONT)
        if ylabel:
            self.d.text((self.x1 + 4, self.y0), ylabel, fill=COLORS["k"], font=FONT)

    def X(self, x):
        lo, hi = self.xlim
        return self.x0 + (np.asarray(x, dtype=float) - lo) / (hi - lo) * (self.x1 - self.x0)

    def Y(self, y):
        lo, hi = self.ylim
        return self.y1 - (np.asarray(y, dtype=float) - lo) / (hi - lo) * (self.y1 - self.y0)

    def line(self, xs, ys, color="k", width=1):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() < 2:
            return
        px = self.X(xs[ok])
        py = self.Y(ys[ok])
        pts = [(float(a), float(b)) for a, b in zip(px, py)]
        self.d.line(pts, fill=COLORS[color], width=width)

    def legend(self, entries, x, y):
        for k, (label, color, dashed) in enumerate(entries):
            yy = y + 13 * k
            if dashed:
                for s in range(0, 14, 4):
                    self.d.line([(x + s, yy), (x + s + 2, yy)], fill=COLORS[color], width=2)
            else:
                self.d.line([(x, yy), (x + 14, yy)], fill=COLORS[color], width=2)
            self.d.text((x + 18, yy - 5), label, fill=COLORS["k"], font=FONT)


def new_canvas(width, height):
    img = Image.new("RGB", (width, height), (255, 255, 255))
    return img, ImageDraw.Draw(img)


def bar_chart(draw, box, labels, values, gate, title, unit):
    x0, y0, x1, y1 = box
    top = max(max(values), gate) * 1.25
    draw.rectangle(box, outline=COLORS["k"], width=1)
    draw.text((x0 + 4, y0 - 13), title, fill=COLORS["k"], font=FONT)
    gy = y1 - gate / top * (y1 - y0)
    draw.line([(x0, gy), (x1, gy)], fill=COLORS["r"], width=1)
    draw.text((x1 - 150, gy - 12), f"gate {gate:g} {unit}", fill=COLORS["r"], font=FONT)
    slot = (x1 - x0) / len(values)
    for k, (lab, v) in enumerate(zip(labels, values)):
        bx0 = x0 + k * slot + slot * 0.2
        bx1 = x0 + (k + 1) * slot - slot * 0.2
        by = y1 - v / top * (y1 - y0)
        color = COLORS["g"] if v <= gate else COLORS["p"]
        draw.rectangle([bx0, by, bx1, y1], fill=color)
        draw.text((bx0, by - 12), f"{v:.3f}", fill=COLORS["k"], font=FONT)
        draw.text((bx0, y1 + 4), lab, fill=COLORS["k"], font=FONT)
