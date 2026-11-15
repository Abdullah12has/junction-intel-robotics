"""Synthetic Lego-like Folder dataset for the anomalib 2.7.0 + OpenVINO laptop check (P2-B-ANOM-LAPTOP, E1).

Layout matches anomalib Folder(normal_dir="train/good", normal_test_dir="test/good", abnormal_dir="test/defect"):
  <out>/train/good/   normal images (default 40)
  <out>/test/good/    normal images, held out (default 10)
  <out>/test/defect/  one defect per image (default 10; black_mark, missing_stud, chipped_corner)
Images are 256x256 RGB PNG: 2-4 bricks (2x2 or 2x4 studs) on a dark mat, with sensor-like noise.
Seeded. manifest.json lists every file, its defect type and brick count.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = 256
UNIT = 22
MAT = (28, 28, 34)
COLORS = {
    "pink": (230, 80, 160),
    "blue": (40, 110, 220),
    "yellow": (240, 200, 40),
    "orange": (240, 130, 30),
    "white": (235, 235, 230),
}
DEFECTS = ("black_mark", "missing_stud", "chipped_corner")


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def place_bricks(rng):
    bricks = []
    target = int(rng.integers(2, 5))
    tries = 0
    while len(bricks) < target and tries < 1000:
        tries += 1
        cols, rows = (2, 2) if rng.random() < 0.5 else (4, 2)
        if rng.random() < 0.5:
            cols, rows = rows, cols
        w, h = cols * UNIT, rows * UNIT
        x = int(rng.integers(12, SIZE - 12 - w))
        y = int(rng.integers(12, SIZE - 12 - h))
        rect = (x - 6, y - 6, x + w + 6, y + h + 6)
        if any(overlaps(rect, b["rect"]) for b in bricks):
            continue
        color = list(COLORS)[int(rng.integers(len(COLORS)))]
        bricks.append({"rect": rect, "x": x, "y": y, "w": w, "h": h, "cols": cols, "rows": rows, "color": color})
    return bricks


def tint(color, amount):
    return tuple(max(0, min(255, int(c + amount))) for c in color)


def draw_brick(draw, brick, flat_cells=frozenset()):
    base = COLORS[brick["color"]]
    light, dark = tint(base, 40), tint(base, -60)
    x, y, w, h = brick["x"], brick["y"], brick["w"], brick["h"]
    draw.rectangle([x, y, x + w - 1, y + h - 1], fill=base, outline=dark, width=2)
    uw, uh = w / brick["cols"], h / brick["rows"]
    r = min(uw, uh) * 0.26
    for i in range(brick["cols"]):
        for j in range(brick["rows"]):
            cx, cy = x + (i + 0.5) * uw, y + (j + 0.5) * uh
            if (i, j) in flat_cells:
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=base)
            else:
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=light, outline=dark)


def apply_defect(draw, brick, kind, rng):
    x, y, w, h = brick["x"], brick["y"], brick["w"], brick["h"]
    if kind == "black_mark":
        cx = int(rng.integers(x + 10, x + w - 10))
        cy = int(rng.integers(y + 10, y + h - 10))
        rr = int(rng.integers(5, 9))
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(12, 12, 12))
    elif kind == "chipped_corner":
        s = int(rng.integers(14, 21))
        corner = int(rng.integers(4))
        if corner == 0:
            pts = [(x, y), (x + s, y), (x, y + s)]
        elif corner == 1:
            pts = [(x + w, y), (x + w - s, y), (x + w, y + s)]
        elif corner == 2:
            pts = [(x, y + h), (x + s, y + h), (x, y + h - s)]
        else:
            pts = [(x + w, y + h), (x + w - s, y + h), (x + w, y + h - s)]
        draw.polygon(pts, fill=MAT)
    else:
        raise ValueError(kind)


def render(rng, defect):
    bricks = place_bricks(rng)
    img = Image.new("RGB", (SIZE, SIZE), MAT)
    draw = ImageDraw.Draw(img)
    target = int(rng.integers(len(bricks))) if defect else -1
    for idx, brick in enumerate(bricks):
        if idx == target and defect == "missing_stud":
            cells = [(i, j) for i in range(brick["cols"]) for j in range(brick["rows"])]
            picks = rng.choice(len(cells), size=2, replace=False)
            draw_brick(draw, brick, {cells[int(p)] for p in picks})
        else:
            draw_brick(draw, brick)
            if idx == target:
                apply_defect(draw, brick, defect, rng)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + rng.normal(0.0, 3.0, arr.shape).astype(np.float32)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img, len(bricks), target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=40)
    ap.add_argument("--n_test_good", type=int, default=10)
    ap.add_argument("--n_test_defect", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    entries = []
    plan = [("train", "good", args.n_train), ("test", "good", args.n_test_good), ("test", "defect", args.n_test_defect)]
    for split, kind, n in plan:
        folder = out / split / kind
        folder.mkdir(parents=True, exist_ok=True)
        kinds = [DEFECTS[i % len(DEFECTS)] for i in range(n)] if kind == "defect" else [None] * n
        kinds = [kinds[int(k)] for k in rng.permutation(n)]
        for i, defect in enumerate(kinds):
            img, n_bricks, target = render(rng, defect)
            name = f"{kind}_{i:03d}.png"
            img.save(folder / name)
            entries.append({
                "file": f"{split}/{kind}/{name}",
                "split": split,
                "label": "defect" if kind == "defect" else "good",
                "defect_type": defect,
                "n_bricks": n_bricks,
                "defect_brick_index": target if defect else None,
            })
    manifest = {
        "seed": args.seed,
        "size": SIZE,
        "mat_rgb": list(MAT),
        "counts": {"train_good": args.n_train, "test_good": args.n_test_good, "test_defect": args.n_test_defect},
        "defect_types": list(DEFECTS),
        "files": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(entries)} images under {out}")


if __name__ == "__main__":
    main()
