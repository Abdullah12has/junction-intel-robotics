#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 10: contact sheets of still-window overlays for visual checks (read-only on recordings).

Picks evenly spaced windows from out/detect_<cam>.json by jaw count and tiles their overlays 3 x 2 at 0.6 scale.
Writes overlays/sheet_<cam>_<group>_<n>.jpg and out/sheet_index.json (window index -> sheet cell).
Run: envs/data-env/bin/python -I tools/s10_sheets.py
"""
import json

import cv2
import numpy as np

E1 = "/home/intel/intel-junction-2026/work/P3-TWIN-CAMS/E1"
SCALE = 0.6


def pick(rows, n_jaws, k):
    idx = [r["win"] for r in rows if len(r.get("jaws", [])) == n_jaws]
    if not idx:
        return []
    sel = np.linspace(0, len(idx) - 1, min(k, len(idx))).round().astype(int)
    return [idx[i] for i in sel]


def tile(paths, labels):
    ims = []
    for p, lab in zip(paths, labels):
        im = cv2.imread(p)
        im = cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        cv2.putText(im, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(im, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        ims.append(im)
    while len(ims) < 6:
        ims.append(np.zeros_like(ims[0]))
    rows = [np.hstack(ims[0:3]), np.hstack(ims[3:6])]
    return np.vstack(rows)


def main():
    index = {}
    for cam in ("top", "close"):
        with open(f"{E1}/out/detect_{cam}.json") as fh:
            rows = json.load(fh)
        groups = [("two", 2, 12), ("one", 1, 6), ("zero", 0, 6)]
        for gname, n, k in groups:
            wins = pick(rows, n, k)
            for s in range(0, len(wins), 6):
                chunk = wins[s:s + 6]
                paths = [f"{E1}/overlays/still_{cam}_{w:03d}.jpg" for w in chunk]
                labels = [f"{cam} win {w} {len(next(r for r in rows if r['win'] == w).get('jaws', []))} jaws" for w in chunk]
                sheet = tile(paths, labels)
                name = f"sheet_{cam}_{gname}_{s // 6}.jpg"
                cv2.imwrite(f"{E1}/overlays/{name}", sheet)
                for w in chunk:
                    index[f"{cam}_{w}"] = name
                print(name, chunk)
    with open(f"{E1}/out/sheet_index.json", "w") as fh:
        json.dump(index, fh, indent=1)


if __name__ == "__main__":
    main()
