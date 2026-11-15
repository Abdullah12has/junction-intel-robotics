"""Opus check of P2-B-TESTS finding F-1: draw what pick_any.blobs() returns on servo_180002_red.png."""
import sys

import cv2

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
import grasp_auto as ga  # noqa: E402
import pick_any as pa  # noqa: E402

T = "/home/intel/intel-junction-2026/app/tools/tests/fixtures/servo/"
OUT = "/home/intel/intel-junction-2026/work/opus-verify/"
print("jaw mask", None if ga.JAW_MASK is None else (ga.JAW_MASK.shape, int(ga.JAW_MASK.sum())))
for name, logged in (("servo_180002_red.png", (422.7, 268.8)),):
    img = cv2.imread(T + name)
    print(name, img.shape)
    vis = img.copy()
    for min_area in (1500, 700):
        bl = pa.blobs(img, min_area=min_area)
        print("min_area", min_area, "n", len(bl))
        for b in bl:
            print("  ", b["colour"], b["centre"], "short", b["short"], "area", b["area"], "hue", b["hue"],
                  "sat", b["sat"], "val", b["val"], "border", b["border"])
    for b in pa.blobs(img, min_area=700):
        x, y = map(int, b["centre"])
        cv2.circle(vis, (x, y), 6, (0, 255, 255), 2)
        cv2.putText(vis, f"{b['colour']} {b['short']:.0f}", (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if ga.JAW_MASK is not None and ga.JAW_MASK.shape == img.shape[:2]:
        vis[ga.JAW_MASK] = (vis[ga.JAW_MASK] * 0.5).astype(vis.dtype)
    cv2.drawMarker(vis, tuple(map(int, logged)), (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
    cv2.imwrite(OUT + "f1_" + name.replace(".png", ".jpg"), vis)
    print("written", OUT + "f1_" + name.replace(".png", ".jpg"))
