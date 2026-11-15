"""V1 live-frame check: prints every brick of the two 153426 stills (analysis only). usage: python3 -I -B live2_v1.py APP_DIR"""
import sys

import cv2

sys.path.insert(0, sys.argv[1])
import inspector as I  # noqa: E402

top = cv2.imread("/home/intel/intel-junction-2026/logs/detlive/top_153426.jpg")
close = cv2.imread("/home/intel/intel-junction-2026/logs/detlive/close_153426.jpg")
res = I.scan_full(top, close)
print("hands", len(res["hands"]), "plates_top", res["plates_rejected"], "partial_close", res["partial_at_border"],
      "rejected_close", res["rejected"])
for b in res["bricks"]:
    tape = b.get("tape_score")
    print(b["id"], b["colour"], b["hsv_med"], "area", b["area"], "bbox", b["bbox"], "aspect", b["aspect"],
          "shape", b["shape"], round(b["shape_conf"], 2), "merged", b["merged"], "split", b["split"],
          "partial", b.get("partial"), "tape", tape, b["verdict"], b["reasons"])
