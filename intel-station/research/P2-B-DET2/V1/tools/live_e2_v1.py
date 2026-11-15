"""Before-picture: E2 detector on the two live 153426 stills (analysis only). usage: python3 -I -B live_e2_v1.py E2_APP_DIR"""
import sys

import cv2

sys.path.insert(0, sys.argv[1])
import inspector as I  # noqa: E402

top = cv2.imread("/home/intel/intel-junction-2026/logs/detlive/top_153426.jpg")
close = cv2.imread("/home/intel/intel-junction-2026/logs/detlive/close_153426.jpg")
print("E2 has scan_full:", hasattr(I, "scan_full"), "| has triage:", hasattr(I, "triage"))
if hasattr(I, "scan_full"):
    res = I.scan_full(top, close)
    print("hands", len(res.get("hands", [])), "rejected", res.get("rejected"), "partial", res.get("partial_at_border"))
    bricks = res["bricks"]
else:
    bricks = I.scan(top, close)
for b in bricks:
    print(b.get("view"), b.get("colour"), b.get("area"), b.get("bbox"), b.get("shape"), b.get("tape_score"),
          b.get("verdict", "-"), b.get("reasons", "-"))
print("E2 brick count", len(bricks))
