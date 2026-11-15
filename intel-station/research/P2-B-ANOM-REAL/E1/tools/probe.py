import sys
import cv2
import numpy as np

seg, out_prefix, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
cap = cv2.VideoCapture(seg, cv2.CAP_FFMPEG)
sel, i = [], 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    if i % step == 0:
        sel.append((i, f))
    i += 1
print("frames_total", i, "selected", len(sel))
cols, tw, th = 5, 320, 240
rows = (len(sel) + cols - 1) // cols
canvas = np.zeros((rows * th, cols * tw, 3), np.uint8)
mcanvas = np.zeros_like(canvas)
for k, (idx, f) in enumerate(sel):
    r, c = divmod(k, cols)
    t = cv2.resize(f, (tw, th))
    cv2.putText(t, f"{idx/30:.1f}s", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    canvas[r*th:(r+1)*th, c*tw:(c+1)*tw] = t
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = (((h < 10) | (h > 170)) & (s > 110) & (v > 60)).astype(np.uint8) * 255
    m = cv2.cvtColor(cv2.resize(red, (tw, th)), cv2.COLOR_GRAY2BGR)
    cv2.putText(m, f"red {int(red.astype(bool).sum())}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    mcanvas[r*th:(r+1)*th, c*tw:(c+1)*tw] = m
cv2.imwrite(out_prefix + "_rgb.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
cv2.imwrite(out_prefix + "_red.jpg", mcanvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
print("written", out_prefix)
