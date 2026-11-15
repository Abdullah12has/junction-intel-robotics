import sys
import time

import cv2

for dev in sys.argv[1:]:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ok, img = False, None
    t0 = time.monotonic()
    n = 0
    while time.monotonic() - t0 < 3.0:
        ok2, f = cap.read()
        if ok2:
            ok, img, n = True, f, n + 1
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cc = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    print(dev, "opened", cap.isOpened(), "ok", ok, "frames_in_3s", n, "fourcc", cc,
          "size", None if img is None else img.shape, "fps_prop", cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if ok:
        name = dev.replace("/dev/", "")
        cv2.imwrite(f"/home/intel/intel-junction-2026/work/marker/probe_{name}.jpg", img)
