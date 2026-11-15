#!/usr/bin/env python3
# V1 tool: native pixel size of recorded JPEG/PNG frames. Stdlib only; run as python3 -I.
# Output per file: bytes, format, width, height, path (tab separated).
import os
import struct
import sys

JPEG_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_size(f):
    f.seek(2)
    while True:
        b = f.read(1)
        if not b:
            return None
        if b != b"\xff":
            continue
        m = f.read(1)
        while m == b"\xff":
            m = f.read(1)
        if not m:
            return None
        code = m[0]
        if code in (0xD8, 0xD9, 0xDA):
            return None
        if code == 0x01 or 0xD0 <= code <= 0xD7:
            continue
        seglen = struct.unpack(">H", f.read(2))[0]
        if code in JPEG_SOF:
            f.read(1)
            h, w = struct.unpack(">HH", f.read(4))
            return w, h
        f.seek(seglen - 2, 1)


def size(path):
    with open(path, "rb") as f:
        head = f.read(24)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return "png", w, h
        if head[:2] == b"\xff\xd8":
            wh = jpeg_size(f)
            if wh:
                return "jpg", wh[0], wh[1]
        return "unknown", None, None


for p in sys.argv[1:]:
    fmt, w, h = size(p)
    print("%d\t%s\t%s\t%s\t%s" % (os.path.getsize(p), fmt, w, h, p))
