"""Top-camera marker for Physical AI Studio recordings.

Owns the top RealSense colour node, paints one fixed X over a chosen cube and serves the painted view as MJPEG, which
Studio reads as an IP camera. The X stays at the chosen pixel until the next target is set, so a recorded episode sees
it at the cube's initial location the whole time.

  envs/data-env/bin/python -u -B app/tools/marker.py [--serial 344223023450 | --device /dev/video8] [--port 8090]

  GET  /                 control page (debug view, new random cube, clear, click to set)
  GET  /top.mjpg         painted stream (Studio IP camera URL); /top.jpg one frame
  GET  /raw.mjpg         unpainted stream; /raw.jpg one frame
  GET  /debug.mjpg       painted stream plus workspace, keep-out and detected cubes (operator only, never recorded)
  GET  /target           current target and the last detection, JSON
  POST /target/random    pick a detected cube at random
  POST /target?u=&v=     set the target pixel (controller at run time, or a click)
  POST /target/clear     remove the X
"""
import argparse
import glob
import json
import os
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

import paths
import topdet

DEFAULT_CONFIG = {
    "workspace": [[[135, 25], [640, 25], [640, 370], [130, 350]]],
    "keepout": [[[318, 0], [470, 0], [470, 200], [318, 200]]],
    "cross": {"half": 26, "width": 7, "outline": 2, "bgr": [255, 0, 255]},
    "jpeg_quality": 90,
}


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def log(msg):
    print(f"{utc()} {msg}", flush=True)


def atomic_write(path, data):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def device_for_serial(serial):
    """RGB colour node of the RealSense with this USB serial (interface 1.3, index 0), found through sysfs."""
    for d in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            iface = os.path.realpath(os.path.join(d, "device"))
            with open(os.path.join(iface, "..", "serial")) as f:
                s = f.read().strip()
            with open(os.path.join(d, "index")) as f:
                idx = f.read().strip()
        except OSError:
            continue
        if s == serial and iface.endswith(":1.3") and idx == "0":
            return "/dev/" + os.path.basename(d)
    raise SystemExit(f"no RealSense colour node for USB serial {serial}")


class Marker:
    def __init__(self, args):
        self.args = args
        cfg = dict(DEFAULT_CONFIG)
        if os.path.exists(args.config):
            with open(args.config) as f:
                cfg.update(json.load(f))
        self.cfg = cfg
        self.cond = threading.Condition()
        self.raw, self.seq, self.t_frame = None, 0, 0.0
        self.jpg = {}
        self.target = self._load_target()
        self.last_detection = []
        self.frames, self.fps = 0, 0.0
        os.makedirs(args.log_dir, exist_ok=True)

    def _load_target(self):
        try:
            with open(self.args.state) as f:
                t = json.load(f)
            log(f"target restored from {self.args.state}: {t}")
            return t
        except (OSError, ValueError):
            return None

    def set_target(self, t, why, candidates=None, snapshot=None):
        with self.cond:
            self.target = t
            self.jpg.clear()
        atomic_write(self.args.state, json.dumps(t).encode())
        rec = {"utc": utc(), "why": why, "target": t, "candidates": candidates}
        if snapshot is not None and t is not None:
            p = os.path.join(self.args.log_dir, f"snap_{t['id']}.jpg")
            cv2.imwrite(p, snapshot)
            rec["snapshot"] = p
        with open(os.path.join(self.args.log_dir, "targets.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"target {why}: {t}")

    def random_target(self):
        img = self.latest_raw()
        if img is None:
            return None, "no frame yet"
        det = topdet.detect(img, self.cfg["workspace"], self.cfg["keepout"])
        self.last_detection = det
        ok = [d for d in det if d["ok"]]
        if not ok:
            return None, "no cube detected"
        d = random.choice(ok)
        t = {"id": utc().replace(":", "").replace("-", "") + "_" + uuid.uuid4().hex[:4], "u": d["u"], "v": d["v"],
             "colour": d["colour"], "set_utc": utc(), "mode": "random", "n_candidates": len(ok)}
        self.set_target(t, "random", det, img)
        return t, None

    def manual_target(self, u, v):
        t = {"id": utc().replace(":", "").replace("-", "") + "_" + uuid.uuid4().hex[:4], "u": round(u, 1),
             "v": round(v, 1), "colour": None, "set_utc": utc(), "mode": "manual"}
        self.set_target(t, "manual", None, self.latest_raw())
        return t

    def latest_raw(self):
        with self.cond:
            return None if self.raw is None else self.raw.copy()

    def paint(self, img):
        t = self.target
        if t is None:
            return img
        c = self.cfg["cross"]
        u, v, h = int(round(t["u"])), int(round(t["v"])), int(c["half"])
        out = img.copy()
        for (x0, y0, x1, y1) in ((u - h, v - h, u + h, v + h), (u - h, v + h, u + h, v - h)):
            cv2.line(out, (x0, y0), (x1, y1), (0, 0, 0), c["width"] + 2 * c["outline"], cv2.LINE_AA)
        for (x0, y0, x1, y1) in ((u - h, v - h, u + h, v + h), (u - h, v + h, u + h, v - h)):
            cv2.line(out, (x0, y0), (x1, y1), tuple(int(x) for x in c["bgr"]), c["width"], cv2.LINE_AA)
        return out

    def debug(self, img):
        out = self.paint(img)
        for poly in self.cfg["workspace"]:
            cv2.polylines(out, [np.asarray(poly, np.int32)], True, (255, 255, 0), 1)
        for poly in self.cfg["keepout"]:
            cv2.polylines(out, [np.asarray(poly, np.int32)], True, (0, 165, 255), 1)
        for d in self.last_detection:
            col = (0, 255, 0) if d["ok"] else (0, 0, 255)
            cv2.circle(out, (int(d["u"]), int(d["v"])), 4, col, -1)
            if not d["ok"]:
                cv2.putText(out, d["reject"], (int(d["u"]) + 6, int(d["v"]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            col, 1, cv2.LINE_AA)
        cv2.putText(out, f"{self.fps:4.1f} fps", (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return out

    def jpeg(self, kind, wait_after=None, timeout=2.0):
        with self.cond:
            if wait_after is not None:
                self.cond.wait_for(lambda: self.seq > wait_after, timeout)
            if self.raw is None:
                return None, self.seq
            if kind not in self.jpg:
                img = {"top": self.paint, "raw": lambda x: x, "debug": self.debug}[kind](self.raw)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg["jpeg_quality"])])
                self.jpg[kind] = buf.tobytes()
            return self.jpg[kind], self.seq

    def capture(self):
        dev = self.args.device or device_for_serial(self.args.serial)
        still = os.path.isfile(dev)
        last_live, t0, n0 = 0.0, time.monotonic(), 0
        while True:
            cap = None if still else cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if cap is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
                cap.set(cv2.CAP_PROP_FPS, self.args.fps)
                if not cap.isOpened():
                    log(f"cannot open {dev}; retrying in 2 s")
                    time.sleep(2)
                    continue
            log(f"capturing {dev}")
            fails = 0
            while True:
                if still:
                    time.sleep(1.0 / self.args.fps)
                    ok, img = True, cv2.imread(dev)
                else:
                    ok, img = cap.read()
                if not ok or img is None:
                    fails += 1
                    if fails >= 30:
                        log(f"{dev}: 30 failed reads; reopening")
                        break
                    continue
                fails = 0
                if img.shape[1] != self.args.width or img.shape[0] != self.args.height:
                    img = cv2.resize(img, (self.args.width, self.args.height), interpolation=cv2.INTER_AREA)
                now = time.monotonic()
                with self.cond:
                    self.raw, self.t_frame = img, now
                    self.seq += 1
                    self.jpg.clear()
                    self.cond.notify_all()
                self.frames += 1
                if now - t0 >= 2.0:
                    self.fps, t0, n0 = (self.frames - n0) / (now - t0), now, self.frames
                if self.args.live_jpg and now - last_live >= 0.125:
                    last_live = now
                    ok2, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok2:
                        atomic_write(self.args.live_jpg, buf.tobytes())
            if cap is not None:
                cap.release()
            time.sleep(1)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Top marker</title><style>
body{font-family:sans-serif;margin:16px;background:#111;color:#eee}img{width:640px;height:480px;cursor:crosshair;
display:block;border:1px solid #444}button{font-size:16px;margin:8px 8px 8px 0;padding:8px 14px}pre{font-size:12px}
</style></head><body><h3>Top camera marker</h3><img id="v" src="/debug.mjpg"><div>
<button onclick="post('/target/random')">New random cube</button><button onclick="post('/target/clear')">Clear</button>
<span>Click a cube to set it by hand.</span></div><pre id="s"></pre><script>
async function post(p){const r=await fetch(p,{method:'POST'});show(await r.json());}
async function show(j){document.getElementById('s').textContent=JSON.stringify(j||await (await fetch('/target')).json(),null,1);}
document.getElementById('v').onclick=e=>{const b=e.target.getBoundingClientRect();
post('/target?u='+Math.round((e.clientX-b.left)*640/b.width)+'&v='+Math.round((e.clientY-b.top)*480/b.height));};
show();setInterval(()=>show(),3000);
</script></body></html>"""


def make_handler(m):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self, kind):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            seq = -1
            try:
                while True:
                    jpg, seq2 = m.jpeg(kind, wait_after=seq)
                    if jpg is None or seq2 == seq:
                        continue
                    seq = seq2
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path in ("/top.mjpg", "/raw.mjpg", "/debug.mjpg"):
                self._stream(path[1:].split(".")[0])
            elif path in ("/top.jpg", "/raw.jpg", "/debug.jpg"):
                jpg, _ = m.jpeg(path[1:].split(".")[0])
                if jpg is None:
                    return self._json(503, {"error": "no frame yet"})
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
            elif path == "/target":
                self._json(200, {"target": m.target, "detection": m.last_detection, "fps": round(m.fps, 1),
                                 "frame_age_s": round(time.monotonic() - m.t_frame, 3)})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            u = urlsplit(self.path)
            q = parse_qs(u.query)
            if u.path == "/target/random":
                t, err = m.random_target()
                return self._json(200 if t else 409, {"target": t, "error": err, "detection": m.last_detection})
            if u.path == "/target/clear":
                m.set_target(None, "clear")
                return self._json(200, {"target": None})
            if u.path == "/target" and "u" in q and "v" in q:
                t = m.manual_target(float(q["u"][0]), float(q["v"][0]))
                return self._json(200, {"target": t})
            self._json(404, {"error": "not found"})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="344223023450")
    ap.add_argument("--device", default=None)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--config", default=os.path.join(paths.APP, "config", "marker_top.json"))
    ap.add_argument("--state", default=os.path.join(paths.RUN_ROOT, "marker_target.json"))
    ap.add_argument("--log-dir", default=os.path.join(paths.LOG_ROOT, "marker"))
    ap.add_argument("--live-jpg", default=os.path.join(paths.RUN_ROOT, "live_top.jpg"))
    args = ap.parse_args()
    m = Marker(args)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(m))
    srv.daemon_threads = True
    atomic_write(os.path.join(paths.RUN_ROOT, "marker.pid"), str(os.getpid()).encode())
    threading.Thread(target=m.capture, name="capture", daemon=True).start()
    log(f"serving http://{args.host}:{args.port}/ (Studio IP camera URL http://{args.host}:{args.port}/top.mjpg)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
