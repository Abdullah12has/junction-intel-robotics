"""anom3 scorer: one OpenVINO PatchCore IR per view, CPU only (no GPU jobs on this laptop).

score(view, bgr)         -> {"view", "score", "threshold", "label", "latency_ms", "device", "version"}
score_batch(view, list)  -> list of the same dicts (one request per chunk of the IR batch size)
label = "DEFECT" when score > threshold, else "GOOD". Crops are BGR (cv2), any size.
version "current" = models/current, a symlink that only Opus moves.

CLI: python3 -I -B infer.py --view pre_close --image CROP.png [--image MORE.png] [--version current]
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(HERE, "models")
SIZE = 256
DEVICES = ("CPU",)  # CPU only: no GPU jobs on this laptop (coordinator update, 18:32Z)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(bgr, size=SIZE):
    """BGR uint8 crop -> float32 CHW tensor: resize to size x size, RGB, [0, 1], ImageNet normalisation."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


def read_threshold(view_dir):
    path = os.path.join(view_dir, "threshold.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return float(json.load(fh)["threshold"])


class ViewModel:
    """One compiled IR (first device in `devices` that compiles). threshold is None until threshold.json exists."""

    def __init__(self, core, view_dir, devices=DEVICES):
        bad = [d for d in devices if d != "CPU"]
        if bad:
            raise ValueError("anom3 runs on CPU only; refused devices %s" % bad)
        self.threshold = read_threshold(view_dir)
        self._core, self._xml = core, os.path.join(view_dir, "model.xml")
        self._cpu_req, self.nan_fallbacks = None, 0
        errors = []
        self.device = None
        for dev in devices:
            try:
                # GPU runs FP32 graphs in FP16 by default and the squared patch distances overflow (NaN);
                # FP32 inference precision matches CPU (checked 2026-11-14, scratch/diag_gpu.py)
                cfg = {"INFERENCE_PRECISION_HINT": "f32"} if dev == "GPU" else {}
                self.compiled = core.compile_model(self._xml, dev, cfg)
                self.device = dev
                break
            except Exception as exc:  # device missing or busy: try the next one
                errors.append("%s: %s" % (dev, str(exc)[:200]))
        if self.device is None:
            raise RuntimeError("compile failed: " + "; ".join(errors))
        pshape = self.compiled.input(0).partial_shape
        dims = [d.get_length() if d.is_static else -1 for d in pshape]
        self.batch = dims[0] if dims[0] > 0 else 1
        self.size = dims[2] if len(dims) == 4 and dims[2] > 0 else SIZE
        self.port, self.is_map = self._pick_output()
        self.req = self.compiled.create_infer_request()

    def _pick_output(self):
        for port in self.compiled.outputs:
            try:
                names = port.get_names()
            except RuntimeError:
                names = set()
            if any("pred_score" in n for n in names):
                return port, False
        for port in self.compiled.outputs:
            try:
                names = port.get_names()
            except RuntimeError:
                names = set()
            if any("anomaly_map" in n for n in names):
                return port, True
        raise RuntimeError("IR has neither a pred_score nor an anomaly_map output")

    def _run(self, x):
        arr = np.asarray(self.req.infer({0: x})[self.port], dtype=np.float32)
        if self.device != "CPU" and not np.isfinite(arr).all():
            # non-finite output on this device: recompute the chunk on CPU and count it
            self.nan_fallbacks += 1
            if self._cpu_req is None:
                cpu = self._core.compile_model(self._xml, "CPU")
                self._cpu_req = (cpu.create_infer_request(), self.port.get_any_name())
                self._cpu_port = cpu.output(self._cpu_req[1])
            arr = np.asarray(self._cpu_req[0].infer({0: x})[self._cpu_port], dtype=np.float32)
        return arr

    def scores(self, bgr_list):
        n = self.batch
        out = []
        for start in range(0, len(bgr_list), n):
            chunk = [preprocess(c, self.size) for c in bgr_list[start:start + n]]
            real = len(chunk)
            chunk += [chunk[-1]] * (n - real)
            res = self._run(np.stack(chunk).astype(np.float32)).reshape(n, -1)
            vals = res.max(axis=1) if self.is_map else res[:, 0]
            out.extend(float(v) for v in vals[:real])
        return out


class Scorer:
    def __init__(self, version="current", models_root=MODELS_ROOT, devices=DEVICES):
        self.root = os.path.join(models_root, version)
        if not os.path.isdir(self.root):
            raise FileNotFoundError("no model folder " + self.root)
        self.version = os.path.basename(os.path.realpath(self.root))
        self.devices = devices
        import openvino as ov
        self._core = ov.Core()
        self._views = {}

    def _view(self, view):
        if view not in self._views:
            vdir = os.path.join(self.root, view)
            if not os.path.isfile(os.path.join(vdir, "model.xml")):
                raise KeyError("no model for view %r in %s" % (view, self.root))
            vm = ViewModel(self._core, vdir, self.devices)
            if vm.threshold is None:
                raise RuntimeError("threshold.json missing in " + vdir)
            self._views[view] = vm
        return self._views[view]

    def score_batch(self, view, bgr_list):
        vm = self._view(view)
        t0 = time.perf_counter()
        vals = vm.scores(list(bgr_list))
        ms = round((time.perf_counter() - t0) * 1000.0 / max(1, len(vals)), 2)
        return [{"view": view, "score": s, "threshold": vm.threshold,
                 "label": "DEFECT" if s > vm.threshold else "GOOD",
                 "latency_ms": ms, "device": vm.device, "version": self.version} for s in vals]

    def score(self, view, bgr):
        return self.score_batch(view, [bgr])[0]


_SCORERS = {}


def _scorer(version, models_root):
    key = os.path.realpath(os.path.join(models_root, version))  # a moved current symlink gives a new key
    if key not in _SCORERS:
        _SCORERS[key] = Scorer(version, models_root)
    return _SCORERS[key]


def score(view, bgr, version="current", models_root=MODELS_ROOT):
    return _scorer(version, models_root).score(view, bgr)


def score_batch(view, bgr_list, version="current", models_root=MODELS_ROOT):
    return _scorer(version, models_root).score_batch(view, bgr_list)


def score_files(view_dir, paths, devices=DEVICES):
    """Raw scores for image files of one view folder (None where a file cannot be read). Returns (scores, device)."""
    import openvino as ov
    vm = ViewModel(ov.Core(), view_dir, devices)
    imgs, idx = [], []
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)
            idx.append(i)
    out = [None] * len(paths)
    for i, s in zip(idx, vm.scores(imgs)):
        out[i] = s
    dev = vm.device + (" nan_fallbacks=%d" % vm.nan_fallbacks if vm.nan_fallbacks else "")
    return out, dev


def main():
    ap = argparse.ArgumentParser(description="Score BGR crops with an anom3 view model.")
    ap.add_argument("--view", required=True)
    ap.add_argument("--image", required=True, action="append")
    ap.add_argument("--version", default="current")
    ap.add_argument("--models-root", default=MODELS_ROOT)
    args = ap.parse_args()
    imgs = []
    for p in args.image:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit("cannot read " + p)
        imgs.append(img)
    print(json.dumps(score_batch(args.view, imgs, args.version, args.models_root), indent=1))


if __name__ == "__main__":
    main()
