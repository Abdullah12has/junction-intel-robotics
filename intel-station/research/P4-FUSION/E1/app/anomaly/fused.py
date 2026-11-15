"""Anomaly evidence for the close-view tape rule (P4-FUSION E1, stage fusion-v1).

Off unless app/anomaly/fused_config.json sets "enabled": true. score() runs the Anomalib model exported to OpenVINO IR
(device from the config) on the close-view crop of the brick under the jaws, with the crop and preprocessing of the LOEO
evaluation, and returns the evidence that sort_loop.decide() and episode.json record. fused_verdict() applies the
configured rule (AND, OR or LR) to the close tape verdict and that evidence. Metrics and the adoption check: CHANGES.md.
"""
import json
import importlib.util
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "fused_config.json")
_cache = {}
_scorers = {}


def config():
    """The configuration file, read once."""
    if "cfg" not in _cache:
        with open(CONFIG, encoding="utf-8") as fh:
            _cache["cfg"] = json.load(fh)
    return _cache["cfg"]


def crop_of(img, bbox, pad=12):
    """The crop that sort_loop.save_crop writes for the same bbox and pad, as an array."""
    if img is None or bbox is None:
        return None
    x, y, w, h = bbox
    H, W = img.shape[:2]
    return img[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]


class _Scorer:
    """OpenVINO IR scorer, as in the LOEO evaluation: anom3 preprocessing, the anomaly map maximum (or pred_score),
    and padding of a single crop to the static batch of the export."""

    def __init__(self, cfg):
        import openvino as ov
        spec = importlib.util.spec_from_file_location("p4_fusion_anom3_infer", os.path.join(HERE, "anom3", "infer.py"))
        infer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(infer)
        self.infer = infer
        self.size = int(cfg["size"])
        self.device = cfg["device"]
        self.ir_dir = cfg["ir_dir"]
        with open(os.path.join(self.ir_dir, "threshold.json"), encoding="utf-8") as fh:
            self.threshold = float(json.load(fh)["threshold"])
        props = {"INFERENCE_PRECISION_HINT": "f32"} if self.device == "GPU" else {}
        self.compiled = ov.Core().compile_model(os.path.join(self.ir_dir, "model.xml"), self.device, props)
        self.req = self.compiled.create_infer_request()
        shape = self.compiled.input(0).partial_shape
        self.batch = shape[0].get_length() if shape[0].is_static else 1
        self.port, self.is_map = None, False
        for want, is_map in (("pred_score", False), ("anomaly_map", True)):
            for p in self.compiled.outputs:
                try:
                    names = p.get_names()
                except RuntimeError:
                    names = set()
                if any(want in n for n in names):
                    self.port, self.is_map = p, is_map
                    break
            if self.port is not None:
                break
        if self.port is None:
            raise RuntimeError("IR has no pred_score or anomaly_map output: " + self.ir_dir)

    def score(self, bgr):
        xs = [self.infer.preprocess(bgr, self.size)]
        xs += [xs[-1]] * (self.batch - 1)
        x = np.stack(xs).astype(np.float32)
        res = np.asarray(self.req.infer({0: x})[self.port], dtype=np.float32).reshape(self.batch, -1)
        vals = res.max(axis=1) if self.is_map else res[:, 0]
        return float(vals[0])


def _lr_params(cfg):
    with open(cfg["lr_params"], encoding="utf-8") as fh:
        return json.load(fh)["models"][cfg["model"]]


def _lr_probability(p, tape_score, anomaly_score):
    x = np.array([tape_score, anomaly_score], float)
    z = (x - np.asarray(p["mu"])) / np.asarray(p["sd"])
    return 1.0 / (1.0 + np.exp(-(z @ np.asarray(p["w"]) + p["b"])))


def score(crop, log=None):
    """Anomaly evidence for one close-view crop (BGR array), or None when the flag is off, there is no crop, or the
    evidence cannot be computed (logged; the tape verdict then stands)."""
    cfg = config()
    if not cfg.get("enabled") or crop is None:
        return None
    try:
        key = (cfg["ir_dir"], cfg["device"])
        if key not in _scorers:
            _scorers[key] = _Scorer(cfg)
        sc = _scorers[key]
        t0 = time.perf_counter()
        s = sc.score(crop)
        ms = (time.perf_counter() - t0) * 1000.0
        lr = _lr_params(cfg) if cfg["rule"] == "LR" else None
    except Exception as exc:  # optional evidence must not stop the loop: the tape verdict stands, as before
        if log:
            log("anomaly unavailable, tape verdict stands: %s: %s" % (type(exc).__name__, exc))
        return None
    if log:
        log("anomaly %s score %.3f threshold %.3f device %s latency %.1f ms"
            % (cfg["model"], s, sc.threshold, sc.device, ms))
    return {"model": cfg["model"], "device": sc.device, "score": s, "threshold": sc.threshold,
            "latency_ms": round(ms, 2), "rule": cfg["rule"], "lr": lr}


def fused_verdict(pre, anomaly):
    """CONFIDENT_DEFECT or CONFIDENT_GOOD: the configured rule on the close tape verdict and the anomaly evidence."""
    tape = pre.get("tape_verdict", pre.get("verdict")) == "CONFIDENT_DEFECT"
    over = anomaly["score"] > anomaly["threshold"]
    rule = anomaly["rule"]
    if rule == "AND":
        defect = tape and over
    elif rule == "OR":
        defect = tape or over
    elif rule == "LR":
        defect = _lr_probability(anomaly["lr"], float(pre.get("tape_score") or 0.0), anomaly["score"]) >= 0.5
    else:
        raise ValueError("unknown fusion rule %r" % (rule,))
    return "CONFIDENT_DEFECT" if defect else "CONFIDENT_GOOD"


def effective_verdict(pre, anomaly):
    """The close verdict the loop acts on: the fused verdict when there is anomaly evidence, else the tape verdict."""
    if anomaly is None:
        return pre.get("verdict")
    return fused_verdict(pre, anomaly)
