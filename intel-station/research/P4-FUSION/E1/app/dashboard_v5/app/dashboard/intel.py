"""Intel panel sources (P2-B-OPT): the telemetry jsonl of app/telemetry and the bench.json of the benchmark.

Read-only and stdlib only. The telemetry file is read from its tail (one sample per second, so a day is
large); the bench file is cached by (mtime, size). Every number keeps its source: the sample's own src
string for telemetry, the bench file path for benchmark rows. A missing value is None, never a guess.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

from dashboard.policy import _num, _text, parse_t

TAIL_BYTES = 128 * 1024
WINDOW_S = 60.0
STALE_S = 10.0
_BENCH_CACHE: dict[str, tuple[tuple[float, int], dict[str, Any]]] = {}


def read_tail(path: Path, max_bytes: int = TAIL_BYTES) -> list[dict[str, Any]]:
    """Complete JSON object lines from the last max_bytes of the file; a partial first or last line is dropped."""
    try:
        size = os.path.getsize(path)
        start = max(0, size - max_bytes)
        with open(path, "rb") as handle:
            handle.seek(start)
            data = handle.read()
    except OSError:
        return []
    if start > 0:
        newline = data.find(b"\n")
        data = data[newline + 1:] if newline >= 0 else b""
    records: list[dict[str, Any]] = []
    for raw in data.split(b"\n"):
        text = raw.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except (ValueError, RecursionError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _mean(values: list[float | None]) -> float | None:
    xs = [v for v in values if v is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def summarise_telemetry(samples: list[dict[str, Any]], now: float, source_file: str) -> dict[str, Any]:
    """The latest sample plus 60 s means. Metrics the sampler reports as null stay null (power on this laptop)."""
    rows = [(parse_t(s.get("t")), s) for s in samples]
    rows = sorted(((t, s) for t, s in rows if t is not None), key=lambda pair: pair[0])
    if not rows:
        return {"available": False, "file": source_file, "reason": "no telemetry samples yet"}
    last_t, last = rows[-1]
    window = [s for t, s in rows if now - WINDOW_S <= t <= now + 5.0]

    def series(getter) -> list[float | None]:
        return [getter(s) for s in window]

    temps = last.get("temp_c") if isinstance(last.get("temp_c"), dict) else {}
    gpu = last.get("gpu") if isinstance(last.get("gpu"), dict) else {}
    npu = last.get("npu") if isinstance(last.get("npu"), dict) else {}
    mem = last.get("mem") if isinstance(last.get("mem"), dict) else {}
    temp_values = [_num(v) for v in temps.values()]
    temp_values = [v for v in temp_values if v is not None]
    gpu_busy = _num(gpu.get("busy")) if "busy" in gpu else _num(last.get("gpu_busy"))
    npu_busy = _num(npu.get("busy")) if "busy" in npu else _num(last.get("npu_busy"))
    pkg_w = _num(last.get("pkg_w"))
    return {
        "available": True,
        "file": source_file,
        "age_s": round(now - last_t, 1),
        "stale": now - last_t > STALE_S,
        "src": _text(last.get("src")),
        "samples_in_window": len(window),
        "power": {"pkg_w": pkg_w, "available": pkg_w is not None,
                  "avg_60s_w": _mean(series(lambda s: _num(s.get("pkg_w"))))},
        "temp_c": {"pkg": _num(temps.get("pkg")), "core_max": _num(temps.get("core_max")),
                   "max_all": max(temp_values) if temp_values else None,
                   "avg_60s_pkg": _mean(series(lambda s: _num((s.get("temp_c") or {}).get("pkg")
                                                              if isinstance(s.get("temp_c"), dict) else None)))},
        "cpu_util": {"latest": _num(last.get("cpu_util")),
                     "avg_60s": _mean(series(lambda s: _num(s.get("cpu_util"))))},
        "gpu": {"freq_mhz": _num(gpu.get("freq_mhz")), "max_mhz": _num(gpu.get("max_mhz")),
                "freq_ratio": _num(gpu.get("freq_ratio")), "busy": gpu_busy,
                "avg_60s_freq_ratio": _mean(series(lambda s: _num((s.get("gpu") or {}).get("freq_ratio")
                                                                  if isinstance(s.get("gpu"), dict) else None)))},
        "npu": {"busy": npu_busy, "freq_mhz": _num(npu.get("freq_mhz")),
                "avg_60s_busy": _mean(series(lambda s: _num((s.get("npu") or {}).get("busy")
                                                            if isinstance(s.get("npu"), dict) else None)))},
        "mem": {"used_pct": _num(mem.get("used_pct")), "avail_mb": _num(mem.get("avail_mb"))},
    }


def read_bench(path: Path) -> dict[str, Any]:
    """The benchmark rows of bench.json (P2-B-OPT): device, model, p50/p95 latency, throughput and telemetry."""
    key = str(path)
    try:
        stat = os.stat(path)
    except OSError:
        return {"available": False, "file": key, "reason": "bench.json not found"}
    signature = (stat.st_mtime, stat.st_size)
    cached = _BENCH_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return {"available": False, "file": key, "reason": "bench.json unreadable"}
    if not isinstance(data, dict):
        return {"available": False, "file": key, "reason": "bench.json is not an object"}
    rows: list[dict[str, Any]] = []
    for raw in (data.get("runs") if isinstance(data.get("runs"), list) else [])[:60]:
        if not isinstance(raw, dict):
            continue
        latency = raw.get("latency_ms") if isinstance(raw.get("latency_ms"), dict) else {}
        during = raw.get("telemetry_during_timed") if isinstance(raw.get("telemetry_during_timed"), dict) else {}
        status = _text(raw.get("status")) or "-"
        rows.append({
            "model": _text(raw.get("model")) or "-",
            "device": _text(raw.get("device")) or "-",
            "status": status,
            "p50_ms": _num(latency.get("p50")),
            "p95_ms": _num(latency.get("p95")),
            "throughput_ips": _num(raw.get("throughput_ips")) if raw.get("throughput_ips") is not None
            else _num(raw.get("images_per_s")),
            "temp_c_mean": _num(during.get("pkg_temp_c_mean")),
            "cpu_util_mean": _num(during.get("cpu_util_mean")),
            "npu_busy_mean": _num(during.get("npu_busy_mean")),
            "energy_per_inference_j": _num(raw.get("energy_per_inference_j")),
            "error": None if status == "ok" else (_text(raw.get("error")) or status),
        })
    power = data.get("power") if isinstance(data.get("power"), dict) else {}
    payload = {
        "available": True,
        "file": key,
        "status": _text(data.get("status")),
        "started_utc": _text(data.get("started_utc")),
        "finished_utc": _text(data.get("finished_utc")),
        "openvino": _text(data.get("openvino")),
        "warmup": _num(data.get("warmup")),
        "timed": _num(data.get("timed")),
        "batch": _num(data.get("batch")),
        "power_note": _text(power.get("pkg_w")),
        "energy_note": _text(power.get("energy_per_inference_j")) or "not measured",
        "rows": rows,
    }
    _BENCH_CACHE[key] = (signature, payload)
    return payload


class IntelSources:
    """Reads the telemetry of the current UTC day and the bench file on every snapshot."""

    def __init__(self, telemetry_dir: Path | None, bench_path: Path | None) -> None:
        self.telemetry_dir = None if telemetry_dir is None else Path(telemetry_dir)
        self.bench_path = None if bench_path is None else Path(bench_path)

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        moment = time.time() if now is None else now
        day = dt.datetime.fromtimestamp(moment, dt.timezone.utc)
        telemetry = None
        if self.telemetry_dir is not None:
            path = self.telemetry_dir / f"{day:%Y%m%d}.jsonl"
            telemetry = summarise_telemetry(read_tail(path), moment, str(path))
        bench = read_bench(self.bench_path) if self.bench_path is not None else None
        return {"telemetry": telemetry, "bench": bench}
