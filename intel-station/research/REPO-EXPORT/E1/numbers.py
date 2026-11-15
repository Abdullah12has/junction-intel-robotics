#!/usr/bin/env python3
"""Recompute the numbers used in the README outline from their source logs and metrics.

Run on the laptop with the data-environment interpreter (read-only on R):
    /home/intel/intel-junction-2026/envs/data-env/bin/python -I -B numbers.py

Sources, all inside R: logs/sort/<run>/cycles.jsonl, logs/monitor/hand_watch.log,
logs/telemetry/20261114.jsonl, work/P4-OVBENCH/E1/bench.tsv and metrics.json,
work/P3-TAPE-EVAL/E2/metrics.json, app/anomaly/det2_f2/models/tape_thresholds.json,
app/tools/tests/fixtures (hand_labels.csv and its images) and app/tools/handdet.py,
which is imported for is_hand only (write_heartbeat is never called).
Nothing under R/data or R/run is read. Writes numbers.json into this folder.
"""

import collections
import csv
import datetime
import hashlib
import json
import math
import os
import re
import statistics
import sys

R = "/home/intel/intel-junction-2026"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ("any_1731", "any_1740", "any_2055", "any_2127")
PARITY_ABS_TOL = 0.05  # flag level chosen for this export, not a published standard
PARITY_KEYS = ("tag", "device", "status", "native_batch", "per_image_p50_ms", "call_p50_ms",
               "parity_max_abs_vs_cpu", "parity_max_rel", "error")
POSITIVE = {"hand", "1", "yes", "true", "y", "positive"}
TRUE_WORDS = {"1", "true", "yes", "y"}
SOURCES = {}


def rel(path):
    return os.path.relpath(path, R)


def source(path):
    key = rel(path)
    if key not in SOURCES:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 16), b""):
                digest.update(block)
        SOURCES[key] = {"sha256": digest.hexdigest(), "bytes": os.path.getsize(path)}
    return key


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def to_float(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def ratio(num, den):
    return round(num / den, 4) if den else None


def stats(values):
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pct(q):
        k = (len(ordered) - 1) * q
        lo, hi = math.floor(k), math.ceil(k)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)

    return {"n": len(ordered), "mean": round(statistics.fmean(ordered), 4),
            "median": round(statistics.median(ordered), 4), "p90": round(pct(0.9), 4),
            "min": round(ordered[0], 4), "max": round(ordered[-1], 4)}


def flatten(obj, prefix="", out=None, cap=150):
    if out is None:
        out = []
    if len(out) >= cap:
        return out
    if isinstance(obj, dict):
        for key in sorted(obj):
            flatten(obj[key], "%s.%s" % (prefix, key) if prefix else str(key), out, cap)
    elif isinstance(obj, list):
        for index, item in enumerate(obj[:30]):
            flatten(item, "%s[%d]" % (prefix, index), out, cap)
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        out.append([prefix, obj])
    return out


def parse_time(value):
    if finite(value):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def counts_of(rows, key):
    return dict(collections.Counter(str(row.get(key)) for row in rows).most_common(12))


def cycles_section():
    result = {}
    for run in RUNS:
        path = os.path.join(R, "logs", "sort", run, "cycles.jsonl")
        if not os.path.isfile(path):
            result[run] = {"error": "missing " + rel(path)}
            continue
        src = source(path)
        rows, bad = [], 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    bad += 1
        present = collections.Counter(key for row in rows for key, value in row.items() if value is not None)
        cycle_s = [float(row["timings"]["cycle_s"]) for row in rows
                   if isinstance(row.get("timings"), dict) and finite(row["timings"].get("cycle_s"))]
        spans = []
        for row in rows:
            start, end = parse_time(row.get("t_start")), parse_time(row.get("t_end"))
            if start is not None and end is not None and end >= start:
                spans.append(end - start)
        result[run] = {
            "source": src,
            "rows": len(rows),
            "unparsed_lines": bad,
            "success": counts_of(rows, "success"),
            "failure": counts_of(rows, "failure"),
            "mode": counts_of(rows, "mode"),
            "label": counts_of(rows, "label"),
            "timings_cycle_s": stats(cycle_s),
            "t_end_minus_t_start_s": stats(spans),
            "non_null_fields": dict(present.most_common()),
        }
    return result


def parity_flag(row):
    if row.get("status") != "ok":
        return "not run"
    absolute = to_float(row.get("parity_max_abs_vs_cpu"))
    relative = to_float(row.get("parity_max_rel"))
    if absolute is None:
        return "no parity"
    if absolute == 1 and relative == 1:
        return "placeholder?"
    if absolute > PARITY_ABS_TOL:
        return "deviates"
    return "ok"


def ovbench_section():
    base = os.path.join(R, "work", "P4-OVBENCH", "E1")
    tsv_path = os.path.join(base, "bench.tsv")
    json_path = os.path.join(base, "metrics.json")
    with open(tsv_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    with open(json_path, encoding="utf-8") as handle:
        metrics = json.load(handle)
    table = []
    for row in rows:
        entry = {key: row.get(key) or "" for key in PARITY_KEYS}
        entry["parity"] = parity_flag(row)
        table.append(entry)
    eligible = [e for e in table if e["parity"] == "ok" and not e["error"].strip()
                and to_float(e["per_image_p50_ms"]) is not None]
    best = min(eligible, key=lambda e: to_float(e["per_image_p50_ms"])) if eligible else None
    return {
        "source": source(tsv_path),
        "metrics_source": source(json_path),
        "rows": len(rows),
        "status_counts": dict(collections.Counter(row.get("status") or "" for row in rows)),
        "rows_with_error_text": sum(1 for row in rows if (row.get("error") or "").strip()),
        "parity_counts": dict(collections.Counter(e["parity"] for e in table)),
        "table": table,
        "best_parity_ok_without_error": (
            {"device": best["device"], "tag": best["tag"], "per_image_p50_ms": best["per_image_p50_ms"]}
            if best else None),
        "metrics_json_flat": flatten(metrics),
    }


def tape_section():
    metrics_path = os.path.join(R, "work", "P3-TAPE-EVAL", "E2", "metrics.json")
    thresholds_path = os.path.join(R, "app", "anomaly", "det2_f2", "models", "tape_thresholds.json")
    with open(metrics_path, encoding="utf-8") as handle:
        metrics_text = handle.read()
    metrics = json.loads(metrics_text)
    with open(thresholds_path, encoding="utf-8") as handle:
        thresholds = json.load(handle)
    flat = flatten(metrics, cap=400)
    return {
        "metrics_source": source(metrics_path),
        "thresholds_source": source(thresholds_path),
        "thresholds": flatten(thresholds),
        "auroc_or_loo_keys": [pair for pair in flat
                              if "auroc" in pair[0].lower() or "loo" in pair[0].lower()][:60],
        "values_within_0.0005_of_0.966": [pair for pair in flat
                                          if isinstance(pair[1], float) and abs(pair[1] - 0.966) < 0.0005][:20],
        "raw_0966_line_numbers": [n for n, line in enumerate(metrics_text.splitlines(), 1)
                                  if "0.966" in line][:20],
        "metrics_flat_head": flat[:150],
    }


def hand_section():
    fixtures = os.path.join(R, "app", "tools", "tests", "fixtures")
    labels_path = os.path.join(fixtures, "hand_labels.csv")
    handdet_path = os.path.join(R, "app", "tools", "handdet.py")
    with open(labels_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    images = {}
    for dirpath, _dirs, files in os.walk(fixtures):
        for name in sorted(files):
            images.setdefault(name, os.path.join(dirpath, name))
    sys.path.insert(0, os.path.join(R, "app", "tools"))
    # Imported here so the other sections still run when OpenCV is missing.
    import cv2
    import handdet

    confusion = collections.defaultdict(collections.Counter)
    labels, misses, errors = collections.Counter(), [], []
    for row in rows:
        name = os.path.basename((row.get("file") or "").strip())
        view = (row.get("view") or "").strip()
        label = (row.get("label") or "").strip().lower()
        holding = (row.get("holding") or "").strip().lower() in TRUE_WORDS
        labels[label] += 1
        truth = label in POSITIVE
        path = images.get(name)
        if path is None:
            errors.append({"file": name, "error": "image not found under fixtures"})
            continue
        image = cv2.imread(path)
        if image is None:
            errors.append({"file": name, "error": "cv2.imread returned None"})
            continue
        try:
            predicted, _info = handdet.is_hand(image, view, holding)
        except Exception as exc:  # record the failing row and keep evaluating the rest
            errors.append({"file": name, "error": "%s: %s" % (type(exc).__name__, str(exc)[:120])})
            continue
        predicted = bool(predicted)
        key = "tp" if truth and predicted else "fp" if predicted else "fn" if truth else "tn"
        confusion[view][key] += 1
        confusion["ALL"][key] += 1
        if predicted != truth:
            misses.append({"file": name, "view": view, "label": label, "holding": holding,
                           "predicted": predicted})
    summary = {}
    for view, c in sorted(confusion.items()):
        n = c["tp"] + c["fp"] + c["fn"] + c["tn"]
        summary[view] = {
            "n": n, "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
            "accuracy": ratio(c["tp"] + c["tn"], n),
            "precision": ratio(c["tp"], c["tp"] + c["fp"]),
            "recall": ratio(c["tp"], c["tp"] + c["fn"]),
        }
    return {
        "labels_source": source(labels_path),
        "handdet_source": source(handdet_path),
        "rows": len(rows),
        "label_counts": dict(labels),
        "positive_labels": sorted(POSITIVE & set(labels)),
        "confusion": summary,
        "misses": misses[:40],
        "errors": errors[:40],
    }


def telemetry_section():
    path = os.path.join(R, "logs", "telemetry", "20261114.jsonl")
    keys, numeric, stamps = collections.Counter(), collections.defaultdict(list), {}
    rows, bad = 0, 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(obj, dict):
                bad += 1
                continue
            rows += 1
            for key, value in obj.items():
                keys[key] += 1
                if finite(value):
                    numeric[key].append(float(value))
                if key in ("t", "ts", "time", "timestamp") and value is not None:
                    stamps.setdefault(key, [value, value])[1] = value
    return {
        "source": source(path),
        "rows": rows,
        "unparsed_lines": bad,
        "key_counts": dict(keys.most_common(40)),
        "numeric_stats": {key: stats(values) for key, values in sorted(numeric.items())[:25]},
        "time_key_first_last": stamps,
    }


def hand_watch_section():
    path = os.path.join(R, "logs", "monitor", "hand_watch.log")
    stamp = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
    words, lines, first_ts, last_ts = collections.Counter(), 0, None, None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines += 1
            word = re.search(r"[A-Za-z_]{3,}", line)
            words[word.group(0) if word else "(none)"] += 1
            found = stamp.search(line)
            if found:
                first_ts = first_ts or found.group(0)
                last_ts = found.group(0)
    return {
        "source": source(path),
        "lines": lines,
        "first_word_top": dict(words.most_common(15)),
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
    }


def main():
    sections = {}
    runners = (("cycles", cycles_section), ("ovbench", ovbench_section), ("tape", tape_section),
               ("hand", hand_section), ("telemetry", telemetry_section), ("hand_watch", hand_watch_section))
    for name, run in runners:
        try:
            sections[name] = run()
        except Exception as exc:  # keep the other sections; the failure is recorded in the output
            sections[name] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
    out = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "work/REPO-EXPORT/E1/numbers.py",
        "python": sys.version.split()[0],
        "sources": SOURCES,
    }
    out.update(sections)
    with open(os.path.join(HERE, "numbers.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
        handle.write("\n")
    print("numbers.json written; sections: " + ", ".join(
        "%s=%s" % (name, "error" if "error" in value else "ok") for name, value in sections.items()))


if __name__ == "__main__":
    main()
