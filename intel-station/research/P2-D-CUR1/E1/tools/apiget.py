#!/usr/bin/env python3
"""Read-only GET of the local Studio API. Secret-looking keys are replaced before the file is written.

Usage: python3 -I apiget.py <api path> <output json path>
"""
import json
import re
import sys
import urllib.request

BASE = "http://127.0.0.1:7860"
SECRET = re.compile(r"token|secret|passw|api_?key|credential|authoriz", re.I)


def scrub(obj):
    if isinstance(obj, dict):
        return {k: ("<redacted>" if SECRET.search(k) else scrub(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def main():
    path, out = sys.argv[1], sys.argv[2]
    req = urllib.request.Request(BASE + path, method="GET", headers={"Accept": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(scrub(data), fh, indent=1, ensure_ascii=False)
    count = len(data) if isinstance(data, list) else "object"
    print("GET", path, "bytes", len(raw), "items", count, "->", out)


if __name__ == "__main__":
    main()
