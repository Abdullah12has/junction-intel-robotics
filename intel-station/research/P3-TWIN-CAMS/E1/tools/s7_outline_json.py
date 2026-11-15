#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 7: read-only outline of another session's JSON result files (no writes, no code run).

Prints key paths, types, lengths and short values, up to a given depth, so the result schemas can be read
without loading whole files into context. Usage: envs/data-env/bin/python -I tools/s7_outline_json.py DEPTH FILE...
"""
import json
import sys


def outline(o, depth, maxdepth, prefix):
    if isinstance(o, dict):
        for k, v in list(o.items())[:80]:
            if isinstance(v, (dict, list)) and depth < maxdepth:
                print(f"{prefix}{k}: {type(v).__name__}[{len(v)}]")
                outline(v, depth + 1, maxdepth, prefix + "  ")
            else:
                s = json.dumps(v)
                print(f"{prefix}{k}: {s[:150]}")
    elif isinstance(o, list):
        for i, v in enumerate(o[:3]):
            if isinstance(v, (dict, list)) and depth < maxdepth:
                print(f"{prefix}[{i}]: {type(v).__name__}[{len(v)}]")
                outline(v, depth + 1, maxdepth, prefix + "  ")
            else:
                print(f"{prefix}[{i}]: {json.dumps(v)[:150]}")
        if len(o) > 3:
            print(f"{prefix}... ({len(o)} items)")


def main():
    maxdepth = int(sys.argv[1])
    for path in sys.argv[2:]:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        print("=====", path)
        outline(data, 0, maxdepth, "")


if __name__ == "__main__":
    main()
