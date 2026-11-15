"""Print per-frame brick listing from out/bricks_build.json (for matching label keys to brick ids).
Run: envs/data-env/bin/python -I -B tools/list_bricks_e3.py top|close"""
import json
import sys

WORK = "/home/intel/intel-junction-2026/work/P2-B-DET2/E3"


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else "close"
    recs = json.load(open(f"{WORK}/out/bricks_build.json"))
    frames = {}
    for r in recs:
        if r["view"] == want:
            frames.setdefault((r["ep"], r["which"]), []).append(r)
    for key in sorted(frames):
        print(f"e{key[0]:02d}_{key[1]}  n={len(frames[key])}")
        for r in sorted(frames[key], key=lambda r: r["bbox"][0]):
            cx, cy = r["centre_px"]
            dg = r["tape_feats"]["dark_grey"]
            print(f"  k{r['k']} {r['colour']:7s} c=({cx:.0f},{cy:.0f}) area={r['area']} "
                  f"shape={r.get('shape', '-')} conf={r.get('shape_conf', '-')} touch={int(r['touch_edge'])} "
                  f"dark_grey={dg}")


if __name__ == "__main__":
    main()
