#!/usr/bin/env python3
"""v2 curation (D-067): each pick-and-place take ends at release + 1.0 s; the return home is scripted in the loop.

release = first frame after the grasp (gripper closed below 5) where the gripper stays >= 6 for 10 frames in a row.
cut_start = v1 analysis value (idle head trimmed). cut_end = release frame + 30 (1.0 s at 30 fps).
Source is read-only. Datasets are written and checked by build.py (build_dataset, verify), unchanged from v1.

Usage:
  python3 -I build_v2.py cuts  <src_dataset_dir> <analysis_json> <grip_json> <cuts_out_json> <ep,ep,...>
  python3 -I build_v2.py sheet <src_dataset_dir> <cuts_json> <spec_json> <dataset_name> <out_jpg> <camera_key> <W> <H>
  python3 -I build_v2.py build <src_dataset_dir> <cuts_json> <spec_json> <out_dir> <episodes_v2_csv> <episodes_v1_csv>
spec_json: {"datasets": {"NAME": [{"episode": 7, "text": "...", "caption": "..."}]}}
"""
import csv
import glob
import json
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build  # noqa: E402  v1 writer and verifier, used unchanged

FPS = 30.0
POST_RELEASE = 30  # 1.0 s
STABLE = 10        # frames the gripper must stay >= 6
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def load(src):
    data = build.read_glob(src + "/data/chunk-*/file-*.parquet")
    meta = build.read_glob(src + "/meta/episodes/chunk-*/file-*.parquet")
    ep_col = data.column("episode_index").to_numpy()
    fi_col = data.column("frame_index").to_numpy()
    states = np.array(data.column("observation.state").to_pylist(), dtype=float)
    per_ep = {}
    for ep in np.unique(ep_col):
        idx = np.where(ep_col == ep)[0]
        per_ep[int(ep)] = idx[np.argsort(fi_col[idx], kind="stable")]
    return data, meta, fi_col, states, per_ep


def release_index(g):
    """Gripper values in frame order -> (grasp, release) positions; release is None when not found."""
    opened = np.where(g > 8)[0]
    if opened.size == 0:
        return None, None
    closed = np.where(g[opened[0]:] < 5)[0]
    if closed.size == 0:
        return None, None
    grasp = int(opened[0] + closed[0])
    for r in range(grasp + 1, len(g) - STABLE):
        if np.all(g[r:r + STABLE] >= 6):
            return grasp, r
    return grasp, None


def cmd_cuts(src, analysis_path, grip_path, out_json, eps):
    _, _, fi_col, states, per_ep = load(src)
    with open(analysis_path, encoding="utf-8") as fh:
        analysis = {int(r["episode"]): r for r in json.load(fh)}
    with open(grip_path, encoding="utf-8") as fh:
        grip = json.load(fh)["episodes"]
    out = {}
    for ep in eps:
        idx = per_ep[ep]
        g = states[idx, 5]
        fr = fi_col[idx]
        grasp, rel = release_index(g)
        if rel is None:
            raise SystemExit(f"ep{ep}: no grasp followed by a stable release (grasp={grasp})")
        cut_start = int(analysis[ep]["cut_start"])
        cut_end = int(fr[rel]) + POST_RELEASE
        if cut_end > int(fr[-1]) + 1:
            raise SystemExit(f"ep{ep}: release + 1.0 s is past the end of the take")
        pan = float(states[idx[rel], 0])
        out[str(ep)] = {
            "episode": ep,
            "frames_in_take": int(len(idx)),
            "cut_start": cut_start,
            "grasp": int(fr[grasp]),
            "release": int(fr[rel]),
            "release_v1_grip": grip[str(ep)]["release"],
            "release_gripper": round(float(g[rel]), 2),
            "hold_median": round(float(np.median(g[grasp + 10:rel])), 3),
            "pan_release": round(pan, 2),
            "dest": "BLUE" if pan > 0 else ("GREEN" if pan < -30 else "AMBIG"),
            "cut_end": cut_end,
            "frames_kept": cut_end - cut_start,
        }
        print(json.dumps(out[str(ep)]))
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


def cmd_sheet(src, cuts_path, spec_path, name, out_jpg, cam, width, height):
    from PIL import Image, ImageDraw, ImageFont
    with open(cuts_path, encoding="utf-8") as fh:
        cuts = json.load(fh)
    with open(spec_path, encoding="utf-8") as fh:
        members = json.load(fh)["datasets"][name]
    meta = {}
    for path in sorted(glob.glob(src + "/meta/episodes/chunk-*/file-*.parquet")):
        for r in pq.read_table(path).to_pylist():
            meta[int(r["episode_index"])] = r
    pad, label_w, head = 6, 480, 40
    font = ImageFont.truetype(FONT, 16)
    small = ImageFont.truetype(FONT, 14)
    titles = ["start", "grasp (+10 f)", "release (+6 f)", "end (-5 f)"]
    img = Image.new("RGB", (label_w + 4 * (width + pad), head + len(members) * (height + pad)), "white")
    draw = ImageDraw.Draw(img)
    for k, title in enumerate(titles):
        draw.text((label_w + k * (width + pad) + 4, 10), title, fill="black", font=font)
    tmp = tempfile.mkdtemp(prefix="v2-sheet-")
    for r, m in enumerate(members):
        ep = m["episode"]
        c = cuts[str(ep)]
        mr = meta[ep]
        chunk = int(mr[f"videos/{cam}/chunk_index"])
        fidx = int(mr[f"videos/{cam}/file_index"])
        t0 = float(mr[f"videos/{cam}/from_timestamp"])
        video = f"{src}/videos/{cam}/chunk-{chunk:03d}/file-{fidx:03d}.mp4"
        frames = [c["cut_start"], c["grasp"] + 10, c["release"] + 6, c["cut_end"] - 5]
        y = head + r * (height + pad)
        for k, f in enumerate(frames):
            png = os.path.join(tmp, f"ep{ep}_{k}.png")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0 + f / FPS:.4f}", "-i", video,
                            "-frames:v", "1", "-vf", f"scale={width}:{height}", png], check=True)
            x = label_w + k * (width + pad)
            img.paste(Image.open(png).convert("RGB"), (x, y))
            draw.rectangle([x, y + height - 20, x + width, y + height], fill="black")
            draw.text((x + 4, y + height - 19), f"ep{ep} f{f} {f / FPS:.1f}s", fill="yellow", font=small)
        lines = [f"ep{ep:02d}  {m.get('text', '(candidate)')}",
                 f"release pan {c['pan_release']}  dest {c['dest']}  hold {c['hold_median']}  "
                 f"cut {c['cut_start']}-{c['cut_end']}"]
        lines += textwrap.wrap(m.get("caption", ""), width=62)
        ty = y + 2
        for j, line in enumerate(lines):
            draw.text((6, ty), line, fill="black", font=font if j == 0 else small)
            ty += 22 if j == 0 else 17
    img.save(out_jpg, quality=90)
    print("wrote", out_jpg, "size", img.size, "episodes", [m["episode"] for m in members])


def cmd_build(src, cuts_path, spec_path, out_dir, csv_out, v1_csv):
    data, meta, fi_col, _, per_ep = load(src)
    tasks_schema = pq.read_table(src + "/meta/tasks.parquet").schema
    with open(src + "/meta/info.json", encoding="utf-8") as fh:
        info = json.load(fh)
    with open(src + "/meta/stats.json", encoding="utf-8") as fh:
        stats = json.load(fh)
    with open(cuts_path, encoding="utf-8") as fh:
        cuts = json.load(fh)
    with open(spec_path, encoding="utf-8") as fh:
        spec = json.load(fh)["datasets"]
    where, report = {}, []
    for name, members in spec.items():
        items = []
        for m in members:
            c = cuts[str(m["episode"])]
            key = f"{name}#{m['episode']}"
            build.TEXT[key] = m["text"]  # build.py maps an item's dest to its task text
            items.append({"episode": m["episode"], "dest": key, "cut_start": c["cut_start"], "cut_end": c["cut_end"]})
            where.setdefault(m["episode"], []).append((name, m["text"], m.get("caption", "")))
        total, texts = build.build_dataset(name, items, src, out_dir, data, fi_col, meta, tasks_schema, info,
                                           stats, per_ep)
        report.append(f"{name}: episodes={len(items)} frames={total} tasks={texts}")
        report += ["  " + line for line in build.verify(os.path.join(out_dir, name), total)]
    with open(v1_csv, encoding="utf-8") as fh:
        v1 = {int(r["episode"]): r for r in csv.DictReader(fh)}
    fields = ["episode", "v1_decision", "v1_reasons", "v2_decision", "v2_datasets", "v2_text", "dest_by_pan",
              "pan_release", "grasp_frame", "release_frame", "cut_start", "cut_end", "frames_kept", "hold_median",
              "notes"]
    rows = []
    for ep in sorted(v1):
        row = {k: "" for k in fields}
        row.update({"episode": ep, "v1_decision": v1[ep]["decision"], "v1_reasons": v1[ep]["reasons"]})
        if ep in where:
            texts_ep = {t for _, t, _ in where[ep]}
            if len(texts_ep) != 1:
                raise SystemExit(f"ep{ep}: different texts across datasets: {texts_ep}")
            c = cuts[str(ep)]
            row.update({
                "v2_decision": "KEEP", "v2_datasets": "|".join(n for n, _, _ in where[ep]),
                "v2_text": where[ep][0][1], "dest_by_pan": c["dest"], "pan_release": c["pan_release"],
                "grasp_frame": c["grasp"], "release_frame": c["release"], "cut_start": c["cut_start"],
                "cut_end": c["cut_end"], "frames_kept": c["frames_kept"], "hold_median": c["hold_median"],
                "notes": where[ep][0][2],
            })
        else:
            row.update({"v2_decision": "DROP", "notes": "not in any v2 set (D-067: eps 0-2, 8-10, 13 stay out)"})
        rows.append(row)
    with open(csv_out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(report))
    print("wrote", csv_out, len(rows), "rows")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "cuts" and len(sys.argv) == 7:
        cmd_cuts(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], [int(x) for x in sys.argv[6].split(",")])
    elif mode == "sheet" and len(sys.argv) == 10:
        cmd_sheet(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7],
                  int(sys.argv[8]), int(sys.argv[9]))
    elif mode == "build" and len(sys.argv) == 8:
        cmd_build(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
