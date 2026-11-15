import json, os, subprocess
import numpy as np
from PIL import Image

R = "/home/intel/intel-junction-2026"
WD = os.path.join(R, "work/P3-LABEL-STUDIO/E1")
OUTD = os.path.join(WD, "out")
FRD = os.path.join(WD, "frames", "offset_check")
SD = os.path.join(R, "data", "studio-datasets")
FPS = 30
SERIALS = {"top": "243622061421", "close": "243722064316"}

os.makedirs(FRD, exist_ok=True)
EPS = {}
for line in open(os.path.join(OUTD, "episodes_census.jsonl"), encoding="utf-8"):
    e = json.loads(line)
    EPS["%s:%d" % (e["dataset"][:8], e["episode_index"])] = e


def frame(e, cam, idx, tag):
    k = [x for x in e["video_segments"] if x.endswith(SERIALS[cam])][0]
    s = e["video_segments"][k]
    out = os.path.join(FRD, "%s_%s_f%04d.png" % (tag, cam, idx))
    if not os.path.exists(out):
        mp4 = os.path.join(SD, e["dataset"], s["mp4"])
        subprocess.run(["ffmpeg", "-v", "error", "-ss", "%.4f" % (s["from_ts"] + idx / FPS), "-i", mp4,
                        "-frames:v", "1", "-y", out], capture_output=True, text=True)
    return out if os.path.exists(out) else None


def mae(p, q):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(q).convert("RGB"), dtype=np.float32)
    return float(np.abs(a - b).mean()) if a.shape == b.shape else None


PAIRS = [("a4ba6ede:0", "6d6bdce3:3"), ("a4ba6ede:1", "6d6bdce3:4"),
         ("a4ba6ede:2", "6d6bdce3:5"), ("a4ba6ede:3", "6d6bdce3:6")]
OFFS = range(-3, 4)
out = {}
for ck, rk in PAIRS:
    c, r = EPS[ck], EPS[rk]
    table = {}
    for cam in ("top", "close"):
        for idx in (60, 150, c["n_frames"] // 2):
            a = frame(c, cam, idx, ck.replace(":", "_"))
            for d in OFFS:
                j = idx + d
                if a is None or j < 0 or j >= r["n_frames"] - 2:
                    continue
                b = frame(r, cam, j, rk.replace(":", "_"))
                m = mae(a, b) if b else None
                if m is not None:
                    table.setdefault(cam, {}).setdefault(str(d), []).append(round(m, 3))
    summary = {cam: {d: round(float(np.mean(v)), 3) for d, v in dd.items()} for cam, dd in table.items()}
    out[ck] = {"rep": rk, "n_frames_copy": c["n_frames"], "n_frames_rep": r["n_frames"],
               "raw": table, "mean_mae_by_offset": summary}
    print(ck, "vs", rk, "n", c["n_frames"], r["n_frames"])
    for cam, dd in summary.items():
        best = min(dd, key=lambda d: dd[d])
        print("  ", cam, "mean MAE by offset", dd, "best offset", best, dd[best])
json.dump(out, open(os.path.join(OUTD, "offset_check.json"), "w"), indent=1)
print("written", os.path.join(OUTD, "offset_check.json"))
