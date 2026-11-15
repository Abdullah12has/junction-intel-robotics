import glob
import os
import subprocess

import pyarrow.parquet as pq
from PIL import Image, ImageDraw

R = "/home/intel/intel-junction-2026"
D = R + "/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af"
OUT = R + "/work/P2-D-DATA/V2/frames"
REP = R + "/data/studio-datasets-report/V2"
os.makedirs(OUT, exist_ok=True)
os.makedirs(REP, exist_ok=True)
FPS = 30.0
CAM = {
    "TOP": "realsense 243622061421",
    "CLOSE": "realsense 243722064316",
}
meta = {}
for f in sorted(glob.glob(D + "/meta/episodes/chunk-*/file-*.parquet")):
    for r in pq.read_table(f).to_pylist():
        meta[int(r["episode_index"])] = r


def video_for(e, cam_tag):
    m = meta[e]
    cam = CAM[cam_tag]
    base = "videos/observation.images." + cam + "/"
    ci = m["videos/observation.images." + cam + "/chunk_index"]
    fi = m["videos/observation.images." + cam + "/file_index"]
    ft = m["videos/observation.images." + cam + "/from_timestamp"]
    path = D + "/" + base + "chunk-%03d/file-%03d.mp4" % (ci, fi)
    return path, float(ft)


def grab(e, local, cam_tag, name):
    path, ft = video_for(e, cam_tag)
    t = ft + local / FPS
    out = OUT + "/" + name + ".png"
    rc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "%.4f" % t, "-i", path,
                         "-frames:v", "1", out], capture_output=True, text=True).returncode
    print("GRAB", name, "ep", e, "local", local, cam_tag, "t", round(t, 3), "file", os.path.basename(path), "rc", rc)
    return out if rc == 0 and os.path.exists(out) else None


def sheet(items, cols, out_name, title):
    tiles = []
    for label, p in items:
        if p is None:
            im = Image.new("RGB", (320, 240), (60, 0, 0))
        else:
            im = Image.open(p).convert("RGB").resize((320, 240))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 319, 16], fill=(0, 0, 0))
        d.text((4, 3), label, fill=(255, 255, 0))
        tiles.append(im)
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (320 * cols, 240 * rows + 18), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    d.text((4, 2), title, fill=(255, 255, 255))
    for i, im in enumerate(tiles):
        canvas.paste(im, ((i % cols) * 320, 18 + (i // cols) * 240))
    canvas.save(REP + "/" + out_name)
    print("SHEET", REP + "/" + out_name)


plate = [
    ("REF ep0 f0 TOP arm home", grab(0, 0, "TOP", "ref_ep0_f0_top")),
    ("ep7 release f1209 TOP", grab(7, 1209, "TOP", "ep7_rel1209_top")),
    ("ep7 +0.7s f1230 TOP", grab(7, 1230, "TOP", "ep7_rel1230_top")),
    ("ep8 release f148 TOP", grab(8, 148, "TOP", "ep8_rel148_top")),
    ("ep8 +0.5s f165 TOP", grab(8, 165, "TOP", "ep8_rel165_top")),
    ("ep9 release f187 TOP", grab(9, 187, "TOP", "ep9_rel187_top")),
    ("ep10 release f140 TOP", grab(10, 140, "TOP", "ep10_rel140_top")),
    ("ep10 +0.5s f155 TOP", grab(10, 155, "TOP", "ep10_rel155_top")),
]
sheet(plate, 4, "plate_check_top_view.png", "TOP realsense 243622061421: plates + releases (eps 7-10)")
spot = [
    ("ep3 put-down +0.7s f627 TOP", grab(3, 627, "TOP", "ep3_spot_top")),
    ("ep4 put-down +0.7s f652 TOP", grab(4, 652, "TOP", "ep4_spot_top")),
    ("ep5 put-down +0.7s f566 TOP", grab(5, 566, "TOP", "ep5_spot_top")),
    ("ep6 put-down +0.7s f443 TOP", grab(6, 443, "TOP", "ep6_spot_top")),
    ("ep3 put-down +0.7s f627 CLOSE", grab(3, 627, "CLOSE", "ep3_spot_close")),
    ("ep4 put-down +0.7s f652 CLOSE", grab(4, 652, "CLOSE", "ep4_spot_close")),
    ("ep5 put-down +0.7s f566 CLOSE", grab(5, 566, "CLOSE", "ep5_spot_close")),
    ("ep6 put-down +0.7s f443 CLOSE", grab(6, 443, "CLOSE", "ep6_spot_close")),
]
sheet(spot, 4, "spot_check_put_down.png", "Put-down spot eps 3-6 (row 1 TOP, row 2 CLOSE)")
