import os, re, json
from PIL import Image, ImageDraw, ImageFont
R = "/home/intel/intel-junction-2026"
E1 = R + "/work/P3-DATA-CENSUS/E1"
IMG = os.path.join(E1, "frames/images")
os.makedirs(IMG, exist_ok=True)
units = [json.loads(l) for l in open(E1 + "/raw/unit_detail.jsonl")]
sets = [u for u in units if u["kind"] == "image_set" and u["source"] != "ours/twin-chat"]
picks = []
for u in sets:
    p = u["path"]
    folder = re.sub(r"\s*\[.*\]$", "", p).rstrip("/")
    cands = []
    if "[loose files]" in p:
        cands = [os.path.join(R, folder, f) for f in os.listdir(os.path.join(R, folder)) if f.lower().endswith((".jpg", ".png")) and os.path.isfile(os.path.join(R, folder, f))]
    else:
        for root, _, fs in os.walk(os.path.join(R, folder)):
            for f in fs:
                if f.lower().endswith((".jpg", ".png")):
                    cands.append(os.path.join(root, f))
    m = re.search(r"estop_(\w+)", p)
    if m:
        cands = [c for c in cands if ("estop_" + m.group(1)) in os.path.basename(c)]
    cands.sort()
    picks.append((u["id"], p, cands[0] if cands else None, len(cands)))
font = ImageFont.load_default()
with open(E1 + "/raw/image_pick.tsv", "w") as f:
    f.write("id\tpath\tpicked_file\tcandidates\n")
    for pid, p, pick, n in picks:
        f.write("%s\t%s\t%s\t%d\n" % (pid, p, os.path.relpath(pick, R) if pick else "-", n))
CW, CH, COLS, ROWS = 326, 262, 5, 2
per = COLS * ROWS
for si in range(0, len(picks), per):
    chunk = picks[si:si + per]
    sheet = Image.new("RGB", (COLS * CW, ROWS * CH), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for i, (pid, p, pick, n) in enumerate(chunk):
        x0 = (i % COLS) * CW + 3
        y0 = (i // COLS) * CH + 3
        d.text((x0, y0), ("%s %s" % (pid, p))[:58], fill=(255, 255, 255), font=font)
        if pick:
            im = Image.open(pick).convert("RGB")
            im.thumbnail((320, 240))
            sheet.paste(im, (x0, y0 + 14))
            d.text((x0, y0 + 14 + im.size[1] + 2), ("file " + os.path.basename(pick))[:58], fill=(190, 190, 190), font=font)
        else:
            d.text((x0 + 4, y0 + 60), "NO IMAGE FILE", fill=(255, 0, 0), font=font)
    sheet.save(os.path.join(IMG, "image_sheet_%02d.png" % (si // per + 1)))
print("image sets", len(picks), "sheets", (len(picks) + per - 1) // per)
