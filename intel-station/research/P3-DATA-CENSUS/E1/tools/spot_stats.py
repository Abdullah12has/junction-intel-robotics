import os, re
from PIL import Image, ImageStat, ImageDraw, ImageFont, ImageChops
R = "/home/intel/intel-junction-2026"
E1 = R + "/work/P3-DATA-CENSUS/E1"
FR = E1 + "/frames"
SH = FR + "/sheets"
os.makedirs(SH, exist_ok=True)
plan = [l.rstrip("\n").split("\t") for l in open(E1 + "/raw/spot_plan.tsv")][1:]
font = ImageFont.load_default()
items = []
with open(E1 + "/raw/frame_stats.tsv", "w") as sf:
    sf.write("key\tsource\tkind\tpath\ta_mean\ta_std\tb_mean\tb_std\tflags\n")
    for key, src, kind, path, dur, ta, tb in plan:
        ims = {}
        flags = []
        vals = {}
        for tag in ("a", "b"):
            fp = os.path.join(FR, "%s_%s.png" % (key, tag))
            if os.path.exists(fp):
                im = Image.open(fp).convert("RGB")
                ims[tag] = im
                st = ImageStat.Stat(im.convert("L"))
                vals[tag] = (round(st.mean[0], 1), round(st.stddev[0], 1))
                if st.mean[0] < 10:
                    flags.append("DARK_" + tag)
                if st.stddev[0] < 4:
                    flags.append("FLAT_" + tag)
            else:
                flags.append("FRAME_MISSING_" + tag)
        if "a" in ims and "b" in ims:
            diff = ImageStat.Stat(ImageChops.difference(ims["a"], ims["b"]).convert("L")).mean[0]
            if diff < 0.5:
                flags.append("SAME_FRAMES")
        if dur == "-":
            flags.append("NO_DURATION")
        sf.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (key, src, kind, path,
                 vals.get("a", ("-", "-"))[0], vals.get("a", ("-", "-"))[1],
                 vals.get("b", ("-", "-"))[0], vals.get("b", ("-", "-"))[1], ";".join(flags) or "-"))
        short = os.path.basename(path)
        if kind == "sim_attempt":
            short = "sim " + path.split("/")[-2] + " " + path.split("/")[-1][:-4]
        if path.startswith("data/sim/2026-11-14/extract-lerobot"):
            short = "lerobot " + "/".join(path.split("/")[-3:])
        items.append((key, src, path, short, flags, ims))
CW, CH, COLS, ROWS = 392, 166, 3, 4
index = open(E1 + "/raw/sheet_index.tsv", "w")
index.write("sheet\tkey\tsource\tpath\tflags\n")
for si in range(0, len(items), COLS * ROWS):
    chunk = items[si:si + COLS * ROWS]
    sheet = Image.new("RGB", (COLS * CW, ROWS * CH), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    sn = si // (COLS * ROWS) + 1
    for i, (key, src, path, short, flags, ims) in enumerate(chunk):
        x0 = (i % COLS) * CW + 4
        y0 = (i // COLS) * CH + 4
        cap = ("%s %s a25 b75" % (key, short))[:62]
        d.text((x0, y0), cap, fill=(255, 255, 255), font=font)
        for j, tag in enumerate(("a", "b")):
            px = x0 + j * 196
            if tag in ims:
                sheet.paste(ims[tag], (px, y0 + 14))
            else:
                d.rectangle([px, y0 + 14, px + 191, y0 + 14 + 143], outline=(255, 0, 0))
                d.text((px + 4, y0 + 70), "MISSING", fill=(255, 0, 0), font=font)
        index.write("%02d\t%s\t%s\t%s\t%s\n" % (sn, key, src, path, ";".join(flags) or "-"))
    sheet.save(os.path.join(SH, "sheet_%02d.png" % sn))
index.close()
print("items", len(items), "sheets", (len(items) + COLS * ROWS - 1) // (COLS * ROWS))
