"""Scratch tool of builder E1 (not a deliverable). Writes app/tools/tests/fixtures/MANIFEST.tsv: one row per fixture file.

Expected values come from the sources: episode.json (hover_blob, pre_close, t_start), the servo lines of
logs/sort_any_any_1740.out, raw/L19_extract_summary.tsv (frame table), hand_labels.csv, scan_calib.json.
The constants of tests/conftest.py are parsed with ast (conftest is not imported) and checked against those sources;
a disagreement is printed as PROBLEM. Copies are compared byte for byte with their source by sha256.
"""
import ast
import csv
import datetime
import hashlib
import json
import os
import re
import sys

REPO = "/home/intel/intel-junction-2026"
TESTS = os.path.join(REPO, "app/tools/tests")
FIX = os.path.join(TESTS, "fixtures")
RAW = os.path.join(REPO, "work/P2-B-TESTS/E1/raw")
EPS = os.path.join(REPO, "logs/sort/any_1740")
SERVO_LOG = os.path.join(REPO, "logs/sort_any_any_1740.out")
OUT = os.path.join(FIX, "MANIFEST.tsv")
rows = {}
problems = []


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def utc(ts):
    return datetime.datetime.fromtimestamp(float(ts), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_mtime(path):
    return utc(os.path.getmtime(path))


def conftest_literal(name):
    path = os.path.join(TESTS, "conftest.py")
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def episode(ep):
    with open(os.path.join(EPS, f"ep_{ep}", "episode.json"), encoding="utf-8") as f:
        return json.load(f)


def t_start(e):
    return utc(e["t_start"]) if "t_start" in e else "not recorded"


def compare_to_source(fix_rel, src_rel):
    """sha256 of the fixture copy and a verdict on the byte copy against its source."""
    fix_sha = sha256(os.path.join(FIX, fix_rel))
    src = os.path.join(REPO, src_rel)
    if not os.path.isfile(src):
        problems.append(f"{fix_rel}: source missing: {src_rel}")
        return fix_sha, "source missing"
    same = sha256(src) == fix_sha
    if not same:
        problems.append(f"{fix_rel}: differs from {src_rel}")
    return fix_sha, ("identical to source" if same else "DIFFERS from source")


def add(fix_rel, source, when, expected, how, sha):
    rows[fix_rel] = [fix_rel, source, when, expected, how, sha]


with open(os.path.join(TESTS, "conftest.py"), encoding="utf-8") as f:
    conftest_text = f.read()
if "split/split_1816%02d.png" not in conftest_text:
    problems.append("conftest.py: SPLIT pattern not found")

# Fixture 1: hover blobs
for rel, colour, (x, y), short, hue in conftest_literal("HOVER"):
    ep = re.search(r"ep(\d{4})", rel).group(1)
    e = episode(ep)
    hb = e["hover_blob"]
    ok = (abs(hb["centre"][0] - x) < 0.05 and abs(hb["centre"][1] - y) < 0.05 and abs(hb["short"] - short) < 0.05
          and hb.get("hue") == hue and (colour is None or hb.get("colour") == colour))
    if not ok:
        problems.append(f"HOVER {rel}: conftest {(x, y, short, hue, colour)} vs episode.json "
                        f"{(hb['centre'], hb['short'], hb.get('hue'), hb.get('colour'))}")
    src = f"logs/sort/any_1740/ep_{ep}/crops/wrist_hover.png"
    sha, state = compare_to_source(rel, src)
    add(rel, src, f"episode t_start {t_start(e)}",
        f"centre=({x},{y}) short={short} hue={hue} colour={colour or 'not checked'} (episode.json hover_blob)",
        f"{state}; episode.json hover_blob {'matches' if ok else 'DIFFERS'} (builder check); test_hover_blob_replay", sha)

# Fixture 5: close pre-verdicts
for ep, rel, verdict, reasons, want, hue_rec in conftest_literal("CLOSE"):
    e = episode(ep)
    pc = e["pre_close"]
    if "hover_blob" in e:
        want_src = f"hover_blob hue {e['hover_blob']['hue']}"
        want_ok = e["hover_blob"]["hue"] == want
    else:
        want_src = f"pre_close hue {pc['hue']} + hue_diff {pc['hue_diff']}"
        want_ok = pc["hue"] + pc["hue_diff"] == want
    ok = (pc["verdict"] == verdict and pc["reasons"] == reasons and want_ok
          and (hue_rec is None or pc.get("hue") == hue_rec))
    if not ok:
        problems.append(f"CLOSE {rel}: conftest {(verdict, reasons, want, hue_rec)} vs episode.json "
                        f"{(pc['verdict'], pc['reasons'], want_src, pc.get('hue'))}")
    src = f"logs/sort/any_1740/ep_{ep}/crops/pre_close_full.jpg"
    sha, state = compare_to_source(rel, src)
    add(rel, src, f"episode t_start {t_start(e)}",
        f"verdict={verdict} reasons={reasons} want_hue={want} ({want_src})",
        f"{state}; episode.json pre_close {'matches' if ok else 'DIFFERS'} (builder check); "
        f"test_close_pre_verdict_replay", sha)

# Frame table from the extraction run (raw/L19_extract_summary.tsv): servo and split frames
with open(os.path.join(RAW, "L19_extract_summary.tsv"), encoding="utf-8", newline="") as f:
    frames = {r["file"]: r for r in csv.DictReader(f, delimiter="\t")}


def servo_line(n):
    with open(SERVO_LOG, encoding="utf-8", errors="replace") as f:
        line = f.read().splitlines()[n - 1]
    t = re.search(r"(\d{2}:\d{2}:\d{2})Z", line)
    b = re.search(r"(\w+) brick \[([\d.]+), ([\d.]+)\]", line)
    s = re.search(r"short ([\d.]+)", line)
    if not (t and b and s):
        return None, line
    return {"utc": t.group(1), "colour": b.group(1), "x": float(b.group(2)), "y": float(b.group(3)),
            "short": float(s.group(1))}, line


# Fixture 2: servo frames (logged values read from the log line itself)
for rel, colour, x, y, short, n in conftest_literal("SERVO"):
    got, line = servo_line(n)
    ok = (got is not None and got["colour"] == colour and abs(got["x"] - x) < 0.05
          and abs(got["y"] - y) < 0.05 and abs(got["short"] - short) < 0.05)
    if not ok:
        problems.append(f"SERVO {rel}: conftest {(colour, x, y, short, n)} vs log line {n}: {line[:160]}")
    fr = frames[os.path.basename(rel)]
    add(rel, f"logs/sort_any_any_1740.out line {n}; video data/wrist/raw/{fr['segment']} at +{fr['offset_used_s']} s",
        f"2026-11-14 {got['utc'] if got else '?'}Z (log line {n})",
        f"colour={colour} centre=({x},{y}) short={short} (log line {n})",
        f"frame extracted at the offset (raw/L20_extract_v2.txt); log line parsed {'matches' if ok else 'DIFFERS'} "
        f"(builder check); test_servo_replay_seeded_at_logged_centre", sha256(os.path.join(FIX, rel)))

# Fixture 4: split frames (names computed exactly as conftest.SPLIT)
for s in range(10, 31, 2):
    rel = f"split/split_1816{s:02d}.png"
    fr = frames[os.path.basename(rel)]
    add(rel, f"video data/wrist/raw/{fr['segment']} at +{fr['offset_used_s']} s",
        f"2026-11-14 {fr['utc']}Z (ep 16 window 18:16:10Z-18:16:30Z)",
        "one white blob from blobs(min_area=1500), short 60-140 px (test rule)",
        "frame extracted at the offset (raw/L20_extract_v2.txt); table row in raw/L19_extract_summary.tsv; "
        "test_split_brick_single_white_blob_1816Z", sha256(os.path.join(FIX, rel)))

# Fixture 3: scan frame
src = "logs/depth/servo/scan_now.jpg"
sha, state = compare_to_source("scan_now.jpg", src)
add("scan_now.jpg", src, f"{utc_mtime(os.path.join(REPO, src))} (file mtime; spec: 17:54Z)",
    "candidates keep a red and a yellow blob as separate candidates; green and blue plate blobs and the white brick "
    "(r = 346.5 mm) are excluded",
    f"{state}; test_scan_candidates_exclude_plates_1754Z; probe output raw/L21_probe.txt and raw/L23_probe2.txt", sha)

# scan_calib.json (kin fixture 7)
src = "app/tools/scan_calib.json"
sha, state = compare_to_source("scan_calib.json", src)
with open(os.path.join(FIX, "scan_calib.json"), encoding="utf-8") as f:
    cal = json.load(f)
add("scan_calib.json", src, f"{utc_mtime(os.path.join(REPO, src))} (file mtime)",
    f"q_scan={cal.get('q_scan')} q_hover={cal.get('q_hover')} scale={cal.get('scale')} "
    f"target_scan={cal.get('target_scan')}",
    f"{state}; test_fk_scan_and_hover_heights; test_ik_rel_round_trip_5x5_grid_60mm", sha)

# Fixture 6: estop frames and the labels
lab_path = os.path.join(FIX, "hand_labels.csv")
with open(lab_path, encoding="utf-8", newline="") as f:
    label_rows = list(csv.DictReader(f))
labels = {os.path.basename(r["file"]): r for r in label_rows}
for name in sorted(os.listdir(os.path.join(FIX, "hand"))):
    m = re.match(r"estop_(close|top|wrist)_(\d+)\.jpg$", name)
    if not m:
        problems.append(f"unexpected hand fixture name {name}")
        continue
    src = f"logs/monitor/{name}"
    sha, state = compare_to_source(f"hand/{name}", src)
    r = labels.get(name)
    if r is None:
        expected = "no label row in hand_labels.csv (smoke test only)"
        how = f"{state}; test_handdet_runs_on_all_estop_frames"
    else:
        expected = f"label={r['label']} holding={r['holding']}" + (f" ({r['note']})" if r["note"] else "")
        how = (f"{state}; test_handdet_runs_on_all_estop_frames; "
               + ("wrist row: not asserted (wrist view gated off while holding)" if r["view"] == "wrist"
                  else "test_hand_labels_if_present"))
    add(f"hand/{name}", src, f"{utc(m.group(2))} (epoch in file name)", expected, how, sha)

feat = os.path.join(REPO, "work/handdet/labels_features.tsv")
counts = {}
for r in label_rows:
    key = (r["view"], r["label"])
    counts[key] = counts.get(key, 0) + 1
summary = "; ".join(f"{v} {lab} {n}" for (v, lab), n in sorted(counts.items()))
add("hand_labels.csv",
    f"Opus labels (coordinator message 18:53Z); feature table work/handdet/labels_features.tsv sha256 "
    f"{sha256(feat)} mtime {utc_mtime(feat)}",
    f"{utc_mtime(lab_path)} (file mtime)",
    f"{len(label_rows)} rows: {summary}; holding 0 or 1",
    "not re-derived by E1; used by test_hand_labels_if_present (wrist rows skipped, hand_partial not asserted)",
    sha256(lab_path))

# Coverage: every file under fixtures/ has exactly one row
found = set()
for dirpath, _dirs, files in os.walk(FIX):
    for name in files:
        rel = os.path.relpath(os.path.join(dirpath, name), FIX)
        if rel != "MANIFEST.tsv":
            found.add(rel)
if set(rows) != found:
    problems.append(f"coverage: rows without file {sorted(set(rows) - found)}; "
                    f"files without row {sorted(found - set(rows))}")

with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
    w.writerow(["file", "source_path", "source_time_utc", "expected_values", "how_verified", "sha256"])
    for k in sorted(rows):
        w.writerow([str(v).replace("\t", " ").replace("\n", " ") for v in rows[k]])

groups = {}
for k in rows:
    g = k.split("/")[0] if "/" in k else k
    groups[g] = groups.get(g, 0) + 1
print("rows", len(rows), "fixture files on disk", len(found), "groups", groups)
print("problems", len(problems))
for p in problems:
    print("PROBLEM", p)
print("MANIFEST", OUT, "sha256", sha256(OUT))
sys.exit(2 if problems else 0)
