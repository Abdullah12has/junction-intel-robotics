"""Build the public repository from app/: the live system's import closure, its scripts and configs, the tests and
their non-image fixtures. Text is scrubbed (assistant names, personal names, workspace paths); then gates run and
the export aborts on any hit: assistant/personal words, secrets, absolute home paths, images outside the allowlist,
files over 5 MB. Writes EXPORT/ (a git work tree) and prints the file list. Usage: python -I export.py APP EXPORT"""
import fnmatch
import glob
import modulefinder
import os
import re
import shutil
import sys

APP, OUT = (os.path.abspath(a) for a in sys.argv[1:3])
ENTRY = ["tools/sort_any.py", "tools/hand_watch.py", "tools/motion_server.py", "tools/jaw_check.py",
         "tools/run_report.py", "tools/cycle_peek.py", "tools/mcmd.py", "tools/mseq.py", "tools/tests/run_tests.py"]
SEARCH = [os.path.join(APP, d) for d in ("tools", "tools/tests", "anomaly/det2_f2", "motion", "eyeinhand", ".")]
EXTRA = ["README.md", ".gitignore.public", "config/*.json", "tools/plan_survey.py", "tools/fit_faces.py", "tools/park_release.sh", "eyeinhand/wristcam.py", "eyeinhand/wrist_cam_model.json", "tools/start_all.sh", "tools/stop_all.sh", "tools/monitor_cams.sh", "tools/wrist_rec.sh", "tools/README.md",
         "motion/gm/*.yaml", "motion/gm/*.json", "motion/project_robots.json",
         "tools/scan_calib.json", "tools/wrist_calib.json", "tools/jaw_calib.json",
         "tools/wrist_jaw_mask.png", "tools/close_jaw_mask_hover.png",
         "anomaly/det2_f2/models/*.json", "tools/tests/*.md",
         "tools/tests/fixtures/*.csv", "tools/tests/fixtures/*.tsv", "tools/tests/fixtures/*.json",
         "tools/tests/fixtures/*/*.csv", "tools/tests/fixtures/*/*.tsv", "tools/tests/fixtures/*/*.json",
         "tools/tests/fixtures/*/*.jsonl"]
IMG_ALLOW = {"tools/wrist_jaw_mask.png", "tools/close_jaw_mask_hover.png"}
TEXT = (".public", ".py", ".sh", ".md", ".yaml", ".yml", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".cfg", ".toml")

files = set()
for e in ENTRY:
    f = os.path.join(APP, e)
    mf = modulefinder.ModuleFinder(path=[os.path.dirname(f)] + SEARCH + sys.path)
    mf.run_script(f)
    for m in mf.modules.values():
        if m.__file__ and os.path.abspath(m.__file__).startswith(APP + os.sep):
            files.add(os.path.relpath(os.path.abspath(m.__file__), APP))
    files.add(e)
files |= {f"tools/tests/{t}" for t in os.listdir(os.path.join(APP, "tools/tests")) if t.endswith(".py")}
for pat in EXTRA:
    for p in glob.glob(os.path.join(APP, pat)):
        files.add(os.path.relpath(p, APP))
files = sorted(f for f in files if f != ".gitignore" and "__pycache__" not in f and not f.endswith((".orig", ".tmp")))

SUBS = [
    (re.compile(r'"""Opus (?:I-\d+(?:\.\.I-\d+)?)?:?\s*'), '"""'),
    (re.compile(r"\s*\((?:[^()]*\b)?Opus\b[^()]*\)"), ""),
    (re.compile(r"Opus's file"), "a manual-label file"),
    (re.compile(r"\bOpus\b"), "the operator"),
    (re.compile(r"Hassan's"), "the operator's"),
    (re.compile(r"\bHassan\b"), "the operator"),
    (re.compile(r"^.*?/home/claude.*$\n?", re.M), ""),
    (re.compile(r"/home/intel/intel-junction-2026/envs/"), "envs/"),
    (re.compile(r"/home/intel/intel-junction-2026/app/"), "app/"),
    (re.compile(r"/home/intel/intel-junction-2026"), "<workspace>"),
]
CAP = re.compile(r'"""([a-z])')

if os.path.isdir(OUT):
    for x in os.listdir(OUT):
        if x != ".git":
            p = os.path.join(OUT, x)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
os.makedirs(OUT, exist_ok=True)
for f in files:
    src, dst = os.path.join(APP, f), os.path.join(OUT, f)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if f.endswith(TEXT):
        s = open(src, errors="strict").read()
        for rx, rep in SUBS:
            s = rx.sub(rep, s)
        s = CAP.sub(lambda m: '"""' + m.group(1).upper(), s)
        open(dst, "w").write(s)
    else:
        shutil.copy2(src, dst)
    if f.endswith(".sh"):
        os.chmod(dst, 0o755)
if os.path.exists(os.path.join(OUT, ".gitignore.public")):
    os.replace(os.path.join(OUT, ".gitignore.public"), os.path.join(OUT, ".gitignore"))

GATES = {
    "assistant": re.compile(r"\b(opus|claude|anthropic|haiku|chatgpt|llm)\b|opus[-_]|_opus", re.I),
    "personal": re.compile(r"hsn07pk|Hassan|@gmail\.com|/home/claude|/home/intel"),
    "secret": re.compile(r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
                         r"BEGIN [A-Z ]*PRIVATE KEY|(api[_-]?key|password|passwd|secret|token)\s*[:=]\s*['\"][^'\"]{6,}", re.I),
}
bad = []
for root, dirs, fs in os.walk(OUT):
    dirs[:] = [d for d in dirs if d != ".git"]
    for x in fs:
        p = os.path.join(root, x)
        rel = os.path.relpath(p, OUT)
        if os.path.getsize(p) > 5 * 1024 * 1024:
            bad.append(("size", rel, os.path.getsize(p)))
        if rel.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".mkv", ".mp4")) and rel not in IMG_ALLOW:
            bad.append(("image", rel, ""))
        if rel.endswith(TEXT):
            for i, ln in enumerate(open(p, errors="replace"), 1):
                for g, rx in GATES.items():
                    if rx.search(ln):
                        bad.append((g, f"{rel}:{i}", ln.strip()[:120]))
print("FILES", len(files))
for f in files:
    print("  ", f)
print("GATE HITS", len(bad))
for b in bad:
    print("  ", *b)
sys.exit(1 if bad else 0)
