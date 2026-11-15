#!/usr/bin/env python3
# REPO-EXPORT E1 read-only discovery. Writes discover.txt next to this script.
# Reads only R/app (git metadata and file bytes), R top-level names, and metric file names.
# Never opens camera, serial or motion devices. Never reads other /home/intel folders.
import ast
import json
import os
import re
import subprocess
import time

R = "/home/intel/intel-junction-2026"
APP = os.path.join(R, "app")
HERE = os.path.dirname(os.path.abspath(__file__))
out_lines = []


def out(s=""):
    out_lines.append(str(s))


def run(args, timeout=20):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).rstrip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except OSError as e:
        return "ERR " + str(e)


def sh(cmd, timeout=20):
    return run(["/bin/sh", "-c", cmd], timeout)


def size(p):
    try:
        return os.stat(p).st_size
    except OSError:
        return -1


def mask(s):
    return re.sub(r"(://[^/:@\s]+:)[^@\s]+@", r"\1***@", s)


# A. git state and history
out("=== A utc and git ===")
out(run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]))
out("HEAD " + run(["git", "-C", APP, "rev-parse", "--short", "HEAD"]))
out("branches: " + run(["git", "-C", APP, "branch", "-a"]).replace("\n", " | "))
out("stash: " + run(["git", "-C", APP, "stash", "list"]))
out("remotes: " + mask(run(["git", "-C", APP, "remote", "-v"])))
out("user: " + run(["git", "-C", APP, "config", "user.name"]) + " / " + run(["git", "-C", APP, "config", "user.email"]))
out("commits (all refs): " + run(["git", "-C", APP, "rev-list", "--count", "--all"]))
out("status porcelain (first 80):")
out(sh("git -C " + APP + " status --porcelain | head -80"))
out("log newest 60 (hash|date|author|subject):")
out(run(["git", "-C", APP, "log", "--date=iso", "--format=%h|%ad|%an|%s", "-n", "60"]))
raw = run(["git", "-C", APP, "log", "--all", "--format=%H%x1f%an%x1f%ae%x1f%B%x1e"], timeout=60)
ai = re.compile(r"\b(claude|anthropic|opus|haiku|sonnet|gpt|llm|assistant|co-authored-by|copilot|ai-generated|agent)\b", re.I)
authors = {}
bad = []
n_rec = 0
for rec in raw.split("\x1e"):
    parts = rec.strip("\n").split("\x1f")
    if len(parts) < 4:
        continue
    n_rec += 1
    h, an, ae = parts[0], parts[1], parts[2]
    body = "\x1f".join(parts[3:])
    authors[an] = authors.get(an, 0) + 1
    if ai.search(body) or ai.search(an) or ai.search(ae):
        bad.append(h[:9] + " | " + body.strip().split("\n")[0][:100])
out("commit records parsed: " + str(n_rec) + "; author counts: " + str(authors))
out("commits with AI words in message or author: " + str(len(bad)))
for b in bad[:40]:
    out("  " + b)

# B. R top level and data/env/repo names (names only)
out("\n=== B R top-level (names and sizes of files; dirs unsized) ===")
for name in sorted(os.listdir(R)):
    p = os.path.join(R, name)
    if os.path.isdir(p):
        out("DIR " + name)
    else:
        out("FILE " + name + " bytes=" + str(size(p)))
for sub in ("data", "envs", "repos"):
    d = os.path.join(R, sub)
    if os.path.isdir(d):
        out(sub + "/ names: " + ", ".join(sorted(os.listdir(d)))[:1200])

# D. R/app files
out("\n=== D R/app files: T tracked, U untracked not ignored; bytes; newline count for text ===")
tracked = set(x for x in run(["git", "-C", APP, "ls-files"]).split("\n") if x)
others = set(x for x in run(["git", "-C", APP, "ls-files", "--others", "--exclude-standard"]).split("\n") if x)
allf = sorted(tracked | others)
out("count " + str(len(allf)) + " tracked " + str(len(tracked)) + " untracked " + str(len(others - tracked)))
TEXT = (".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".tsv", ".jsonl", ".toml", ".cfg", ".ini", ".urdf", ".xml", ".gitignore")
for f in allf:
    p = os.path.join(APP, f)
    n = size(p)
    tag = "T" if f in tracked else "U"
    nl = -1
    if f.lower().endswith(TEXT) and 0 <= n < 3000000:
        try:
            with open(p, "rb") as fh:
                nl = fh.read().count(b"\n")
        except OSError:
            nl = -1
    out(tag + " " + f + " bytes=" + str(n) + " nl=" + str(nl))

# E. ignored files (largest)
out("\n=== E R/app ignored files, largest 40 ===")
ign = [x for x in run(["git", "-C", APP, "ls-files", "--others", "-i", "--exclude-standard"], 60).split("\n") if x]
out("ignored total " + str(len(ign)))
for s, f in sorted(((size(os.path.join(APP, x)), x) for x in ign), reverse=True)[:40]:
    out("I " + f + " bytes=" + str(s))

py_files = [f for f in allf if f.endswith(".py")]

# F. module docstring first line
out("\n=== F module docstring first line per .py ===")
for f in py_files:
    p = os.path.join(APP, f)
    doc = ""
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
        doc = (ast.get_docstring(tree) or "").strip().split("\n")[0][:100]
    except SyntaxError:
        doc = "SYNTAX_ERROR"
    except OSError:
        doc = "UNREADABLE"
    out(f + " :: " + doc)

# G. import closure (ast) from roots, resolved against repo modules by dotted name
index = {}
for f in py_files:
    parts = f[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    for k in range(len(parts)):
        index.setdefault(".".join(parts[k:]), []).append(f)


def module_hits(mod):
    return index.get(mod, [])


def imports_of(f):
    try:
        with open(os.path.join(APP, f), "r", encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return [], []
    mods, strs = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.append(node.module)
            if node.level == 0 and node.module:
                mods += [node.module + "." + a.name for a in node.names]
            elif node.level > 0 and node.module is None:
                mods += [a.name for a in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            if len(v) < 200 and re.search(r"\.(py|sh|json|jsonl|yaml|yml|csv|tsv|urdf|xml|npz|onnx|bin|mjcf|stl|txt|md)$", v):
                strs.append(v)
    return mods, strs


def where(v, f):
    cands = [
        os.path.join(os.path.dirname(os.path.join(APP, f)), v),
        os.path.join(APP, v),
        os.path.join(R, v),
        os.path.join(APP, "tools", v),
        os.path.join(APP, "motion", v),
    ]
    for c in cands:
        if os.path.exists(c):
            return os.path.relpath(os.path.normpath(c), R)
    return "MISSING"


roots = ["tools/sort_any.py", "tools/motion_server.py", "tools/hand_watch.py"]
roots += [f for f in py_files if f.startswith("tools/tests/")]
stack = [r for r in roots if r in py_files]
out("\n=== G import closure from roots " + ", ".join(roots[:3]) + " + tools/tests ===")
out("missing roots: " + ", ".join(r for r in roots if r not in py_files))
seen = {}
while stack:
    f = stack.pop()
    if f in seen:
        continue
    mods, strs = imports_of(f)
    local = []
    ext = []
    for m in mods:
        hits = module_hits(m)
        if not hits and "." in m:
            hits = module_hits(m.rsplit(".", 1)[0])
        if hits:
            local.append(m + "->" + "/".join(sorted(set(hits))))
            stack.extend(h for h in sorted(set(hits)) if h not in seen)
        else:
            ext.append(m)
    seen[f] = (sorted(set(local)), sorted(set(ext)), sorted(set(strs)))

for f in sorted(seen):
    local, ext, strs = seen[f]
    out(f + " | local: " + "; ".join(local))
    out("    ext: " + ", ".join(sorted(set(e.split(".")[0] for e in ext))))
    if strs:
        out("    paths: " + " ; ".join(v + " => " + where(v, f) for v in strs))
out("closure size " + str(len(seen)) + " of " + str(len(py_files)) + " py files")
out("py NOT in closure: " + ", ".join(f for f in py_files if f not in seen))

# H. dynamic imports and subprocess hints
out("\n=== H dynamic import / subprocess / sys.path hints ===")
for f in py_files:
    try:
        with open(os.path.join(APP, f), "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().split("\n")
    except OSError:
        continue
    hits = ["L" + str(i + 1) + ": " + l.strip()[:110] for i, l in enumerate(lines) if re.search(r"importlib|__import__|subprocess|os\.system|sys\.path|Popen", l)]
    if hits:
        out(f + " :: " + " | ".join(hits[:3]))

# I. paths.py and /home/intel references
out("\n=== I paths.py and /home/intel references ===")
pp = sh("find " + APP + " -name paths.py -not -path '*/.git/*' | head -5")
out("paths.py: " + pp)
for line in pp.split("\n"):
    if line.endswith("paths.py") and os.path.exists(line) and size(line) < 8000:
        with open(line, "r", encoding="utf-8", errors="replace") as fh:
            out(fh.read()[:3000])
out("text files under app mentioning /home/intel: " + sh(
    "grep -rl '/home/intel' " + APP + " --exclude-dir=.git --include=*.py --include=*.sh --include=*.json --include=*.yaml --include=*.yml --include=*.md --include=*.txt --include=*.csv 2>/dev/null | wc -l", 60))

# J. metric, label and log candidates (names only, newest first, capped)
out("\n=== J metric/label/cycle candidate files (R/work, R/logs, R/shared, R/sim, R/docs), newest first ===")
pat = re.compile(r"(cycle|metric|bench|auroc|latency|hand|label|summary|det2|telemetry|eval|loo|triage|result|report|verdict)", re.I)
for base in ("work", "logs", "shared", "sim", "docs"):
    d = os.path.join(R, base)
    if not os.path.isdir(d):
        out(base + ": missing")
        continue
    cmd = ("find '" + d + "' -type f \\( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.tsv' -o -name '*.md' -o -name '*.txt' \\) "
           "-not -path '*/.git/*' -not -path '*/.venv/*' -size -4000k 2>/dev/null")
    cands = [p for p in sh(cmd, 60).split("\n") if p and pat.search(os.path.basename(p))]
    cands = [p for p in cands if os.path.exists(p)]
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    out(base + ": " + str(len(cands)) + " candidates")
    for p in cands[:25]:
        s = size(p)
        mt = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(os.path.getmtime(p)))
        keys = ""
        if p.endswith(".json") and 0 <= s < 300000:
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                keys = ("keys=" + ",".join(list(obj.keys())[:14])) if isinstance(obj, dict) else "type=" + type(obj).__name__
            except (OSError, ValueError):
                keys = "json-unreadable"
        out("  " + os.path.relpath(p, R) + " bytes=" + str(s) + " mtime=" + mt + " " + keys)

# K. cycles.jsonl line counts
out("\n=== K cycles.jsonl files under R/logs (line counts) ===")
cj = [p for p in sh("find '" + R + "/logs' -name cycles.jsonl -not -path '*/.git/*' 2>/dev/null", 90).split("\n") if p]
for p in sorted(cj):
    n = 0
    try:
        with open(p, "rb") as fh:
            for _ in fh:
                n += 1
    except OSError:
        n = -1
    out(os.path.relpath(p, R) + " lines=" + str(n) + " bytes=" + str(size(p)))
out("cycles.jsonl files: " + str(len(cj)))

# L. run folder names (names only)
out("\n=== L R/logs entries and sort run folder names ===")
out("R/logs: " + ", ".join(sorted(os.listdir(os.path.join(R, "logs"))))[:1500])
sd = os.path.join(R, "logs", "sort")
if os.path.isdir(sd):
    names = sorted(os.listdir(sd))
    out("logs/sort count " + str(len(names)) + ": " + ", ".join(names[:300]))

# M. secret-like file names (names only, never contents)
out("\n=== M secret-like file names (names only) ===")
out(sh("find '" + R + "' \\( -path '" + R + "/repos' -o -path '" + R + "/envs' -o -path '" + R + "/data/studio-datasets' -o -path '" + R + "/data/sim' -o -path '" + R + "/.cache' -o -path '" + R + "/.tools' \\) -prune -o \\( -name '.env*' -o -name '*.key' -o -name '*.pem' -o -name '*token*' -o -name '*secret*' -o -name '*credential*' \\) -print 2>/dev/null | head -60", 90))

# N. .gitignore of R/app
out("\n=== N R/app .gitignore ===")
gi = os.path.join(APP, ".gitignore")
if os.path.exists(gi):
    with open(gi, "r", encoding="utf-8", errors="replace") as fh:
        out(fh.read()[:3000])
else:
    out("MISSING")

# O. presence and listings
out("\n=== O presence and listings (names only) ===")
checks = [
    ("tools/start_all.sh", "app"),
    ("tools/tests/fixtures", "app"),
    ("motion/gm", "app"),
    ("motion/project_robots.json", "app"),
    ("anomaly/det2_f2", "app"),
    ("ovrun", "app"),
    ("telemetry", "app"),
    ("sim", "R"),
    ("sim/twin", "R"),
    ("docs", "R"),
    ("shared", "R"),
]
for rel, base in checks:
    p = os.path.join(APP if base == "app" else R, rel)
    if os.path.isdir(p):
        out(rel + "/: " + ", ".join(sorted(os.listdir(p)))[:1200])
    elif os.path.exists(p):
        out(rel + ": file bytes=" + str(size(p)))
    else:
        out(rel + ": MISSING")

with open(os.path.join(HERE, "discover.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out_lines) + "\n")
print("WROTE", os.path.join(HERE, "discover.txt"), len(out_lines), "lines")
