#!/usr/bin/env python3
"""Export scan for the robot-cell application.

Run on the laptop with the data-environment interpreter (read-only on R):
    /home/intel/intel-junction-2026/envs/data-env/bin/python -I -B scan.py

Reads the application tree R/app (git listing and text files), the R/work, R/app and
R/docs trees for the coefficient search, R/repos for the URDF locate, and the cycle
logs for the scrub. It does not read R/data, R/run, device paths or other /home folders.
Writes into this folder only: filelist.tsv, scrub_hits.tsv, scan_summary.txt.
"""

import ast
import collections
import datetime
import os
import re
import subprocess
import sys

R = "/home/intel/intel-junction-2026"
APP = os.path.join(R, "app")
HERE = os.path.dirname(os.path.abspath(__file__))
STDLIB = frozenset(getattr(sys, "stdlib_module_names", ()))
SKIP_DIRS = frozenset({"venv", ".venv", "uvcache", "site-packages", "node_modules",
                       "__pycache__", ".git", ".cache", "pkg"})
SKIP_WALK = re.compile(r"STUDIO|FRIEND", re.IGNORECASE)
TEXT_EXT = frozenset({".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".json", ".jsonl",
                      ".csv", ".tsv", ".urdf", ".cfg", ".toml", ".ini"})
TEXT_NAMES = frozenset({".gitignore", "LICENSE", "NOTICE", "README"})
CODE_EXT = frozenset({".py", ".sh"})
IMAGE_EXT = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
ARTEFACT_EXT = frozenset({".npz", ".npy", ".pt", ".pth", ".onnx", ".bin", ".ckpt",
                          ".safetensors", ".pkl", ".engine", ".blob", ".mp4", ".mkv"})
SEARCH_EXT = frozenset({".md", ".txt", ".log", ".json", ".jsonl", ".tsv", ".csv",
                        ".py", ".sh", ".yaml", ".yml"})
COUNT_LIMIT = 3000000
SEARCH_LIMIT = 400000
LIT_RE = re.compile(r"^[\w./-]+\.(?:py|sh|json|jsonl|yaml|yml|csv|tsv|png|jpg|npz|urdf|txt)$")
TOKEN_RE = re.compile(r"[\w./-]+\.(?:py|sh|json|jsonl|yaml|yml|csv|tsv|png|jpg|npz|urdf|txt)\b")
DOC_REF_RE = re.compile(r"\b[\w.-]+\.(?:py|sh|yaml|yml|json|jsonl|csv|tsv|urdf|png|jpg)\b")
FIGURE_RE = re.compile(r"0\.966|\bLOO\b|leave[- ]one[- ]out|\bAUROC\b", re.IGNORECASE)

# Export roots are the live entry points, tests, lint, benchmark and documents.
# Everything else is kept only if an IN file imports it or names it in a path literal.
ROOT_FILES = {
    "tools/sort_any.py": "live entry point: the sort loop",
    "tools/hand_watch.py": "live hand watchdog process",
    "tools/motion_server.py": "live motion server process",
    "assets/so101/so101.urdf": "vendored robot description (upstream license Apache-2.0)",
}
ROOT_PREFIXES = (
    ("tools/tests/", "offline test suite"),
    ("anomaly/det2_f2/tests/", "detector test suite"),
    ("docs/", "project document"),
    ("bench/", "OpenVINO benchmark harness"),
    ("results/ovbench/", "OpenVINO measurement evidence"),
)
REVIEW_FILES = {
    "README.md": "existing README: replaced by the export README (readme_outline.md)",
    "motion/project_robots.json": "contains a device connection string: template before publishing",
    "telemetry/run_final.sh": "hard-coded P2-B-OPT job paths and an NPU sysfs path: check before publishing",
    "telemetry/bench_compact.py": "hard-coded P2-B-OPT work-folder path: check before publishing",
    "results/tape/metrics.json": "tape rule variants: state which one ships",
    "results/tape/results.tsv": "check for absolute paths before publishing",
    "results/tape/eval_set.tsv": "absolute paths inside: rewrite before publishing",
}
# (source relative to R, export path, forced decision or None, reason for forced decision)
EXTRA = [
    ("work/P4-OVBENCH/E1/bench.tsv", "results/ovbench/bench.tsv", None, ""),
    ("work/P4-OVBENCH/E1/metrics.json", "results/ovbench/metrics.json", None, ""),
    ("work/P4-OVBENCH/E1/tools/run_ovbench.sh", "bench/run_ovbench.sh", None, ""),
    ("work/P4-OVBENCH/E1/tools/bench_ov_v2.py", "bench/bench_ov_v2.py", None, ""),
    ("work/P4-OVBENCH/E1/tools/aggregate_v2.py", "bench/aggregate_v2.py", None, ""),
    ("work/P4-OVBENCH/E1/tools/bench_ov.py", "bench/bench_ov.py", "OUT",
     "earlier version of bench_ov_v2.py (256 lines against 269): superseded by name, confirm"),
    ("work/P3-TAPE-EVAL/E2/metrics.json", "results/tape/metrics.json", None, ""),
    ("work/P3-TAPE-EVAL/E2/results.tsv", "results/tape/results.tsv", None, ""),
    ("work/P3-TAPE-EVAL/E2/eval_set.tsv", "results/tape/eval_set.tsv", None, ""),
    ("logs/sort/any_1731/cycles.jsonl", "results/cycles/any_1731.jsonl", None, ""),
    ("logs/sort/any_1740/cycles.jsonl", "results/cycles/any_1740.jsonl", None, ""),
    ("logs/sort/any_2055/cycles.jsonl", "results/cycles/any_2055.jsonl", None, ""),
    ("logs/sort/any_2127/cycles.jsonl", "results/cycles/any_2127.jsonl", None, ""),
    ("logs/monitor/hand_watch.log", "laptop-only/monitor/hand_watch.log", "OUT",
     "operator log kept on the laptop; numbers.json cites it"),
    ("logs/telemetry/20261114.jsonl", "laptop-only/telemetry/20261114.jsonl", "OUT",
     "telemetry log kept on the laptop; numbers.json cites it"),
]
# (export path, search root under R, file name, path fragment that must match)
LOCATE = [
    ("assets/so101/so101.urdf", "repos", "so101.urdf", "so101"),
    ("tools/lint/attrs.py", "work", "attrs.py", "lint"),
    ("tools/lint/undef.py", "work", "undef.py", "lint"),
    ("tools/lint/smoke_offline.py", "work", "smoke_offline.py", "lint"),
]
PATS = [
    ("ai-word", re.compile(r"\b(?:Claude|Anthropic|Opus|Haiku|Sonnet|GPT|ChatGPT|LLM|Copilot|agents?|assistant)\b",
                           re.IGNORECASE)),
    ("ai-word", re.compile(r"\bAI\b|\bai\b|\bA\.I\.")),
    ("decision-id", re.compile(r"\b(?:I-\d+(?:\.\d+)?|OD-\d+|D-\d{2,3}|L-\d{3}|CC-[A-Z0-9-]+|P[234]-[A-Z0-9-]+)\b")),
    ("worker-id", re.compile(r"\bE[1-9]\b")),
    ("clock-stamp", re.compile(r"(?<!\d)\d{2}:\d{2}(?::\d{2})?Z\b")),
    ("person", re.compile(r"\b(?:hassan|hsn07pk)\b", re.IGNORECASE)),
    ("person-context", re.compile(r"\bfriend", re.IGNORECASE)),
    ("abs-path", re.compile(r"/home/(?:intel|claude)\b")),
    ("device-path", re.compile(r"/dev/(?:video\d+|ttyACM\d*|serial/\S+)")),
    ("net-address", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")),
    ("secret", re.compile(r"\b(?:hf_|ghp_|ak-|sk-|AKIA)[A-Za-z0-9]{8,}")),
    ("secret-word", re.compile(r"\b(?:token|password|passwd|api[_-]?key|secret|credentials?)\b",
                               re.IGNORECASE)),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("todo", re.compile(r"\b(?:TODO|FIXME|XXX|HACK)\b")),
]
PATH_CATS = ("ai-word", "decision-id", "person", "abs-path")
PATH_PATS = [(cat, rx) for cat, rx in PATS if cat in PATH_CATS]
GIT_CATS = ("ai-word", "decision-id", "clock-stamp", "person", "abs-path")
GIT_PATS = [(cat, rx) for cat, rx in PATS if cat in GIT_CATS] + [
    ("coauthor", re.compile(r"co-authored-by", re.IGNORECASE)),
    ("generated", re.compile(r"generated with", re.IGNORECASE)),
]
ACTIONS = {
    "ai-word": "rewrite: name the component or the measured fact (manual)",
    "decision-id": "replace the id with the reason it stands for, or delete (manual)",
    "worker-id": "delete",
    "clock-stamp": "delete the stamp, keep the event",
    "person": "delete",
    "person-context": "rewrite without the reference to another person's data (manual)",
    "abs-path": "paths.py constant (R, RUN_ROOT, LOG_ROOT, SIM_ROOT) or a relative path",
    "device-path": "named setting outside the repository (manual)",
    "net-address": "config value or template (manual)",
    "secret": "remove the value and read it from the environment (manual)",
    "secret-word": "check: rename, or remove if it holds a value (manual)",
    "private-key": "remove; keys are never committed",
    "todo": "resolve or delete (manual)",
    "coauthor": "history: rewrite the commit message (decide with the reviewer)",
    "generated": "history: rewrite the commit message (decide with the reviewer)",
}
FINAL = {"ROOT": "IN", "REACHED": "IN", "REVIEW": "REVIEW", "EXCLUDE": "EXCLUDE", "OUT": "OUT"}


def rel(path):
    return os.path.relpath(path, R)


def git(*args):
    proc = subprocess.run(["git", "-C", APP, "-c", "core.quotepath=off"] + list(args),
                          capture_output=True, text=True, timeout=300, check=False)
    return proc.stdout


def git_state():
    head = git("rev-parse", "--short", "HEAD").strip() or "none"
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip() or "none"
    lines = [line for line in git("status", "--porcelain").splitlines() if line]
    untracked = sum(1 for line in lines if line.startswith("??"))
    return head, branch, len(lines) - untracked, untracked


def safe_read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def clean(value, limit=None):
    text = re.sub(r"[\t\r\n]+", " ", str(value)).strip()
    return text[:limit] if limit else text


def is_text(path):
    name = os.path.basename(path)
    return os.path.splitext(name)[1].lower() in TEXT_EXT or name in TEXT_NAMES


def add_record(recs, export, src, origin, forced=None):
    recs[export] = {"src": src, "origin": origin, "mark": "", "forced": forced,
                    "decision": "", "reason": "", "bytes": -1, "lines": -1,
                    "exists": False, "text": False, "final": ""}


def find_files(root, name, fragment):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not SKIP_WALK.search(d)]
        if name in filenames and fragment in dirpath:
            hits.append(os.path.join(dirpath, name))
    return sorted(hits)


def collect():
    recs, located = {}, []
    status = {}
    for line in git("status", "--porcelain").splitlines():
        if len(line) > 3:
            status[line[3:].split(" -> ")[-1]] = line[:2]
    tracked = set(filter(None, git("ls-files").splitlines()))
    untracked = set(filter(None, git("ls-files", "--others", "--exclude-standard").splitlines()))
    skipped = 0
    for path in sorted(tracked | untracked):
        if any(part in SKIP_DIRS for part in path.split("/")):
            skipped += 1
            continue
        if path in untracked:
            mark = "untracked"
        elif status.get(path, "").strip():
            mark = "modified (" + status[path].strip() + ")"
        else:
            mark = "tracked, clean"
        add_record(recs, path, os.path.join(APP, path), "R/app")
        recs[path]["mark"] = mark
    for src_rel, export, forced, why in EXTRA:
        if export in recs:
            continue
        add_record(recs, export, os.path.join(R, src_rel), "R/" + os.path.dirname(src_rel),
                   (forced, why) if forced else None)
        recs[export]["mark"] = "external evidence"
    for export, root_rel, name, fragment in LOCATE:
        if export in recs:
            continue
        hits = find_files(os.path.join(R, root_rel), name, fragment)
        if not hits:
            located.append("%s: no %s under R/%s" % (export, name, root_rel))
            continue
        add_record(recs, export, hits[0], "R/" + os.path.dirname(rel(hits[0])))
        recs[export]["mark"] = "external evidence"
        if len(hits) > 1:
            located.append("%s: %d copies, first used: %s"
                           % (export, len(hits), "; ".join(rel(h) for h in hits)))
    return recs, skipped, located


def policy(path):
    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()
    parts = path.split("/")
    if path == ".gitignore":
        return "EXCLUDE", "replaced by the export .gitignore (gitignore.txt)"
    if name.endswith((".orig", ".bak", ".old", "~")):
        return "EXCLUDE", "backup copy: not exported"
    if ext in ARTEFACT_EXT:
        return "EXCLUDE", "binary or media artefact: kept out of the repository, cited by its log"
    if path.startswith("anomaly/anom3/models/"):
        return "EXCLUDE", "model files (listed in the existing .gitignore)"
    if path.startswith("results/cycles/"):
        if name == "any_2127.jsonl":
            return "REVIEW", "D-121 records this run starting 21:27Z with people in the cell: check contents"
        return "REVIEW", "run log: check fields and paths"
    if "fixtures" in parts[:-1] and ext in IMAGE_EXT:
        return "REVIEW", "fixture image: check for people or hands in view before publishing"
    if path in REVIEW_FILES:
        return "REVIEW", REVIEW_FILES[path]
    if path in ROOT_FILES:
        return "ROOT", "root: " + ROOT_FILES[path]
    for prefix, why in ROOT_PREFIXES:
        if path.startswith(prefix):
            return "ROOT", "root: " + why
    if "lint" in parts[:-1]:
        return "ROOT", "root: static check (lint)"
    return "PENDING", ""


def classify(recs):
    for export, rec in recs.items():
        if os.path.islink(rec["src"]):
            rec["decision"], rec["reason"] = "EXCLUDE", "symbolic link: not followed"
            continue
        if rec["forced"]:
            rec["decision"], rec["reason"] = rec["forced"]
        else:
            rec["decision"], rec["reason"] = policy(export)
        rec["exists"] = os.path.isfile(rec["src"])
        if not rec["exists"]:
            rec["reason"] = (rec["reason"] + "; " if rec["reason"] else "") + "source not found on the laptop"
            continue
        rec["bytes"] = os.path.getsize(rec["src"])
        rec["text"] = is_text(export) and rec["bytes"] < COUNT_LIMIT
        if rec["text"] and not export.startswith("laptop-only/"):
            with open(rec["src"], "rb") as handle:
                rec["lines"] = handle.read().count(b"\n")


def build_index(paths):
    index = collections.defaultdict(list)
    for path in paths:
        if not path.endswith(".py"):
            continue
        parts = path[:-3].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        for start in range(len(parts)):
            index[".".join(parts[start:])].append(path)
    return index


def relative_targets(path, module, level, all_set):
    parts = path.split("/")[:-1]
    if level > 1:
        parts = parts[:max(0, len(parts) - (level - 1))]
    stem = "/".join(parts + module.split(".")) if module else "/".join(parts)
    return [cand for cand in (stem + ".py", stem + "/__init__.py") if cand in all_set]


def package_inits(target, all_set):
    parts = target.split("/")[:-1]
    found = []
    for depth in range(1, len(parts) + 1):
        init = "/".join(parts[:depth]) + "/__init__.py"
        if init in all_set:
            found.append(init)
    return found


def normalise(value):
    marker = "intel-junction-2026/app/"
    if marker in value:
        return value.split(marker, 1)[1]
    return value.lstrip("/")


def choose(cands, importer, text):
    """Pick one candidate: same folder first, then the folders named on sys.path lines."""
    cands = sorted(set(cands))
    if len(cands) == 1:
        return cands[0], ""
    words = set()
    for line in text.splitlines():
        if "sys.path" in line:
            words.update(re.findall(r"[A-Za-z0-9_\-]+", line))
    importer_dir = importer.rsplit("/", 1)[0] if "/" in importer else ""

    def score(candidate):
        folder = candidate.rsplit("/", 1)[0] if "/" in candidate else ""
        same_folder = 1 if folder == importer_dir else 0
        on_path = sum(1 for part in folder.split("/") if part and part in words)
        return same_folder, on_path

    scores = {cand: score(cand) for cand in cands}
    best = max(scores.values())
    winners = [cand for cand in cands if scores[cand] == best]
    if len(winners) == 1 and best != (0, 0):
        return winners[0], "resolved by folder or sys.path"
    return None, "ambiguous: " + " | ".join(cands)


def dependencies(path, recs, index, all_paths, all_set):
    """Return (edges, ambiguous, external module names, unresolved literals) for one file."""
    rec = recs[path]
    edges, ambiguous, external, unresolved = [], [], set(), []
    if not rec["exists"]:
        return edges, ambiguous, external, unresolved
    text = safe_read(rec["src"])
    base_mods, alias_mods, strings = set(), set(), set()
    if path.endswith(".py"):
        try:
            tree = ast.parse(text, filename=path)
        except (SyntaxError, ValueError) as exc:
            ambiguous.append("%s: cannot parse (%s)" % (path, clean(exc, 120)))
            return edges, ambiguous, external, unresolved
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    docstrings.add(id(first.value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                base_mods.update((alias.name, 0) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                level = node.level or 0
                if base:
                    base_mods.add((base, level))
                alias_mods.update(((base + "." if base else "") + alias.name, level)
                                  for alias in node.names)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) not in docstrings:
                    strings.add(node.value)
            elif type(node).__name__ == "Str" and id(node) not in docstrings:
                strings.add(node.s)
    else:
        strings.update(TOKEN_RE.findall(text))

    def add_module(module, level, report_external):
        if level == 0:
            if module.split(".")[0] in STDLIB:
                return
            cands = list(index.get(module, []))
            if not cands:
                if report_external:
                    external.add(module.split(".")[0])
                return
        else:
            cands = relative_targets(path, module, level, all_set)
            if not cands:
                return
        target, note = choose(cands, path, text)
        if target is None:
            ambiguous.append("%s: import %s -> %s" % (path, module, note))
            return
        edges.append((target, "import " + module))
        edges.extend((init, "package of " + target) for init in package_inits(target, all_set))

    for module, level in sorted(base_mods):
        add_module(module, level, True)
    for module, level in sorted(alias_mods):
        add_module(module, level, False)
    for value in sorted(strings):
        if not LIT_RE.match(value):
            continue
        norm = normalise(value)
        cands = [p for p in all_paths if p == norm or p.endswith("/" + norm)]
        if not cands:
            unresolved.append("%s: %s" % (path, value))
            continue
        target, note = choose(cands, path, text)
        if target is None:
            ambiguous.append("%s: literal %s -> %s" % (path, value, note))
            continue
        edges.append((target, "reads " + value))
    return edges, ambiguous, external, unresolved


def close_over(recs, index, all_paths, all_set):
    importers = collections.defaultdict(list)
    externals = collections.defaultdict(list)
    conflicts, ambiguous, unresolved = [], [], []
    queue = [p for p in all_paths
             if recs[p]["decision"] in ("ROOT", "REVIEW") and os.path.splitext(p)[1] in CODE_EXT]
    done = set()
    while queue:
        path = queue.pop()
        if path in done:
            continue
        done.add(path)
        edges, amb, ext, unres = dependencies(path, recs, index, all_paths, all_set)
        ambiguous.extend(amb)
        unresolved.extend(unres)
        for name in sorted(ext):
            externals[name].append(path)
        for target, how in edges:
            importers[target].append("%s [%s]" % (path, how))
            rec = recs[target]
            if rec["decision"] == "EXCLUDE":
                conflicts.append("%s is reached from %s [%s] but is EXCLUDE" % (target, path, how))
                continue
            if rec["decision"] == "PENDING":
                rec["decision"] = "REACHED"
                rec["reason"] = "reached: import or path literal from an IN file"
            if os.path.splitext(target)[1] in CODE_EXT and target not in done:
                queue.append(target)
    for path in all_paths:
        rec = recs[path]
        if rec["decision"] == "PENDING":
            rec["decision"] = "OUT"
            rec["reason"] = "not reached from the roots (entry points, tests, lint, bench, docs)"
    return importers, externals, conflicts, ambiguous, unresolved


def scrub(recs):
    rows, counts, per_file = [], collections.Counter(), collections.Counter()
    for path in sorted(recs):
        rec = recs[path]
        if rec["final"] not in ("IN", "REVIEW"):
            continue
        for cat, rx in PATH_PATS:
            for match in rx.finditer(path):
                key = "path:" + cat
                rows.append((path, rec["final"], 0, key, match.group(0), path,
                             ACTIONS[cat] + " (or rename the file)"))
                counts[key] += 1
                per_file[path] += 1
        if not rec["text"]:
            continue
        for number, line in enumerate(safe_read(rec["src"]).splitlines(), 1):
            seen = set()
            for cat, rx in PATS:
                for match in rx.finditer(line):
                    if (cat, match.group(0)) in seen:
                        continue
                    seen.add((cat, match.group(0)))
                    rows.append((path, rec["final"], number, cat, match.group(0), clean(line, 160),
                                 ACTIONS[cat]))
                    counts[cat] += 1
                    per_file[path] += 1
    return rows, counts, per_file


def git_history():
    rows, counts, flagged, commits = [], collections.Counter(), [], 0
    raw = git("log", "--all", "--format=%H%x1f%an%x1f%ae%x1f%B%x1e")
    for record in raw.split("\x1e"):
        fields = record.strip("\n").split("\x1f")
        if len(fields) < 4:
            continue
        commits += 1
        sha = "git:" + fields[0][:9]
        body = "\x1f".join(fields[3:]).strip("\n")
        lines = body.splitlines()
        subject = lines[0] if lines else ""
        author = fields[1] + " <" + fields[2] + ">"
        flag = False
        for label, text in (("author", author), ("message", body)):
            for cat, rx in GIT_PATS:
                for match in rx.finditer(text):
                    line = next((ln for ln in text.splitlines() if match.group(0) in ln), subject)
                    key = "git-%s-%s" % (label, cat)
                    rows.append((sha, "history", 0, key, match.group(0), clean(line, 160), ACTIONS[cat]))
                    counts[key] += 1
                    flag = flag or cat in ("ai-word", "coauthor", "generated")
        if flag:
            flagged.append("%s %s" % (sha, clean(subject, 120)))
    return rows, counts, commits, flagged


def figure_search(roots):
    hits = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not SKIP_WALK.search(d)]
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                if os.path.splitext(name)[1].lower() not in SEARCH_EXT or os.path.islink(path):
                    continue
                if os.path.getsize(path) > SEARCH_LIMIT:
                    continue
                for number, line in enumerate(safe_read(path).splitlines(), 1):
                    if FIGURE_RE.search(line):
                        hits.append((rel(path), number, clean(line, 160)))
    hits.sort(key=lambda hit: ("0.966" not in hit[2], hit[0], hit[1]))
    return hits


def doc_refs_missing(docs_text, all_set):
    basenames = set(os.path.basename(p) for p in all_set)
    missing = []
    for doc in sorted(docs_text):
        for ref in sorted(set(DOC_REF_RE.findall(docs_text[doc]))):
            if ref not in basenames:
                missing.append("%s -> %s" % (doc, ref))
    return missing


def write_tsv(name, header, rows):
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(clean(value) for value in row) + "\n")


def write_filelist(recs, importers, docs_text, per_file):
    header = ["kind", "path", "decision", "bytes", "lines", "git", "scrub_hits",
              "used_by", "doc_refs", "source", "reason"]
    folders = collections.OrderedDict()
    for path in sorted(recs):
        top = path.split("/")[0] if "/" in path else "(top level)"
        bucket = folders.setdefault(top, collections.Counter())
        bucket[recs[path]["final"]] += 1
        if recs[path]["final"] == "IN":
            bucket["bytes"] += max(recs[path]["bytes"], 0)
        bucket["hits"] += per_file.get(path, 0)
    rows = []
    for top, bucket in folders.items():
        rows.append(["folder", top, "", bucket["bytes"], "", "", bucket["hits"], "", "", "",
                     "IN %d, REVIEW %d, EXCLUDE %d, OUT %d; bytes column is IN bytes"
                     % (bucket["IN"], bucket["REVIEW"], bucket["EXCLUDE"], bucket["OUT"])])
    for path in sorted(recs):
        rec = recs[path]
        users = importers.get(path, [])
        used_by = "; ".join(users[:3]) + (" (+%d more)" % (len(users) - 3) if len(users) > 3 else "")
        base = os.path.basename(path)
        docs = [name for name in sorted(docs_text)
                if name != path and re.search(r"\b" + re.escape(base) + r"\b", docs_text[name])]
        rows.append(["file", path, rec["final"],
                     rec["bytes"] if rec["bytes"] >= 0 else "",
                     rec["lines"] if rec["lines"] >= 0 else "",
                     rec["mark"], per_file.get(path, 0), used_by, "; ".join(docs[:3]),
                     rec["origin"], rec["reason"]])
    write_tsv("filelist.tsv", header, rows)


def build_summary(recs, externals, conflicts, ambiguous, unresolved, located, skipped,
                  scrub_counts, per_file, doc_missing, figures, git_info, commits, flagged):
    head, branch, modified, untracked = git_info
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    final = collections.Counter(rec["final"] for rec in recs.values())
    sizes = collections.Counter()
    for rec in recs.values():
        sizes[rec["final"]] += max(rec["bytes"], 0)
    lines = [
        "scan %s; python %s; R/app HEAD %s on %s; modified %d; untracked %d; listing skipped by name: %d"
        % (stamp, sys.version.split()[0], head, branch, modified, untracked, skipped),
        "decisions: " + "; ".join("%s %d files, %d bytes" % (k, final[k], sizes[k])
                                  for k in ("IN", "REVIEW", "EXCLUDE", "OUT")),
        "",
        "== conflicts: EXCLUDE files reached from IN code (resolve before export)",
    ]
    lines += conflicts or ["none"]
    lines += ["", "== ambiguous imports or literals (first 40)"]
    lines += ambiguous[:40] or ["none"]
    lines += ["", "== unresolved path-like strings in code (first 60; docstring text or run outputs)"]
    lines += unresolved[:60] or ["none"]
    lines += ["", "== modules not found in the tree: name, number of code files importing it"]
    for name in sorted(externals, key=lambda n: (-len(externals[n]), n)):
        lines.append("%s %d, e.g. %s" % (name, len(externals[name]), externals[name][0]))
    lines += ["", "== REVIEW files (data and secrets list)"]
    lines += ["%s: %s" % (p, recs[p]["reason"]) for p in sorted(recs) if recs[p]["final"] == "REVIEW"] or ["none"]
    lines += ["", "== EXCLUDE files (first 80)"]
    lines += ["%s: %s" % (p, recs[p]["reason"]) for p in sorted(recs)
              if recs[p]["final"] == "EXCLUDE"][:80] or ["none"]
    lines += ["", "== missing sources (exported path: reason)"]
    lines += ["%s: %s" % (p, recs[p]["reason"]) for p in sorted(recs) if not recs[p]["exists"]] or ["none"]
    lines += ["", "== located or lookup notes"]
    lines += located or ["none"]
    lines += ["", "== doc references without a matching file name in the candidate set (first 80)"]
    lines += doc_missing[:80] or ["none"]
    lines += ["", "== git history: commits %d; commits with an AI or co-author match: %d"
              % (commits, len(flagged))]
    lines += flagged[:20] or ["none"]
    lines += ["", "== scrub categories (rows)"]
    lines += ["%6d %s" % (n, cat) for cat, n in scrub_counts.most_common()] or ["none"]
    lines += ["", "== scrub hits per file (top 25)"]
    lines += ["%4d %s" % (n, p) for p, n in per_file.most_common(25)] or ["none"]
    lines += ["", "== coefficient search (0.966, LOO, leave-one-out, AUROC): %d hits, first 60" % len(figures)]
    lines += ["%s:%d :: %s" % hit for hit in figures[:60]] or ["none"]
    return lines


def main():
    recs, skipped, located = collect()
    classify(recs)
    all_paths = sorted(recs)
    all_set = set(all_paths)
    index = build_index(all_paths)
    importers, externals, conflicts, ambiguous, unresolved = close_over(recs, index, all_paths, all_set)
    for path in all_paths:
        recs[path]["final"] = FINAL[recs[path]["decision"]]
    scrub_rows, scrub_counts, per_file = scrub(recs)
    history_rows, history_counts, commits, flagged = git_history()
    scrub_rows.extend(history_rows)
    scrub_counts.update(history_counts)
    git_info = git_state()
    docs_text = {path: safe_read(recs[path]["src"]) for path in all_paths
                 if path.endswith(".md") and recs[path]["final"] in ("IN", "REVIEW") and recs[path]["exists"]}
    doc_missing = doc_refs_missing(docs_text, all_set)
    figures = figure_search([os.path.join(R, "work"), APP, os.path.join(R, "docs")])
    write_filelist(recs, importers, docs_text, per_file)
    write_tsv("scrub_hits.tsv", ["file", "decision", "line", "category", "match", "excerpt", "proposal"],
              scrub_rows)
    summary = build_summary(recs, externals, conflicts, ambiguous, unresolved, located, skipped,
                            scrub_counts, per_file, doc_missing, figures, git_info, commits, flagged)
    with open(os.path.join(HERE, "scan_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary) + "\n")
    final = collections.Counter(recs[path]["final"] for path in all_paths)
    print("scan: candidates %d; IN %d, REVIEW %d, EXCLUDE %d, OUT %d; scrub rows %d; "
          "conflicts %d; ambiguous %d; LOO/AUROC search hits %d"
          % (len(all_paths), final["IN"], final["REVIEW"], final["EXCLUDE"], final["OUT"],
             len(scrub_rows), len(conflicts), len(ambiguous), len(figures)))


if __name__ == "__main__":
    main()
