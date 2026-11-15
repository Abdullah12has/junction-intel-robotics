"""Remove the hard-coded workspace path from app/ and scrub assistant-name mentions from comments.
- tools/paths.py derives the roots from its own location (JUNCTION_ROOT overrides): R = workspace, APP = repository.
- Every `R = "<abs>"` becomes `from paths import APP, R` (after making sure the file's own folder is on sys.path,
  which -I mode does not do), every f"{R}/app/..." becomes f"{APP}/...", absolute sys.path inserts become relative.
- Shell scripts derive R from their own location.
Writes backups to run/bak_paths_<file>; prints every remaining absolute path and every assistant-name mention."""
import os
import py_compile
import re
import shutil
import sys

RL = "/home/intel/intel-junction-2026"
APPD = f"{RL}/app"
BAK = f"{RL}/run/bak_paths"
os.makedirs(BAK, exist_ok=True)

PATHS_PY = '''"""Run-time roots of the sorter, derived from where this file is (no fixed install path).
APP is the repository (this file is APP/tools/paths.py); R is the workspace that holds app/, envs/, run/ and logs/
(JUNCTION_ROOT overrides it). A simulated cell sets JUNCTION_RUN_ROOT."""
import os

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("JUNCTION_ROOT") or os.path.dirname(APP)
TOOLS = os.path.join(APP, "tools")
SIM_ROOT = os.environ.get("JUNCTION_RUN_ROOT") or None
RUN_ROOT = SIM_ROOT or os.path.join(R, "run")  # live frames, flags, motion server state, heartbeat
LOG_ROOT = os.path.join(SIM_ROOT, "logs") if SIM_ROOT else os.path.join(R, "logs")  # episodes, monitor logs, e-stop images
'''

SCRUB = [  # exact text -> replacement (comments, docstrings and log text only)
    ('"""Opus I-59 automatic spot-cell grasp:', '"""Automatic spot-cell grasp:'),
    ('"""Opus hand watchdog (I-58; rules shared with the sort loop since 18:43Z).', '"""Hand watchdog (rules shared with the sort loop through handdet.py).'),
    ("Opus visual labels in", "manual visual labels in"),
    ('"""Opus I-60 kinematics helpers', '"""Kinematics helpers'),
    ('"""Opus I-59 precise motion server', '"""Precise motion server'),
    ('"""Opus I-60 pick-anywhere:', '"""Pick-anywhere:'),
    ("Hassan 17:24Z: touching the mat loosens the fixed jaw", "touching the mat loosens the fixed jaw"),
    ('"""Opus I-59: height above the mat', '"""Height above the mat'),
    ('"""Opus I-59 depth grab', '"""Depth grab'),
    ('"""Opus I-60 autonomous sort loop, PICK ANYWHERE', '"""Autonomous sort loop, PICK ANYWHERE'),
    ('"""Opus I-60 autonomous sort loop, spot cell', '"""Autonomous sort loop, spot cell'),
    ("Hassan puts a brick under the jaws", "The operator puts a brick under the jaws"),
    ('waiting for Opus")', 'waiting for the operator")'),
    ('not supplied yet (Opus)")', 'not supplied (local labelled data)")'),
    ("work/opus-verify/reach_map.log", "the reach map log"),
    ("(fixtures/hand_labels.csv, supplied by Opus)", "(fixtures/hand_labels.csv, manual labels)"),
    ("# Continuous wrist-camera recorder (Opus 14:33Z; v2 I-59 17:05Z):", "# Continuous wrist-camera recorder:"),
    ("# Opus I-61: bring the pick-anywhere stack back up", "# Bring the pick-anywhere stack back up"),
    ("# Run ONLY when Hassan has said the robot is ours again.", "# Run only when the robot is free for this stack."),
    ('"""Tiny stdlib-only static check (Opus I-61):', '"""Tiny stdlib-only static check:'),
    ('"""Cross-module attribute check (Opus I-61, stdlib only):', '"""Cross-module attribute check (stdlib only):'),
    ("(no Opus ground truth yet)", "(no manual ground truth yet)"),
]

SH_R = 'R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # workspace root, from this script\'s location'


def patch_py(path):
    s0 = s = open(path).read()
    rel = os.path.relpath(path, APPD)
    for a, b in SCRUB:
        s = s.replace(a, b)
    s = s.replace(f'sys.path.insert(0, "{APPD}/tools")', "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))")
    s = s.replace(f'sys.path.insert(0, "{APPD}/motion")',
                  'sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "motion"))')
    if rel.startswith("tools/") and rel.count("/") == 1 and rel != "tools/paths.py":
        lines = s.split("\n")
        idx = [i for i, ln in enumerate(lines) if ln.strip() == f'R = "{RL}"']
        if idx:
            i = idx[0]
            head = "\n".join(lines[:i])
            add = []
            if not re.search(r"^import (.*\b)?sys\b", head, re.M):
                add.append("import sys  # noqa: E402")
            if not re.search(r"^import (.*\b)?os\b", head, re.M):
                add.append("import os  # noqa: E402")
            if "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))" not in head:
                add.append("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))")
            add.append("from paths import APP, R  # noqa: E402  (roots from this file's location)")
            lines[i:i + 1] = add
            s = "\n".join(lines)
            s = s.replace("{R}/app/", "{APP}/")
        s = s.replace(f'open("{APPD}/tools/jaw_calib.json")', 'open(os.path.join(APP, "tools", "jaw_calib.json"))')
        s = s.replace(f'os.environ.get("JUNCTION_RUN_ROOT") or "{RL}/run"',
                      'os.environ.get("JUNCTION_RUN_ROOT") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "run")')
    if s != s0:
        shutil.copy(path, os.path.join(BAK, rel.replace("/", "__")))
        open(path + ".tmp", "w").write(s)
        py_compile.compile(path + ".tmp", doraise=True)
        os.replace(path + ".tmp", path)
        print("patched", rel)


def patch_sh(path):
    s0 = s = open(path).read()
    rel = os.path.relpath(path, APPD)
    for a, b in SCRUB:
        s = s.replace(a, b)
    s = re.sub(rf"^R={re.escape(RL)}\s*$", SH_R, s, flags=re.M)
    s = s.replace(f"bash {APPD}/tools/", "bash app/tools/")
    s = s.replace(f"kill $(cat {RL}/run/", "kill $(cat run/")
    if s != s0:
        shutil.copy(path, os.path.join(BAK, rel.replace("/", "__")))
        open(path, "w").write(s)
        print("patched", rel)


open(f"{APPD}/tools/paths.py.tmp", "w").write(PATHS_PY)
py_compile.compile(f"{APPD}/tools/paths.py.tmp", doraise=True)
shutil.copy(f"{APPD}/tools/paths.py", os.path.join(BAK, "tools__paths.py"))
os.replace(f"{APPD}/tools/paths.py.tmp", f"{APPD}/tools/paths.py")
print("rewrote tools/paths.py")
for root, dirs, files in os.walk(APPD):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "fixtures")]
    for f in files:
        p = os.path.join(root, f)
        if f.endswith(".py"):
            patch_py(p)
        elif f.endswith(".sh"):
            patch_sh(p)
print("== remaining absolute workspace paths")
ai = re.compile(r"\b(opus|claude|anthropic|haiku)\b|opus[-_]|_opus|Hassan|hsn07pk", re.I)
for root, dirs, files in os.walk(APPD):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "fixtures")]
    for f in files:
        if f.endswith((".py", ".sh", ".md", ".yaml", ".json", ".txt", ".tsv")):
            p = os.path.join(root, f)
            for i, ln in enumerate(open(p, errors="replace"), 1):
                if RL in ln or "/home/intel" in ln:
                    print(f"ABS {os.path.relpath(p, APPD)}:{i}: {ln.strip()[:140]}")
                if ai.search(ln):
                    print(f"AI  {os.path.relpath(p, APPD)}:{i}: {ln.strip()[:140]}")
