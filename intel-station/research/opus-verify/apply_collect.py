"""Calibration runs: sort_any.py --collect GOOD|DEFECT records the truth of every brick (all bricks on the mat are of
that kind) in episode.json for fit_faces.py; start_all.sh passes COLLECT from the environment."""
import os
import py_compile
import shutil
import sys
import time

T = "/home/intel/intel-junction-2026/app/tools"
B = "/home/intel/intel-junction-2026/run"
p = f"{T}/sort_any.py"
s = open(p).read()
E = [
    ('    ap.add_argument("--profile", choices=("safe", "fast"), default="safe",',
     '    ap.add_argument("--collect", choices=("GOOD", "DEFECT"), default=None,\n'
     '                    help="calibration run: every brick on the mat is of this kind (truth in episode.json for fit_faces.py)")\n'
     '    ap.add_argument("--profile", choices=("safe", "fast"), default="safe",'),
    ('    CFG["profile"] = a.profile\n', '    CFG["profile"] = a.profile\n    CFG["collect"] = a.collect\n'),
    ('    EP = {"episode": k, "run": os.path.basename(RUN), "mode": "pick_anywhere"}\n',
     '    EP = {"episode": k, "run": os.path.basename(RUN), "mode": "pick_anywhere", "truth": CFG.get("collect")}\n'),
]
for a, b in E:
    if s.count(a) != 1:
        sys.exit(f"ABORT anchor x{s.count(a)}: {a[:70]!r}")
    s = s.replace(a, b)
shutil.copy(p, f"{B}/bak_collect_sort_any_{time.strftime('%H%M', time.gmtime())}.py")
open(p + ".tmp", "w").write(s)
py_compile.compile(p + ".tmp", doraise=True)
os.replace(p + ".tmp", p)
print("patched sort_any.py")
sh = f"{T}/start_all.sh"
t = open(sh).read()
a = '--profile "$PROFILE" \\'
if t.count(a) == 1 and "COLLECT" not in t:
    shutil.copy(sh, f"{B}/bak_collect_start_all_{time.strftime('%H%M', time.gmtime())}.sh")
    open(sh, "w").write(t.replace(a, '--profile "$PROFILE" ${COLLECT:+--collect "$COLLECT"} \\'))
    print("patched start_all.sh")
else:
    print("start_all.sh anchor count", t.count(a))
