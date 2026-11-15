"""Make app/ itself match the public export: apply the export's text substitutions in place to every file the export
copies (comments, docstrings and log text only), so patches made against the public repository apply to app/."""
import importlib.util
import os
import sys

R = "/home/intel/intel-junction-2026"
spec = importlib.util.spec_from_file_location("exp", f"{R}/work/opus-verify/export.py")
src = open(f"{R}/work/opus-verify/export.py").read()
head = src.split("files = set()")[0]  # constants only: APP/OUT parsing needs argv
sys.argv = ["export.py", f"{R}/app", f"{R}/export/junction-2026"]
ns = {}
exec(compile(src.split("if os.path.isdir(OUT):")[0], "export_head", "exec"), ns)
files, SUBS, CAP, TEXT = ns["files"], ns["SUBS"], ns["CAP"], ns["TEXT"]
n = 0
for f in files:
    if not f.endswith(TEXT) or f == ".gitignore.public":
        continue
    p = os.path.join(R, "app", f)
    s0 = s = open(p).read()
    for rx, rep in SUBS:
        s = rx.sub(rep, s)
    s = CAP.sub(lambda m: '"""' + m.group(1).upper(), s)
    if s != s0:
        open(p, "w").write(s)
        n += 1
        print("scrubbed", f)
print("files changed", n)
