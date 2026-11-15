"""Read-only checks for round 3: RAM, CPU count, and whether WinCLIP / open_clip weights are already cached.

Nothing is downloaded and no package is imported. Files are only looked up by name.
"""
import glob
import os

HOME = os.path.expanduser("~")
PROJ = "/home/intel/intel-junction-2026"
print("home", HOME)
print("cpu_count", os.cpu_count())
with open("/proc/meminfo", encoding="utf-8") as fh:
    for line in fh:
        if line.startswith(("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")):
            print(line.strip())

ROOTS = [os.path.join(HOME, ".cache"), os.path.join(PROJ, ".cache")]
KEYS = ("open_clip", "clip_240", "laion", "b-16-plus", "b_16_plus", "winclip", "clip-vit", "clip_vit",
        "vit_base_patch16")
hits = []
for root in ROOTS:
    if not os.path.isdir(root):
        print("missing cache root", root)
        continue
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dp, dn, fn in os.walk(root):
        if dp.count(os.sep) - base_depth >= 8:
            dn[:] = []
        for name in dn + fn:
            low = name.lower()
            if any(k in low for k in KEYS):
                p = os.path.join(dp, name)
                size = os.path.getsize(p) if os.path.isfile(p) else -1
                hits.append((p, size))
print("cache hits for WinCLIP / open_clip names:", len(hits))
for p, size in sorted(hits):
    print("  %s %d" % (p, size))

for root in ROOTS + [os.path.join(HOME, ".cache", "huggingface"), os.path.join(PROJ, ".cache", "torch")]:
    if os.path.isdir(root):
        print("top level of", root, sorted(os.listdir(root))[:40])

SITE = sorted(glob.glob(os.path.join(PROJ, "envs", "anomalib-env", "lib", "python*", "site-packages")))
print("site-packages", SITE)
for s in SITE:
    print("open_clip package present:", os.path.isdir(os.path.join(s, "open_clip")))
    wc = os.path.join(s, "anomalib", "models", "image", "winclip")
    print("anomalib winclip folder present:", os.path.isdir(wc),
          sorted(os.listdir(wc)) if os.path.isdir(wc) else "")
