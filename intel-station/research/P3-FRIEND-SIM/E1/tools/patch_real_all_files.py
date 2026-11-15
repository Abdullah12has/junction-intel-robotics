"""Extend the real image pass in tools/sim_real_compare.py from chunk-000/file-000.mp4 only
(1253 of 10158 frames, about 42 s) to all four chunk-000 files per camera (file-000..003 =
1253 + 3113 + 2327 + 3465 = 10158 frames, the full 16-episode Studio dataset).

Edits only tools/sim_real_compare.py (a backup of the v1 script goes to logs/). Each old
string must occur exactly once or nothing is written.
"""
import shutil

P = "/home/intel/intel-junction-2026/work/P3-FRIEND-SIM/E1/tools/sim_real_compare.py"
BAK = "/home/intel/intel-junction-2026/work/P3-FRIEND-SIM/E1/logs/sim_real_compare.v1-file000-only.py.bak"

with open(P, encoding="utf-8") as fh:
    src = fh.read()

reps = [
    ('REAL_TOP = REAL + "/videos/observation.images.realsense 243622061421/chunk-000/file-000.mp4"\n',
     'REAL_TOP_FILES = [REAL + "/videos/observation.images.realsense 243622061421/chunk-000/file-00%d.mp4" % k '
     'for k in range(4)]\nREAL_TOP = REAL_TOP_FILES[0]\n'),
    ('REAL_CLOSE = REAL + "/videos/observation.images.realsense 243722064316/chunk-000/file-000.mp4"\n',
     'REAL_CLOSE_FILES = [REAL + "/videos/observation.images.realsense 243722064316/chunk-000/file-00%d.mp4" % k '
     'for k in range(4)]\nREAL_CLOSE = REAL_CLOSE_FILES[0]\n'),
    ('run("real_top", REAL_TOP, True)\nrun("real_close", REAL_CLOSE, True)\n',
     'PERFILE = []\n'
     'for key, paths in [("real_top", REAL_TOP_FILES), ("real_close", REAL_CLOSE_FILES)]:\n'
     '    for i, p in enumerate(paths):\n'
     '        n0 = len(agg[key]["stats"])\n'
     '        run(key, p, i == 0)\n'
     '        seg = agg[key]["stats"][n0:]\n'
     '        PERFILE.append({"view": key, "file": p.rsplit("/", 1)[-1], "frames": len(seg),\n'
     '                        "luma_mean": round(float(np.mean([x["luma"] for x in seg])), 2)})\n'),
    ('"brightness": bright,', '"brightness": bright, "real_per_file": PERFILE,'),
]

for old, _new in reps:
    n = src.count(old)
    assert n == 1, ("expected exactly one match", n, old[:70])

out = src
for old, new in reps:
    out = out.replace(old, new)

shutil.copy(P, BAK)
with open(P, "w", encoding="utf-8") as fh:
    fh.write(out)
print("patched", P)
print("backup", BAK)
