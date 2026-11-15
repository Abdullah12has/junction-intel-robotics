"""Fixture handling for a checkout without the recorded frames: skip, never pass silently; a missing frame in a
checkout that has frames stays an error. Adds README.md and .gitignore to the export set."""
import os
import py_compile
import shutil
import sys

A = "/home/intel/intel-junction-2026/app"
T = f"{A}/tools/tests"
BAK = "/home/intel/intel-junction-2026/run/bak_paths"


def edit(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit(f"ABORT {path}: anchor x{s.count(old)}: {old[:70]!r}")
        s = s.replace(old, new)
    shutil.copy(path, os.path.join(BAK, os.path.basename(path) + ".p3"))
    open(path + ".tmp", "w").write(s)
    if path.endswith(".py"):
        py_compile.compile(path + ".tmp", doraise=True)
    os.replace(path + ".tmp", path)
    print("patched", path)


edit(f"{T}/conftest.py", [(
    '''def load_img(rel):
    img = cv2.imread(os.path.join(FIX, rel))
    if img is None:
        raise Skip(f"fixture image {rel} is local data, not present")
        raise FileNotFoundError(rel)
    return img
''', '''def _has_images(root):
    for _, _, names in os.walk(root):
        if any(n.lower().endswith((".png", ".jpg", ".jpeg")) for n in names):
            return True
    return False


IMAGES_PRESENT = _has_images(FIX)  # recorded camera frames are local data; the public repository has none


def fixture_dir(name):
    """A fixture image folder. Skip when this checkout has no recorded frames at all; a folder missing from a checkout
    that has frames is an error."""
    p = os.path.join(FIX, name)
    if not os.path.isdir(p):
        if not IMAGES_PRESENT:
            raise Skip(f"fixtures/{name}: recorded frames are local data, not in this checkout")
        raise FileNotFoundError(p)
    return p


def load_img(rel):
    img = cv2.imread(os.path.join(FIX, rel))
    if img is None:
        if not IMAGES_PRESENT:
            raise Skip(f"fixture image {rel}: recorded frames are local data, not in this checkout")
        raise FileNotFoundError(rel)
    return img
''')])

edit(f"{T}/test_perception.py", [
    ('files = sorted(os.listdir(os.path.join(C.FIX, "hand")))', 'files = sorted(os.listdir(C.fixture_dir("hand")))'),
    ('unlabelled = sorted(set(os.listdir(os.path.join(C.FIX, "hand"))) - labelled)',
     'unlabelled = sorted(set(os.listdir(C.fixture_dir("hand"))) - labelled)'),
])

guard = '''    if pre is None or st is None:
        import conftest as C
        if not C.IMAGES_PRESENT:
            raise C.Skip("fixtures/closeloc: recorded frames are local data, not in this checkout")
        raise AssertionError("fixture frames missing")
'''
s = open(f"{T}/test_closeloc.py").read()
old = '    assert pre is not None and st is not None, "fixture frames missing"\n'
if s.count(old) != 2:
    sys.exit(f"ABORT test_closeloc anchor x{s.count(old)}")
shutil.copy(f"{T}/test_closeloc.py", os.path.join(BAK, "test_closeloc.py.p3"))
open(f"{T}/test_closeloc.py.tmp", "w").write(s.replace(old, guard))
py_compile.compile(f"{T}/test_closeloc.py.tmp", doraise=True)
os.replace(f"{T}/test_closeloc.py.tmp", f"{T}/test_closeloc.py")
print("patched test_closeloc.py")

GI = """__pycache__/
*.pyc
*.tmp
*.orig
*.bak
.env
run/
logs/
data/
*.mkv
*.mp4
*.npz
*.npy
tools/tests/fixtures/**/*.png
tools/tests/fixtures/**/*.jpg
tools/tests/fixtures/*.png
tools/tests/fixtures/*.jpg
tools/scan_pairs*.jsonl
"""
open(f"{A}/.gitignore.public", "w").write(GI)
e = "/home/intel/intel-junction-2026/work/opus-verify/export.py"
s = open(e).read()
if '"README.md", ".gitignore.public"' not in s:
    s = s.replace('EXTRA = ["tools/start_all.sh"', 'EXTRA = ["README.md", ".gitignore.public", "tools/start_all.sh"')
    s = s.replace("files = sorted(f for f in files if", "files = sorted(f for f in files if f != \".gitignore\" and")
    s = s.replace("    if f.endswith(\".sh\"):\n        os.chmod(dst, 0o755)\n",
                  "    if f.endswith(\".sh\"):\n        os.chmod(dst, 0o755)\nif os.path.exists(os.path.join(OUT, \".gitignore.public\")):\n"
                  "    os.replace(os.path.join(OUT, \".gitignore.public\"), os.path.join(OUT, \".gitignore\"))\n")
    s = s.replace('TEXT = (".py"', 'TEXT = (".public", ".py"')
    open(e, "w").write(s)
    print("export.py: README.md and .gitignore added")
