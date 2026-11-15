"""mcmd.py: motion-server folder from paths.RUN_ROOT; conftest: a missing fixture image skips its test (images are
local data and are not in the public repository)."""
import os
import py_compile
import re
import shutil
import sys

T = "/home/intel/intel-junction-2026/app/tools"
BAK = "/home/intel/intel-junction-2026/run/bak_paths"


def edit(path, fn):
    s0 = open(path).read()
    s = fn(s0)
    if s == s0:
        print("unchanged", path)
        return
    shutil.copy(path, os.path.join(BAK, os.path.basename(path) + ".p2"))
    open(path + ".tmp", "w").write(s)
    py_compile.compile(path + ".tmp", doraise=True)
    os.replace(path + ".tmp", path)
    print("patched", path)


def mcmd(s):
    old = 'D = "/home/intel/intel-junction-2026/run/mserver"'
    if s.count(old) != 1:
        sys.exit("ABORT mcmd anchor")
    head = s.split(old)[0]
    add = []
    if not re.search(r"^import (.*\b)?sys\b", head, re.M):
        add.append("import sys  # noqa: E402")
    if not re.search(r"^import (.*\b)?os\b", head, re.M):
        add.append("import os  # noqa: E402")
    add += ["sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))",
            "from paths import RUN_ROOT  # noqa: E402", 'D = os.path.join(RUN_ROOT, "mserver")']
    return s.replace(old, "\n".join(add))


def conftest(s):
    old = "    img = cv2.imread(os.path.join(FIX, rel))\n"
    if s.count(old) != 1:
        sys.exit("ABORT conftest anchor")
    i = s.index(old) + len(old)
    if "img is None" in s[i:i + 200]:
        nxt = s[i:i + 200].split("\n")[0]
        print("conftest already checks None:", nxt)
        return s.replace(old + nxt + "\n", old + "    if img is None:\n        raise Skip(f\"fixture image {rel} is local data, not present\")\n", 1) \
            if "raise Skip" not in s[i:i + 200] else s
    return s.replace(old, old + "    if img is None:\n        raise Skip(f\"fixture image {rel} is local data, not present\")\n", 1)


edit(f"{T}/mcmd.py", mcmd)
edit(f"{T}/tests/conftest.py", conftest)
