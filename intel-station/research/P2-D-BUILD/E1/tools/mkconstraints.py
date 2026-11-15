import collections
import sys
import tomllib

EXCLUDE = {"setuptools", "pip"}
lock_path, out_path = sys.argv[1], sys.argv[2]
with open(lock_path, "rb") as fh:
    lock = tomllib.load(fh)
versions = collections.defaultdict(set)
for pkg in lock.get("package", []):
    source = pkg.get("source", {})
    if "registry" not in source or "version" not in pkg:
        continue
    versions[pkg["name"]].add(pkg["version"])
pins, multi, excluded = [], [], []
for name in sorted(versions):
    found = versions[name]
    if name in EXCLUDE:
        excluded.append(name)
    elif len(found) == 1:
        pins.append(f"{name}=={next(iter(found))}")
    else:
        multi.append(name)
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(pins) + "\n")
print(f"pins={len(pins)} skipped_multi_version={len(multi)} excluded={','.join(excluded)}")
print("skipped=" + ",".join(multi))
