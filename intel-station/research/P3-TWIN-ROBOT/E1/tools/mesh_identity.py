#!/usr/bin/env python3
"""E1 step: verify that the SO-101 STL meshes our URDF references are byte-identical to the objects the Intel plugin
pins (Git LFS pointers in repos/physicalai/.../urdf/so101/assets), then copy only the verified files into the
deliverable folder R/sim/twin/robot/meshes. The source is a read-only copy of the Menagerie SO-101 assets under
R/twin/vendor (another session's folder; read only, nothing written there). Writes mesh_identity.json.
Run: nice -n 19 $SIM_PY -I mesh_identity.py"""
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, "/home/intel/intel-junction-2026/work/P3-TWIN-ROBOT/E1/tools")
import urdf_tools as ut  # noqa: E402

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-ROBOT/E1"
OUR = (f"{R}/repos/edge-developer-kit-reference-scripts/samples/ai/edge-ai-demo-studio/workers/services/"
       "robotics-ai/platforms/so101/urdf/so101.urdf")
PLUG_ASSETS = f"{R}/repos/physicalai/packages/physicalai-mujoco-plugin/urdf/so101/assets"
VENDOR_ASSETS = f"{R}/twin/vendor/mujoco_menagerie/robotstudio_so101/assets"
DEST = f"{R}/sim/twin/robot/meshes"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pointer(path):
    oid, size = None, None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("oid sha256:"):
                oid = line.split(":", 1)[1].strip()
            elif line.startswith("size "):
                size = int(line.split()[1])
    return oid, size


def main():
    ours = ut.parse_urdf(OUR)
    names = sorted({v["mesh_file"] for vis in ours["visuals"].values() for v in vis})
    rows, all_ok = [], True
    for fname in names:
        oid, psize = pointer(os.path.join(PLUG_ASSETS, fname))
        src = os.path.join(VENDOR_ASSETS, fname)
        vsize = os.path.getsize(src)
        vsha = sha256(src)
        ok = (vsha == oid) and (vsize == psize)
        all_ok = all_ok and ok
        rows.append({"file": fname, "plugin_lfs_oid": oid, "plugin_size": psize,
                     "vendored_sha256": vsha, "vendored_size": vsize, "identical": ok})
    copied = []
    if all_ok:
        os.makedirs(DEST, exist_ok=True)
        for fname in names:
            shutil.copyfile(os.path.join(VENDOR_ASSETS, fname), os.path.join(DEST, fname))
            copied.append(os.path.join(DEST, fname))
    report = {"our_urdf_meshes": names, "all_identical": all_ok, "rows": rows,
              "copied_to": DEST if all_ok else None, "copied_files": len(copied),
              "copy_source": VENDOR_ASSETS}
    with open(f"{E1}/mesh_identity.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    for r in rows:
        print(f"{r['file']:40s} identical={r['identical']} size={r['vendored_size']} sha={r['vendored_sha256'][:12]}")
    print("all_identical:", all_ok, "copied:", len(copied))


if __name__ == "__main__":
    main()
