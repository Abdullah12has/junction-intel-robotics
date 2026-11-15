#!/usr/bin/env python3
"""V1 mesh identity check (stdlib only). For every mesh file that sim/twin/robot/so101_twin.xml references: is the
copy in sim/twin/robot/meshes a binary STL whose sha256 and size equal the oid and size of the Git LFS pointer of the
same file in the plugin's urdf/so101/assets? Also compares the MJCF mesh names with the visual meshes of the URDF
that kin.py loads. Writes V1/outputs/mesh_check.json."""
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET

R = "/home/intel/intel-junction-2026"
V1 = f"{R}/work/P3-TWIN-ROBOT/V1"
MESH = f"{R}/sim/twin/robot/meshes"
PLUG = f"{R}/repos/physicalai/packages/physicalai-mujoco-plugin/urdf/so101/assets"
MJCF = f"{R}/sim/twin/robot/so101_twin.xml"
URDF_KIN = (f"{R}/repos/edge-developer-kit-reference-scripts/samples/ai/edge-ai-demo-studio/workers/services/"
            "robotics-ai/platforms/so101/urdf/so101.urdf")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pointer(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    oid = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.M)
    size = re.search(r"^size (\d+)$", text, re.M)
    is_ptr = text.startswith("version https://git-lfs.github.com/spec/v1")
    return (oid.group(1) if oid else None), (int(size.group(1)) if size else None), is_ptr


def main():
    mjcf_files = sorted({el.get("file") for el in ET.parse(MJCF).getroot().iter("mesh") if el.get("file")})
    root = ET.parse(URDF_KIN).getroot()
    urdf_files = sorted({m.get("filename").split("/")[-1] for v in root.iter("visual")
                         for m in v.iter("mesh") if m.get("filename")})
    rows = []
    for f in mjcf_files:
        oid, psize, is_ptr = pointer(os.path.join(PLUG, f))
        ours = os.path.join(MESH, f)
        h = sha(ours)
        s = os.path.getsize(ours)
        with open(ours, "rb") as fh:
            head = fh.read(5)
        rows.append({"file": f, "plugin_pointer": is_ptr, "plugin_oid": oid, "plugin_size": psize,
                     "ours_sha256": h, "ours_size": s, "ours_first_bytes_is_solid": head == b"solid",
                     "identical": bool(h == oid and s == psize)})
    res = {"mjcf_mesh_files": len(mjcf_files), "urdf_visual_mesh_files": len(urdf_files),
           "mjcf_names_equal_urdf_visual_names": bool(set(mjcf_files) == set(urdf_files)),
           "all_identical": bool(all(r["identical"] for r in rows)), "rows": rows}
    os.makedirs(f"{V1}/outputs", exist_ok=True)
    with open(f"{V1}/outputs/mesh_check.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for r in rows:
        print(r["file"], "identical", r["identical"], "size", r["ours_size"], "sha", r["ours_sha256"][:16])
    print("mjcf_files", len(mjcf_files), "urdf_visual", len(urdf_files), "names_equal",
          res["mjcf_names_equal_urdf_visual_names"], "all_identical", res["all_identical"])


if __name__ == "__main__":
    main()
