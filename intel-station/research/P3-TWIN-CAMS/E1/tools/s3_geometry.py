#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 3: jaw tip geometry on the URDF links (read-only inputs).

Meshes: repos/physicalai/packages/physicalai-mujoco-plugin/urdf/so101/assets (same file names as the
so101.urdf visuals). Visual origins are taken from the edge-developer so101.urdf used by app/tools/kin.py:
  gripper_link   <- wrist_roll_follower_so101_v1.stl  (fixed jaw, visual xyz/rpy from URDF lines 278-280)
  moving_jaw link <- moving_jaw_so101_v1.stl           (visual xyz 0 0 0.0189, URDF lines 318-320)
  joint 'gripper' origin in gripper_link: xyz 0.0202 0.0188 -0.0234, rpy 1.5708 0 0, axis z (URDF lines 333-337)
  gripper_frame_joint origin in gripper_link: xyz -0.0079 -0.000218121 -0.0981274, rpy 0 pi 0 (tool frame TCP)

Tip definition (documented): jaw tip = mesh vertex with the minimum z in the gripper_link frame at the calibration
rest angle, tracked as the moving jaw rotates about its joint axis by theta. Centroid = mean of the jaw mesh vertices.
Run: envs/data-env/bin/python -I tools/s3_geometry.py
"""
import hashlib
import json
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-CAMS/E1"
sys.path.insert(0, f"{R}/app")
from pick_v2.kinematics import rpy_matrix, transform  # noqa: E402  (module read as text by the app; no side effects)

URDF = (f"{R}/repos/edge-developer-kit-reference-scripts/samples/ai/edge-ai-demo-studio/workers/services/"
        "robotics-ai/platforms/so101/urdf/so101.urdf")
MESH_DIRS = [f"{R}/twin/vendor/mujoco_menagerie/robotstudio_so101/assets",  # real STL bytes (LFS pointers below)
             f"{R}/repos/physicalai/packages/physicalai-mujoco-plugin/urdf/so101/assets",
             f"{R}/repos/physicalai/packages/physicalai-so101-plugin/urdf/so101-follower/assets"]


def read_stl(path):
    data = open(path, "rb").read()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            dt = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            arr = np.frombuffer(data, dtype=dt, count=n, offset=84)
            return arr["v"].reshape(-1, 3).astype(np.float64), "binary"
    verts = []
    for line in data.decode("ascii", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("vertex"):
            verts.append([float(v) for v in s.split()[1:4]])
    return np.asarray(verts, dtype=np.float64), "ascii"


def urdf_joint(root, name):
    for el in root.findall("joint"):
        if el.get("name") == name:
            o = el.find("origin")
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
            ax = el.find("axis")
            axis = [float(v) for v in ax.get("xyz", "0 0 1").split()] if ax is not None else [0, 0, 1]
            return transform(rpy_matrix(rpy), xyz), np.asarray(axis)
    raise KeyError(name)


def urdf_visual(root, link, mesh_fragment):
    for ln in root.findall("link"):
        if ln.get("name") != link:
            continue
        for vis in ln.findall("visual"):
            m = vis.find("geometry/mesh")
            if m is not None and mesh_fragment in m.get("filename", ""):
                o = vis.find("origin")
                xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
                scale = m.get("scale")
                return transform(rpy_matrix(rpy), xyz), scale
    raise KeyError((link, mesh_fragment))


def rz(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def find_mesh(name):
    for d in MESH_DIRS:
        p = f"{d}/{name}"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def main():
    root = ET.parse(URDF).getroot()
    T_fix_vis, scale_f = urdf_visual(root, "gripper_link", "wrist_roll_follower")
    T_mov_vis, scale_m = urdf_visual(root, "moving_jaw_so101_v1_link", "moving_jaw")
    T_grip, axis_g = urdf_joint(root, "gripper")
    T_tcp, _ = urdf_joint(root, "gripper_frame_joint")
    lim = [el for el in root.findall("joint") if el.get("name") == "gripper"][0].find("limit")
    fp = find_mesh("wrist_roll_follower_so101_v1.stl")
    mp = find_mesh("moving_jaw_so101_v1.stl")
    md5 = {os.path.basename(p): hashlib.md5(open(p, "rb").read()).hexdigest() for p in
           [fp, mp] + [f"{d}/wrist_roll_follower_so101_v1.stl" for d in MESH_DIRS[1:]]
           + [f"{d}/moving_jaw_so101_v1.stl" for d in MESH_DIRS[1:]]}
    vf, fmt_f = read_stl(fp)
    vm, fmt_m = read_stl(mp)
    Vf = (T_fix_vis[:3, :3] @ vf.T).T + T_fix_vis[:3, 3]      # fixed jaw mesh in gripper_link frame, metres
    Vm = (T_mov_vis[:3, :3] @ vm.T).T + T_mov_vis[:3, 3]      # moving jaw mesh in moving-jaw link frame
    i_f = int(np.argmin(Vf[:, 2]))
    i_m = int(np.argmin(Vm[:, 2]))
    tip_f = Vf[i_f] * 1000.0                                  # gripper_link frame, mm
    rest = T_grip @ np.eye(4)                                 # gripper_link -> moving-jaw joint frame at theta=0
    Vm_g0 = (rest[:3, :3] @ Vm.T).T + rest[:3, 3]
    tip_m_g0 = Vm_g0[int(np.argmin(Vm_g0[:, 2]))] * 1000.0
    Tcp_g = T_tcp[:3, 3] * 1000.0
    out = {
        "units": "mm in gripper_link frame unless stated",
        "mesh_dirs_checked": MESH_DIRS,
        "stl_format": {"fixed_jaw": fmt_f, "moving_jaw": fmt_m},
        "stl_vertices": {"fixed_jaw": int(len(vf)), "moving_jaw": int(len(vm))},
        "mesh_scale_attr": {"fixed_jaw": scale_f, "moving_jaw": scale_m},
        "md5": md5,
        "fixed_jaw_bbox_gripper_mm": [(Vf.min(axis=0) * 1000).round(2).tolist(),
                                       (Vf.max(axis=0) * 1000).round(2).tolist()],
        "fixed_jaw_tip_gripper_mm": tip_f.round(3).tolist(),
        "fixed_jaw_centroid_gripper_mm": (Vf.mean(axis=0) * 1000).round(3).tolist(),
        "moving_jaw_bbox_moving_frame_mm": [(Vm.min(axis=0) * 1000).round(2).tolist(),
                                             (Vm.max(axis=0) * 1000).round(2).tolist()],
        "moving_jaw_tip_index_at_rest": i_m,
        "moving_jaw_tip_moving_frame_mm": (Vm[i_m] * 1000.0).round(3).tolist(),
        "moving_jaw_tip_gripper_mm_at_theta0": tip_m_g0.round(3).tolist(),
        "moving_jaw_centroid_moving_frame_mm": (Vm.mean(axis=0) * 1000).round(3).tolist(),
        "gripper_joint": {"origin_xyz_rpy_matrix": rest.round(6).tolist(), "axis": axis_g.tolist(),
                          "limit_lower_upper_rad": [float(lim.get("lower")), float(lim.get("upper"))]},
        "tcp_gripper_mm": Tcp_g.round(3).tolist(),
        "tip_to_tcp_mm": {"fixed": (tip_f - Tcp_g).round(3).tolist(),
                          "moving_theta0": (tip_m_g0 - Tcp_g).round(3).tolist()},
    }
    with open(f"{E1}/out/jaw_geometry.json", "w") as fh:
        json.dump(out, fh, indent=1)
    for k in ("stl_format", "stl_vertices", "mesh_scale_attr", "fixed_jaw_bbox_gripper_mm",
              "fixed_jaw_tip_gripper_mm", "fixed_jaw_centroid_gripper_mm", "moving_jaw_bbox_moving_frame_mm",
              "moving_jaw_tip_moving_frame_mm", "moving_jaw_tip_gripper_mm_at_theta0", "tcp_gripper_mm",
              "tip_to_tcp_mm"):
        print(k, out[k])
    print("md5", md5)


if __name__ == "__main__":
    main()
