"""TCP forward kinematics for the SO-101 MJCF models (P3-TWIN-ROBOT, E1 G1 gate).

ArmModel loads one MJCF, sets the five body joints from normalized units (units.K), and reads the TCP from site
gripper_frame_link, the URDF tip frame that kin.fk returns. Used for twin_ours and the official_urdf baseline.
"""
import hashlib
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import units  # noqa: E402

TCP_SITE = "gripper_frame_link"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ArmModel:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.qadr = []
        for joint in units.BODY_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if jid < 0:
                raise KeyError(f"joint {joint!r} is not in {xml_path}")
            self.qadr.append(int(self.model.jnt_qposadr[jid]))
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        if self.site < 0:
            raise KeyError(f"site {TCP_SITE!r} is not in {xml_path}")
        self.set_rad(np.zeros(len(self.qadr)))

    def set_rad(self, q5):
        for adr, q in zip(self.qadr, q5):
            self.data.qpos[adr] = float(q)
        mujoco.mj_kinematics(self.model, self.data)

    def tcp_pose(self):
        return (self.data.site_xpos[self.site].copy(),
                self.data.site_xmat[self.site].reshape(3, 3).copy())


def tcp_batch(arm, joints_norm):
    """Normalized body joints (N, at least 5 columns) -> TCP positions in mm (N, 3) and rotations (N, 3, 3), base frame."""
    joints = np.asarray(joints_norm, dtype=float)[:, :5]
    kb = np.array([units.K[j] for j in units.BODY_JOINTS])
    q_all = joints * kb
    pos = np.empty((len(q_all), 3))
    rot = np.empty((len(q_all), 3, 3))
    for i, q in enumerate(q_all):
        arm.set_rad(q)
        p, r = arm.tcp_pose()
        pos[i] = p * 1000.0
        rot[i] = r
    return pos, rot


def pose_errors(p_mm, rot, p_ref_mm, rot_ref):
    """Per-row position error (mm) and orientation error (deg, angle of rot^T rot_ref)."""
    dp = np.linalg.norm(np.asarray(p_mm, dtype=float) - np.asarray(p_ref_mm, dtype=float), axis=1)
    err = np.einsum("nji,njk->nik", np.asarray(rot, dtype=float), np.asarray(rot_ref, dtype=float))
    trace = np.trace(err, axis1=1, axis2=2)
    vec = np.stack([err[:, 2, 1] - err[:, 1, 2],
                    err[:, 0, 2] - err[:, 2, 0],
                    err[:, 1, 0] - err[:, 0, 1]], axis=1) / 2.0
    angle = np.arctan2(np.linalg.norm(vec, axis=1), (trace - 1.0) / 2.0)
    return dp, np.degrees(angle)
