"""Offline simulation only. No robot drivers, serial access, or hardware mapping."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).parent
BRICKS = json.loads((ROOT / "bricks.json").read_text())
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def scene_xml(brick_type="2x4"):
    spec = BRICKS["variants"][brick_type]
    half_size = np.array(spec["body_size_m"]) / 2
    source = ROOT / "assets/robotstudio_so101/so101.xml"
    if not source.exists():
        raise FileNotFoundError("Run uv run python setup_assets.py first")
    root = ET.parse(source).getroot()
    root.find("compiler").set("meshdir", str(source.parent / "assets"))
    # 20 physics steps per 30 Hz observation/action, with no timing drift.
    root.find("option").set("timestep", str(1 / 600))
    for keyframe in root.findall("keyframe"):
        root.remove(keyframe)
    visual = root.find("visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 -0.3 1.5", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="table", type="plane", size="0.6 0.6 0.02", rgba="0.55 0.50 0.42 1")
    ET.SubElement(world, "camera", name="overhead", pos="0 -0.15 0.75", xyaxes="1 0 0 0 1 0", fovy="55")
    # Approximate camera placement for initial simulation; not station calibration.
    gripper = world.find(".//body[@name='gripper']")
    ET.SubElement(gripper, "camera", name="wrist", pos="0.10 0 -0.04", xyaxes="0 1 0 -0.669 0 0.743", fovy="75")
    brick = ET.SubElement(world, "body", name="brick", pos=f"0 -0.23 {half_size[2] + 0.001}")
    ET.SubElement(brick, "freejoint", name="brick_free")
    ET.SubElement(brick, "geom", name="brick_geom", type="box", size=" ".join(map(str, half_size)),
                  mass=str(spec["approximate_mass_kg"]), friction="0.8 0.01 0.001", rgba="0.8 0.15 0.08 1")
    nx, ny = spec["stud_grid"]
    for x in range(nx):
        for y in range(ny):
            pos = [(x - (nx - 1) / 2) * BRICKS["stud_pitch_m"],
                   (y - (ny - 1) / 2) * BRICKS["stud_pitch_m"],
                   half_size[2] + BRICKS["stud_height_m"] / 2]
            ET.SubElement(brick, "geom", name=f"stud_{x}_{y}", type="cylinder",
                          pos=" ".join(map(str, pos)),
                          size=f"{BRICKS['stud_radius_m']} {BRICKS['stud_height_m'] / 2}",
                          mass="0", friction="0.8 0.01 0.001", rgba="0.8 0.15 0.08 1")
    ET.SubElement(world, "site", name="inspection_target", type="box", pos="0.12 -0.20 0.001", size="0.045 0.045 0.001", rgba="0.1 0.8 0.3 0.35")
    return ET.tostring(root, encoding="unicode")


class BrickInspectionEnv:
    """Absolute-radian position control, 30 Hz, randomized seeded resets.

    Observation and action units are intentionally NOT the station's normalized
    LeRobot units. A measured conversion is required before policy transfer.
    """

    def __init__(self, render=False, horizon=900, brick_type="2x4", scene=None):
        self.brick_type = brick_type
        self.half_height = BRICKS["variants"][brick_type]["body_size_m"][2] / 2
        self.model = mujoco.MjModel.from_xml_string(scene if scene is not None else scene_xml(brick_type))
        self.support_height = 0.0
        self.support_geom_ids = {self.model.geom("table").id}
        self.brick_geoms = np.flatnonzero(self.model.geom_bodyid == self.model.body("brick").id)
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.array([self.model.joint(n).id for n in JOINTS])
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.actuator_ids = np.array([
            np.flatnonzero(self.model.actuator_trnid[:, 0] == j)[0] for j in self.joint_ids
        ])
        self.limits = self.model.jnt_range[self.joint_ids].copy()
        self.brick_adr = self.model.joint("brick_free").qposadr[0]
        self.brick_body_name = "brick"
        self.brick_joint_name = "brick_free"
        self.renderer = mujoco.Renderer(self.model, height=480, width=640) if render else None
        self.horizon = horizon
        self.elapsed = 0
        self.success_steps = 0
        self.finished = False
        self.lifted = False

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        # Model-only initial pose. Never interpreted as hardware home values.
        home = np.array([0, -0.5, 0.8, 0.6, 0, 0.5])
        self.data.qpos[self.qadr] = np.clip(home, self.limits[:, 0], self.limits[:, 1])
        self.data.ctrl[self.actuator_ids] = self.data.qpos[self.qadr]
        a = self.brick_adr
        self.data.qpos[a:a + 3] = [rng.uniform(-0.08, 0.06), rng.uniform(-0.28, -0.18), self.half_height + 0.001]
        yaw = rng.uniform(-np.pi, np.pi)
        self.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.model.geom_friction[self.brick_geoms, 0] = rng.uniform(0.6, 1.0)
        self.model.geom_rgba[self.brick_geoms, :3] = rng.uniform(0.1, 0.9, 3)
        mujoco.mj_forward(self.model, self.data)
        self.elapsed = self.success_steps = 0
        self.finished = False
        self.lifted = False
        return self.observe()

    def observe(self):
        obs = {"observation.state": self.data.qpos[self.qadr].astype(np.float32).copy()}
        if self.renderer:
            for name in ("overhead", "wrist"):
                self.renderer.update_scene(self.data, camera=name)
                obs[f"observation.images.{name}"] = self.renderer.render().copy()
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if self.finished:
            raise RuntimeError("Episode ended; call reset before stepping")
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Action must be six finite absolute joint targets in radians")
        if np.any(action < self.limits[:, 0]) or np.any(action > self.limits[:, 1]):
            raise ValueError("Action exceeds the simulation model joint limits")
        self.data.ctrl[self.actuator_ids] = action
        mujoco.mj_step(self.model, self.data, nstep=20)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Simulation became non-finite")
        self.elapsed += 1
        pos = self.data.body(self.brick_body_name).xpos
        self.lifted = self.lifted or bool(pos[2] > 0.05)
        target = self.model.site("inspection_target").pos
        velocity = self.data.qvel[self.model.joint(self.brick_joint_name).dofadr[0]:][:6]
        # Conservative center margin covers the rotated brick footprint.
        placed = bool(np.all(np.abs(pos[:2] - target[:2]) < 0.015) and
                      abs(pos[2] - self.half_height - self.support_height) < 0.001 and np.linalg.norm(velocity) < 0.02)
        touching_arm = any((c.geom1 in self.brick_geoms or c.geom2 in self.brick_geoms) and
                           not self.support_geom_ids.intersection((c.geom1, c.geom2)) for c in self.data.contact)
        self.success_steps = self.success_steps + 1 if self.lifted and placed and not touching_arm else 0
        success = self.success_steps >= 15
        truncated = self.elapsed >= self.horizon and not success
        self.finished = success or truncated
        return self.observe(), float(success), success, truncated, {"success": success, "sim_only": True}

    def close(self):
        if self.renderer:
            self.renderer.close()
