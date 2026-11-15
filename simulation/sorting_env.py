"""Language-conditioned LEGO placement: blue plate left, green plate right."""
import json
from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from so101_env import BrickInspectionEnv, scene_xml
from station_scene import (BOWL_CENTERS, BOWL_FLOOR, BRICK_HALF_HEIGHT, HOME,
                           REFERENCE_COLORS, REFERENCE_POSITIONS, recreate_station, add_bowl, enlarge_brick)

COLORS = {
    "red": (0.85, 0.04, 0.03), "yellow": (0.95, 0.75, 0.02),
    "orange": (1.0, 0.28, 0.02), "purple": (0.48, 0.06, 0.75),
    "white": (0.94, 0.94, 0.94), "cyan": (0.02, 0.75, 0.85),
}
PLATES = BOWL_CENTERS
PLATE_TOP = BOWL_FLOOR
BLOCK_TYPES = ["2x2", "2x2", "2x2", "2x4", "2x2", "2x2"]
# Motor response fitted to real recordings by fit_dynamics.py: the real servos are far softer than
# the model default and act about two control steps late.
DYNAMICS = json.loads(Path(__file__).with_name("fitted_dynamics.json").read_text())


def sorting_scene():
    root = ET.fromstring(scene_xml("2x4"))
    world = root.find("worldbody")
    world.remove(world.find("body[@name='brick']"))
    # Keep the success region invisible; the physical plate supplies the visual.
    world.find("site[@name='inspection_target']").set("rgba", "0 0 0 0")
    recreate_station(root)
    for i, color in enumerate(COLORS):
        body = ET.fromstring(scene_xml(BLOCK_TYPES[i])).find("worldbody/body[@name='brick']")
        for element in body.iter():
            if "name" in element.attrib:
                element.set("name", f"block{i}_" + element.get("name"))
        # The first body retains base-class names; remaining names are unique.
        if i == 0:
            for element in body.iter():
                if "name" in element.attrib:
                    element.set("name", element.get("name").removeprefix("block0_"))
        enlarge_brick(body, root.find("asset"), f"block{i}_", BLOCK_TYPES[i])
        world.append(body)
    for name, (x, y) in PLATES.items():
        add_bowl(root, name, (x,y))
    return ET.tostring(root, encoding="unicode")


class BrickSortingEnv(BrickInspectionEnv):
    def __init__(self, render=False, horizon=900):
        self.instruction = ""
        super().__init__(render=render, horizon=horizon, scene=sorting_scene())
        self.half_height = BRICK_HALF_HEIGHT
        self.blocks = []
        for i in range(6):
            prefix = f"block{i}_" if i else ""
            body_id = self.model.body(prefix + "brick").id
            geoms = np.flatnonzero(self.model.geom_bodyid == body_id)
            self.blocks.append({"body": prefix + "brick", "joint": prefix + "brick_free",
                                "geoms": geoms, "alpha": self.model.geom_rgba[geoms,3].copy(),
                                "contype": self.model.geom_contype[geoms].copy(),
                                "conaffinity": self.model.geom_conaffinity[geoms].copy(),
                                "type": BLOCK_TYPES[i]})
        self.active_indices = []
        # Lying flat on the bowl floor counts, as it does for the operator: real releases
        # landed 0.9-3.7 cm from the bowl centre; the floor's radius is 6.2 cm.
        self.place_tolerance = 0.04
        for name, scale in DYNAMICS["joints"].items():
            actuator = self.model.actuator(name).id
            dof = self.model.joint(name).dofadr[0]
            self.model.actuator_gainprm[actuator, 0] *= scale["kp"]
            self.model.actuator_biasprm[actuator, 1] *= scale["kp"]
            self.model.actuator_biasprm[actuator, 2] *= scale["kv"]
            self.model.dof_damping[dof] *= scale["damping"]
            self.model.dof_frictionloss[dof] *= scale["frictionloss"]
        self.pending = deque()
        self.initial_geom_friction = self.model.geom_friction.copy()
        # Per-episode visual variation, so a policy tolerates the residual camera and lighting error.
        self.camera_ids = [self.model.camera(n).id for n in ("overhead", "wrist")]
        self.base_cam_pos = self.model.cam_pos[self.camera_ids].copy()
        self.base_cam_quat = self.model.cam_quat[self.camera_ids].copy()
        self.base_light = self.model.light_diffuse.copy()
        self.base_headlight = self.model.vis.headlight.diffuse.copy()
        self.base_ambient = self.model.vis.headlight.ambient.copy()
        self.randomize = True

    def reset(self, seed=0, target_color=None, destination=None, count=None, reference=False):
        rng = np.random.default_rng(seed)
        count = int(rng.integers(4, 7)) if count is None else count
        if count not in (4, 5, 6):
            raise ValueError("Choose 4, 5 or 6 blocks")
        if destination is not None and destination not in PLATES:
            raise ValueError("Destination must be blue or green")
        if target_color is not None and target_color not in COLORS:
            raise ValueError("Unknown target color")
        # Reset model parameters as well as MjData; otherwise the previous
        # target retains randomized friction and changes the next episode.
        self.model.geom_friction[:] = self.initial_geom_friction
        self.brick_geoms = self.blocks[0]["geoms"]
        super().reset(seed)
        self.data.qpos[self.qadr] = HOME
        self.data.ctrl[self.actuator_ids] = HOME
        self.block_colors = rng.permutation(list(COLORS)).tolist()
        self.active_indices = sorted(rng.choice(6, count, replace=False).tolist())
        if reference:
            self.block_colors = REFERENCE_COLORS.copy()
            self.active_indices = [0,1,2,4,5]
        if target_color is not None:
            required = self.block_colors.index(target_color)
            if required not in self.active_indices:
                self.active_indices[0] = required
                self.active_indices.sort()
        # Separated cells with slight jitter: no overlapping brick footprints.
        # Real pick-ups were at x 0.07..0.20, y -0.15..-0.30; cells stay clear of both bowls and the resting gripper.
        cells = np.array([(.07,-.17),(.145,-.17),(.07,-.26),(.15,-.25),(.11,-.34),(.20,-.31)])
        rng.shuffle(cells)
        for i, block in enumerate(self.blocks):
            active = i in self.active_indices
            geoms = block["geoms"]
            self.model.geom_contype[geoms] = block['contype'] * int(active)
            self.model.geom_conaffinity[geoms] = block['conaffinity'] * int(active)
            self.model.body_gravcomp[self.model.body(block["body"]).id] = 0 if active else 1
            self.model.geom_rgba[geoms, :3] = COLORS[self.block_colors[i]]
            self.model.geom_rgba[geoms, 3] = block['alpha'] * int(active)
            a = self.model.joint(block["joint"]).qposadr[0]
            xy = cells[i] + rng.uniform(-0.003, 0.003, 2)
            if reference:
                xy = REFERENCE_POSITIONS[i]
            self.data.qpos[a:a + 3] = [*xy, self.half_height + 0.001] if active else [2 + i, 0, -1]
            yaw = rng.uniform(-np.pi, np.pi)
            self.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.target_index = int(rng.choice(self.active_indices)) if target_color is None else self.block_colors.index(target_color)
        self.destination = destination or str(rng.choice(list(PLATES)))
        self.target_color = self.block_colors[self.target_index]
        block = self.blocks[self.target_index]
        self.brick_geoms = block["geoms"]
        self.brick_adr = self.model.joint(block["joint"]).qposadr[0]
        self.brick_body_name = block["body"]
        self.brick_joint_name = block["joint"]
        self.support_height = PLATE_TOP
        self.support_geom_ids = {self.model.geom(f"plate_{self.destination}").id}
        self.model.site("inspection_target").pos[:] = [*PLATES[self.destination], PLATE_TOP]
        self.instruction = f"move the {self.target_color} brick to the {self.destination} plate"
        self.pending = deque([np.array(HOME, float)] * DYNAMICS["delay_steps_30hz"])
        visual = np.random.default_rng([seed, 7])          # separate stream: layouts stay as before
        self.model.cam_pos[self.camera_ids] = self.base_cam_pos
        self.model.cam_quat[self.camera_ids] = self.base_cam_quat
        self.model.light_diffuse[:] = self.base_light
        self.model.vis.headlight.diffuse[:] = self.base_headlight
        self.model.vis.headlight.ambient[:] = self.base_ambient
        if self.randomize and not reference:
            for k, cam in enumerate(self.camera_ids):
                # fitted camera error is about 1 cm and 1-2 degrees
                self.model.cam_pos[cam] = self.base_cam_pos[k] + visual.uniform(-.01, .01, 3)
                axis = visual.normal(size=3); axis /= np.linalg.norm(axis)
                turn, out = np.zeros(4), np.zeros(4)
                mujoco.mju_axisAngle2Quat(turn, axis, np.deg2rad(visual.uniform(0, 1.5)))
                mujoco.mju_mulQuat(out, self.base_cam_quat[k], turn)
                self.model.cam_quat[cam] = out
            brightness = visual.uniform(.8, 1.2)
            self.model.light_diffuse[:] = np.clip(self.base_light * brightness, 0, 1)
            self.model.vis.headlight.diffuse[:] = np.clip(self.base_headlight * visual.uniform(.8, 1.2), 0, 1)
            self.model.vis.headlight.ambient[:] = np.clip(self.base_ambient * brightness, 0, 1)
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def step(self, action):
        # The recorded action is the command sent now; the motors act on it a few steps later.
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Action must be six finite absolute joint targets in radians")
        self.pending.append(action.copy())
        return super().step(self.pending.popleft())

    def observe(self):
        obs = super().observe()
        obs["task"] = self.instruction
        return obs
