"""Language-conditioned LEGO placement: blue plate left, green plate right."""
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
        self.initial_geom_friction = self.model.geom_friction.copy()

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
        cells = np.array([(0,-.10),(.075,-.09),(0,-.25),(.075,-.27),(0,-.34),(.075,-.35)])
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
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def observe(self):
        obs = super().observe()
        obs["task"] = self.instruction
        return obs
