import argparse
import json
from pathlib import Path
import time
from queue import SimpleQueue

import mujoco
import numpy as np
from PIL import Image

from so101_env import BrickInspectionEnv
from sorting_env import BrickSortingEnv, COLORS
from normalized_env import NormalizedSortingEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--units", choices=("normalized", "radians"), default="radians")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--brick", choices=("2x2", "2x4"), default="2x4")
    parser.add_argument("--scene", choices=("sorting", "inspection"), default="sorting")
    parser.add_argument("--layout", choices=("reference", "random"), default="reference")
    args = parser.parse_args()
    sorting_class = NormalizedSortingEnv if args.units == "normalized" else BrickSortingEnv
    env = sorting_class(render=True) if args.scene == "sorting" else BrickInspectionEnv(render=True, brick_type=args.brick)
    try:
        reference = args.layout == "reference"
        obs = env.reset(args.seed, reference=reference) if args.scene == "sorting" else env.reset(args.seed)
        action = obs["observation.state"].astype(float)
        for _ in range(90):
            obs, _, _, _, info = env.step(action)
        out = Path(__file__).parent / "outputs" / ("sorting" if args.scene == "sorting" else args.brick)
        out.mkdir(parents=True, exist_ok=True)
        for camera in ("overhead", "wrist", "wrist_mounted"):
            Image.fromarray(obs[f"observation.images.{camera}"]).save(out / f"{camera}.png")
        report = {"mujoco": mujoco.__version__, "brick": args.brick, "steps": 90, "sim_seconds": env.data.time,
                  "finite": bool(np.isfinite(env.data.qpos).all()), "state": obs["observation.state"].tolist(),
                  "task_success": info["success"], "hardware_mapping_verified": False,
                  "action_units": getattr(env, "action_units", "absolute_joint_radians")}
        if args.scene == "sorting":
            report.update(task=env.instruction, block_count=len(env.active_indices),
                          colors=[env.block_colors[i] for i in env.active_indices])
        (out / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if args.viewer:
            from mujoco import viewer as mj_viewer
            keys = SimpleQueue()
            print("R: randomize. V: reference layout. C: camera. B/G: target plate. No trained policy is running.")
            with mj_viewer.launch_passive(env.model, env.data, key_callback=keys.put) as viewer:
                viewer.cam.lookat[:] = [0, -0.17, 0.02]
                viewer.cam.distance = 0.65
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -70
                camera_mode = 2
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                seed = args.seed
                while viewer.is_running():
                    start = time.monotonic()
                    while not keys.empty():
                        key = keys.get()
                        if key in (67, 49, 50, 51, 52):
                            camera_mode = (camera_mode + 1) % 4 if key == 67 else key - 49
                            if camera_mode == 2:
                                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                            else:
                                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                                viewer.cam.fixedcamid = env.model.camera(("overhead", "wrist", None, "wrist_mounted")[camera_mode]).id
                        if key in (82, 86, 66, 71):
                            if key == 82:
                                seed += 1
                                reference = False
                            if key == 86:
                                reference = True
                            with viewer.lock():
                                if args.scene == "sorting":
                                    obs = env.reset(seed, destination={66: "blue", 71: "green"}.get(key), reference=reference)
                                    print(env.instruction)
                                else:
                                    obs = env.reset(seed)
                    # Viewer is an inspection sandbox; reward/termination is tested
                    # through env.step, not asserted by this passive hold loop.
                    mujoco.mj_step(env.model, env.data, nstep=20)
                    if args.scene == "sorting":
                        viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                          env.instruction + "\n1: overhead | 2: close | 3: orbit | 4: wrist mount (uncalibrated)\nR: random | V: reference | B/G: plate | No trained policy", ""))
                    viewer.sync()
                    time.sleep(max(0, 1 / 30 - (time.monotonic() - start)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
