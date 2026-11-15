"""Collect campaign simulation candidates; never connects to hardware.

Uses current station model and explicitly saves overhead + mounted wrist.
Camera/contact transfer remains provisional and is recorded as such.
"""

import argparse, sys, json, hashlib, time
from pathlib import Path

SIM = Path(__file__).resolve().parents[2] / "simulation"
sys.path.insert(0, str(SIM))
import av, numpy as np
from sorting_env import BrickSortingEnv
from station_coordinates import StationCoordinates, constrain_simulation
from expert import PickPlaceExpert, PlanningError
from collect import EpisodeRecorder


class Recorder(EpisodeRecorder):
    def __init__(self, path, coordinates):
        self.coordinates = coordinates
        self.directory = path
        path.mkdir(parents=True, exist_ok=False)
        self.states = []
        self.actions = []
        self.trace = []
        self.videos = {}
        for key in ["overhead", "wrist_mounted"]:
            container = av.open(str(path / f"{key}.mp4"), "w")
            stream = container.add_stream("libx264", rate=30)
            stream.width = 640
            stream.height = 480
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "fast"}
            self.videos[key] = (container, stream)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--attempts", type=int, default=500)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        str(f.relative_to(SIM)): hashlib.file_digest(f.open("rb"), "sha256").hexdigest()
        for f in SIM.glob("*.py")
    }
    (a.output / "source_hashes.json").write_text(json.dumps(provenance, indent=2))
    env = BrickSortingEnv(render=True, horizon=1800)
    coords = StationCoordinates(env.limits)
    constrain_simulation(env, coords)

    # Override only rendering to omit the unused fixed-close view and reduce work.
    def observe():
        obs = {
            "observation.state": env.data.qpos[env.qadr].astype(np.float32).copy(),
            "task": env.instruction,
        }
        for camera in ["overhead", "wrist_mounted"] if env.renderer else []:
            env.renderer.update_scene(env.data, camera=camera)
            obs[f"observation.images.{camera}"] = env.renderer.render().copy()
        return obs

    env.observe = observe
    rows = []
    accepted = 0
    t0 = time.monotonic()
    try:
        for attempt in range(a.attempts):
            if accepted >= a.episodes:
                break
            folder = a.output / f"attempt-{attempt:04d}"
            if folder.exists() and not (folder / "metadata.json").exists():
                folder.rename(folder.with_name(folder.name + "-interrupted"))
            if folder.exists():
                meta = json.loads((folder / "metadata.json").read_text())
                rows.append(meta)
                accepted += int(meta["accepted"])
                continue
            color = ["red", "orange", "white", "cyan", "yellow", "purple"][attempt % 6]
            seed = 41000 + attempt // 6
            renderer = env.renderer
            env.renderer = None
            env.reset(seed, target_color=color, destination="blue")
            error = None
            try:
                preflight = PickPlaceExpert(env).run()
            except (PlanningError, ValueError) as exc:
                preflight = False
                error = str(exc)
            env.renderer = renderer
            if not preflight:
                folder.mkdir()
                meta = {
                    "accepted": False,
                    "seed": seed,
                    "attempt": attempt,
                    "preflight_only": True,
                    "frames": 0,
                    "error": error,
                }
                (folder / "metadata.json").write_text(json.dumps(meta))
                rows.append(meta)
                continue
            env.reset(seed, target_color=color, destination="blue")
            camera = env.model.camera("overhead")
            cid = camera.id
            world = env.data.body(env.brick_body_name).xpos.copy()
            local = env.data.cam_xmat[cid].reshape(3, 3).T @ (
                world - env.data.cam_xpos[cid]
            )
            focal = 240 / np.tan(np.deg2rad(float(camera.fovy[0])) / 2)
            initial_xy = [
                320 + focal * local[0] / (-local[2]),
                240 - focal * local[1] / (-local[2]),
            ]
            rec = Recorder(folder, coords)
            expert = PickPlaceExpert(env, on_step=rec.record)
            error = None
            try:
                success = expert.run()
            except (PlanningError, ValueError) as exc:
                success = False
                error = str(exc)
            meta = rec.finish(env, success, error)
            meta.update(
                seed=seed,
                attempt=attempt,
                reference_layout=False,
                camera_roles={"overhead": "overhead", "wrist": "wrist_mounted"},
                camera_alignment_verified=False,
                transfer_status="candidate: nominal contact and wrist camera mismatch remain",
                initial_target_xyz=rec.trace[0]["object_xyz"] if rec.trace else None,
                initial_target_xy=initial_xy,
            )
            (folder / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
            rows.append(meta)
            accepted += int(meta["accepted"])
            (a.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "attempt": attempt,
                        "accepted": accepted,
                        "success": meta["accepted"],
                        "frames": meta["frames"],
                        "elapsed_s": round(time.monotonic() - t0, 1),
                        "error": error,
                    }
                ),
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
