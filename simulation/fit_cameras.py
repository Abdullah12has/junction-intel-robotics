"""Fit a sim camera to the real one using the arm as the calibration target.

Inputs are station captures (joint state + camera frame). The joint mapping puts
the model arm in each captured pose; camera position, orientation and field of
view are then optimised so the rendered gripper overlaps the real gripper, which
is found by colour (red fingers, orange mount) after removing static pixels.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_opening, distance_transform_edt
from scipy.optimize import minimize

from sorting_env import BrickSortingEnv
from station_coordinates import StationCoordinates

W, H = 320, 240
SERIAL = {'overhead': '243622061421', 'wrist': '243722064316'}


def load(directory, camera):
    d = Path(directory)
    if (d/'pose.json').exists():
        ticks = np.array([v[0] for v in json.loads((d/'pose.json').read_text())['ticks'].values()], float)
        state = None
    else:
        ticks, state = None, np.array(json.loads((d/'capture.json').read_text())['mean_normalized'])
    image = next(p for p in d.iterdir() if SERIAL[camera] in p.name and p.suffix in ('.jpg', '.png'))
    return ticks, state, np.asarray(Image.open(image).convert('RGB').resize((W, H)), float)


def load_sweep(directory, camera, count):
    """Evenly spaced, near-stationary frames from a sweep recording."""
    rows = json.loads((directory/'states.json').read_text())
    rows = [r for r in rows if r['moved'] < .5]
    name = {'overhead': 'overhead', 'wrist': 'close'}[camera]
    for r in rows[::max(1, len(rows)//count)]:
        image = np.asarray(Image.open(directory/f"{r['i']:04d}-{name}.jpg").convert('RGB').resize((W, H)), float)
        yield f"{directory.name}-{r['i']:04d}", np.array(r['state']), image


def gripper_colour(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 150) & (r > 1.5*b) & (r - np.minimum(g, b) > 70) & (g < 190)   # red fingers and orange mount


def main():
    p = argparse.ArgumentParser()
    p.add_argument('captures', nargs='*', type=Path)
    p.add_argument('--sweep', type=Path, help='Sweep recording: states.json plus NNNN-overhead/close.jpg')
    p.add_argument('--frames', type=int, default=30)
    p.add_argument('--fovy', type=float, default=42.5, help='Vertical field of view, held fixed (D415 colour datasheet)')
    p.add_argument('--camera', choices=tuple(SERIAL), default='overhead')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gate', type=int, default=45, help='Pixels (at 320 wide) around the starting render to search')
    a = p.parse_args()
    env = BrickSortingEnv(render=False)
    env.reset(1, reference=True)
    m, d = env.model, env.data
    c = StationCoordinates(env.limits)
    poses, masks, images, names = [], [], [], []
    items = [(d.name, *load(d, a.camera)) for d in a.captures]
    if a.sweep:
        items += [(n, None, st, im) for n, st, im in load_sweep(a.sweep, a.camera, a.frames)]
    for name, ticks, state, image in items:
        names.append(name)
        q = (ticks if ticks is not None else c.continuous_ticks(state))*c.slope+c.offset
        poses.append(np.clip(q, env.limits[:, 0], env.limits[:, 1])); images.append(image); masks.append(gripper_colour(image))
    bodies = {m.body(n).id for n in ('gripper', 'moving_jaw_so101_v1')}
    arm_geoms = np.array([m.geom_bodyid[g] in bodies and m.geom_rgba[g, 3] > 0 for g in range(m.ngeom)])
    cam = m.camera(a.camera).id
    renderer = mujoco.Renderer(m, height=H, width=W)
    renderer.enable_segmentation_rendering()
    print('starting fovy', m.cam_fovy[cam])
    start = np.r_[m.cam_pos[cam], np.zeros(3), a.fovy]
    quat0 = m.cam_quat[cam].copy()

    def apply(x):
        m.cam_pos[cam] = x[:3]
        dq = np.zeros(4); mujoco.mju_axisAngle2Quat(dq, x[3:6]/(np.linalg.norm(x[3:6])+1e-12), np.linalg.norm(x[3:6]))
        out = np.zeros(4); mujoco.mju_mulQuat(out, quat0, dq); m.cam_quat[cam] = out
        m.cam_fovy[cam] = x[6]

    def render_mask(q):
        d.qpos[env.qadr] = q; mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=a.camera)
        seg = renderer.render()
        ids = seg[..., 0]; kind = seg[..., 1]
        return (kind == mujoco.mjtObj.mjOBJ_GEOM) & arm_geoms[np.clip(ids, 0, m.ngeom-1)] & (ids >= 0)

    def loss(x):
        apply(x); total = 0
        for k, i in enumerate(keep):
            sim = render_mask(poses[i])
            if sim.sum() < 10: total += 120; continue
            # symmetric chamfer distance in pixels
            total += distance[k][sim].mean() + distance_transform_edt(~sim)[masks[i]].mean()
        return total/len(keep)

    # Bricks, tape and the wooden table are red/orange too.
    apply(start)
    if a.sweep and not a.captures:
        # Fixed camera, many frames: anything that differs from the median frame is moving.
        background = np.median(images, axis=0)
        for i in range(len(masks)):
            masks[i] = binary_opening(masks[i] & (np.abs(images[i]-background).sum(axis=2) > 90), iterations=1)
    else:
        # Few frames: keep only colour pixels near where the starting camera puts the gripper.
        for i, q in enumerate(poses):
            masks[i] = binary_opening(masks[i] & binary_dilation(render_mask(q), iterations=a.gate), iterations=1)
    keep = [i for i, mk in enumerate(masks) if mk.sum() > 40]
    distance = [distance_transform_edt(~masks[i]) for i in keep]
    before = loss(start)
    best = start
    for scale in (1, .3):
        result = minimize(loss, best, method='Powell', options={'xtol': 1e-3, 'ftol': 1e-3, 'maxfev': 500},
                          bounds=[(v-.3*scale, v+.3*scale) for v in best[:3]]+[(v-.7*scale, v+.7*scale) for v in best[3:6]]+[(a.fovy-.01, a.fovy+.01)])
        best = result.x
    after = loss(best); apply(best)
    a.output.mkdir(parents=True, exist_ok=True)
    renderer.disable_segmentation_rendering()
    for k, i in enumerate(keep):
        d.qpos[env.qadr] = poses[i]; mujoco.mj_forward(m, d); renderer.update_scene(d, camera=a.camera)
        sim = renderer.render().astype(float)
        blend = (.5*images[i]+.5*sim).astype(np.uint8)
        blend[masks[i]] = [0, 255, 0]
        Image.fromarray(np.hstack([images[i].astype(np.uint8), sim.astype(np.uint8), blend])).save(a.output/f'{a.camera}-{names[i]}.jpg')
    report = {'camera': a.camera, 'captures_used': [names[i] for i in keep], 'chamfer_px_before': before, 'chamfer_px_after': after,
              'image_width': W, 'pos': m.cam_pos[cam].tolist(), 'quat': m.cam_quat[cam].tolist(), 'fovy': float(m.cam_fovy[cam]),
              'real_mask_pixels': [int(masks[i].sum()) for i in keep]}
    (a.output/f'{a.camera}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=1))
    renderer.close(); env.close()


if __name__ == '__main__':
    main()
