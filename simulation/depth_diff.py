"""Compare real depth frames with the sim rendered at the same joint pose.

Input: a capture directory with pose.json (raw ticks, intrinsics) and frames.npz
(colour and depth in metres, aligned to colour, for 'overview' and 'front').
Reports, per camera, the table plane seen by the real camera against the sim camera
(height above the table and tilt), and the depth error over the image.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from sorting_env import BrickSortingEnv
from station_coordinates import StationCoordinates

CAMERAS = {'overview': 'overhead', 'front': 'wrist'}


def plane(points, iterations=300, tolerance=.004, seed=0):
    """RANSAC plane: unit normal n and offset d with n.p + d = 0, normal facing the camera."""
    rng, best = np.random.default_rng(seed), (0, None)
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b-a, c-a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inliers = np.abs((points-a) @ n) < tolerance
        if inliers.sum() > best[0]:
            best = (inliers.sum(), inliers)
    p = points[best[1]]
    centre = p.mean(0)
    n = np.linalg.svd(p-centre)[2][-1]
    if n[2] > 0:
        n = -n            # camera looks along +z; the visible side of the table faces back toward it
    return n, -n @ centre, best[0]/len(points)


def cloud(depth, k):
    v, u = np.nonzero((depth > .15) & (depth < 1.5))
    z = depth[v, u]
    return np.column_stack(((u-k['ppx'])/k['fx']*z, (v-k['ppy'])/k['fy']*z, z))


def main(directory):
    directory = Path(directory)
    pose = json.loads((directory/'pose.json').read_text())
    frames = np.load(directory/'frames.npz')
    env = BrickSortingEnv(render=False)
    env.reset(1, reference=True)
    m, d = env.model, env.data
    for block in env.blocks:              # brick layout differs from the sim's; hide them
        m.geom_rgba[block['geoms'], 3] = 0
    c = StationCoordinates(env.limits)
    ticks = np.array(list(pose['ticks'].values()), float)
    d.qpos[env.qadr] = np.clip(ticks*c.slope+c.offset, env.limits[:, 0], env.limits[:, 1])
    mujoco.mj_forward(m, d)
    renderer = mujoco.Renderer(m, height=480, width=640)
    report = {}
    for theirs, ours in CAMERAS.items():
        k = pose['intrinsics'][theirs]
        real = frames['depth_'+theirs]
        renderer.enable_depth_rendering(); renderer.update_scene(d, camera=ours); sim = renderer.render().copy(); renderer.disable_depth_rendering()
        renderer.update_scene(d, camera=ours); rgb = renderer.render().copy()
        sim_k = dict(k, fx=240/np.tan(np.deg2rad(m.cam_fovy[m.camera(ours).id])/2), ppx=320, ppy=240)
        sim_k['fy'] = sim_k['fx']
        out = {}
        for label, depth, kk in (('real', real, k), ('sim', sim, sim_k)):
            n, dist, inlier = plane(cloud(depth, kk)[::7])
            out[label] = {'camera_height_above_table_m': round(float(abs(dist)), 4),
                          'tilt_from_straight_down_deg': round(float(np.rad2deg(np.arccos(abs(n[2])))), 2),
                          'table_fraction': round(float(inlier), 2)}
        valid = (real > .15) & (real < 1.5) & (sim > .15) & (sim < 1.5)
        error = (sim-real)[valid]
        out['depth_error_mm'] = {'median': round(float(np.median(error)*1000), 1), 'median_abs': round(float(np.median(np.abs(error))*1000), 1),
                                 'p90_abs': round(float(np.percentile(np.abs(error), 90)*1000), 1), 'valid_pixel_fraction': round(float(valid.mean()), 2)}
        report[ours] = out
        shade = lambda z: np.clip((z-.1)/1.0*255, 0, 255).astype(np.uint8)
        diff = np.zeros((480, 640, 3), np.uint8)
        e = np.where(valid, sim-real, 0)
        diff[..., 0] = np.clip(e*4000, 0, 255); diff[..., 2] = np.clip(-e*4000, 0, 255)   # red: sim farther, blue: sim nearer
        Image.fromarray(np.vstack([np.hstack([frames['color_'+theirs], rgb]),
                                   np.hstack([np.dstack([shade(real)]*3), diff])])).resize((960, 720)).save(directory/f'depth-diff-{ours}.jpg', quality=80)
    (directory/'depth-diff.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
