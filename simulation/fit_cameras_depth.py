"""Refine the sim cameras against real depth frames (render-and-compare).

For each capture (pose.json + frames.npz from the station), the sim arm is put in the
recorded pose and rendered as depth from the current camera estimate. Camera position
and orientation are adjusted to minimise the robust depth difference to the real
frame. Depth pins height and tilt through the table but says little about where the
camera sits along the table, so the loss also includes how well the rendered arm
outline overlaps the real arm outline. The real outline is everything standing more
than 3.5 cm above the real table plane near the arm: geometry from depth, not colour.
Bricks are hidden and bowl pixels ignored: their real positions are not ground truth.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import binary_dilation, binary_opening, distance_transform_edt
from scipy.optimize import minimize

from depth_diff import plane

from sorting_env import BrickSortingEnv
from station_coordinates import StationCoordinates

CAMERAS = {'overhead': 'overview', 'wrist': 'front'}
W, H = 320, 240


def main():
    p = argparse.ArgumentParser()
    p.add_argument('captures', nargs='+', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/camera-fit-depth.json'))
    p.add_argument('--start', type=Path, help='JSON with pos/quat per camera to start from instead of the scene values')
    p.add_argument('--cameras', nargs='+', default=list(CAMERAS))
    a = p.parse_args()
    env = BrickSortingEnv(render=False)
    env.reset(1, reference=True)
    m, d = env.model, env.data
    for block in env.blocks:
        m.geom_rgba[block['geoms'], 3] = 0
    bowl = np.array(['bowl' in m.geom(g).name or 'plate' in m.geom(g).name for g in range(m.ngeom)])
    c = StationCoordinates(env.limits)
    renderer = mujoco.Renderer(m, height=H, width=W)
    poses, real, above = [], {name: [] for name in CAMERAS}, {name: [] for name in CAMERAS}
    for directory in a.captures:
        pose = json.loads((directory/'pose.json').read_text())
        frames = np.load(directory/'frames.npz')
        ticks = np.array(list(pose['ticks'].values()), float)
        poses.append(np.clip(ticks*c.slope+c.offset, env.limits[:, 0], env.limits[:, 1]))
        for ours, theirs in CAMERAS.items():
            k = pose['intrinsics'][theirs]
            # resample the real depth onto an ideal centred pinhole at the sim's resolution
            f = (H/2)/np.tan(np.deg2rad(m.cam_fovy[m.camera(ours).id])/2)
            v, u = np.mgrid[0:H, 0:W]
            x = ((u+.5-W/2)/f*k['fx']+k['ppx']).round().astype(int)
            y = ((v+.5-H/2)/f*k['fy']+k['ppy']).round().astype(int)
            inside = (x >= 0) & (x < 640) & (y >= 0) & (y < 480)
            depth = np.zeros((H, W), np.float32)
            depth[inside] = frames['depth_'+theirs][y[inside], x[inside]]
            real[ours].append(depth)
            # height of every pixel above the real table plane, in the ideal pinhole's camera frame
            ok = (depth > .15) & (depth < 1.5)
            points = np.dstack(((u+.5-W/2)/f*depth, (v+.5-H/2)/f*depth, depth))
            n, dist, _ = plane(points[ok][::3])
            above[ours].append(ok & ((points @ n + dist) > .035))
    robot = (m.body_rootid[m.geom_bodyid] == m.body('base').id) & (m.geom_rgba[:, 3] > 0)
    report = {}
    for ours in a.cameras:
        cam = m.camera(ours).id
        if a.start:
            start = json.loads(a.start.read_text())[ours]
            quat = np.array(start.get('quat_wxyz', start.get('quat')), float)
            m.cam_pos[cam], m.cam_quat[cam] = start['pos'], quat/np.linalg.norm(quat)
        pos0, quat0 = m.cam_pos[cam].copy(), m.cam_quat[cam].copy()
        # Keep only raised pixels near where the starting camera puts the arm: the laptop, camera stand and
        # cables are raised too.
        masks = []
        for q, raised, depth in zip(poses, above[ours], real[ours]):
            d.qpos[env.qadr] = q
            mujoco.mj_forward(m, d)
            renderer.enable_segmentation_rendering(); renderer.update_scene(d, camera=ours); seg = renderer.render()[..., 0].copy(); renderer.disable_segmentation_rendering()
            near = binary_dilation(robot[np.clip(seg, 0, m.ngeom-1)] & (seg >= 0), iterations=30)
            masks.append(binary_opening(raised & near, iterations=1))
        distance = [distance_transform_edt(~mk) if mk.sum() >= 200 else None for mk in masks]

        def apply(x):
            m.cam_pos[cam] = pos0+x[:3]
            dq, out = np.zeros(4), np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, x[3:]/(np.linalg.norm(x[3:])+1e-12), np.linalg.norm(x[3:]))
            mujoco.mju_mulQuat(out, quat0, dq)
            m.cam_quat[cam] = out

        def errors(x):
            apply(x)
            out, chamfer = [], []
            for q, depth, mask, dist in zip(poses, real[ours], masks, distance):
                d.qpos[env.qadr] = q
                mujoco.mj_forward(m, d)
                renderer.enable_depth_rendering(); renderer.update_scene(d, camera=ours); sim = renderer.render().copy(); renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering(); renderer.update_scene(d, camera=ours); seg = renderer.render()[..., 0].copy(); renderer.disable_segmentation_rendering()
                ids = np.clip(seg, 0, m.ngeom-1)
                valid = (depth > .15) & (depth < 1.5) & (sim > .05) & (sim < 1.5) & ~bowl[ids]
                out.append((sim-depth)[valid])
                if dist is not None:
                    rendered = robot[ids] & (seg >= 0) & (depth > .15) & (depth < 1.5)   # only where real depth exists
                    chamfer.append(60. if rendered.sum() < 10 else dist[rendered].mean()+distance_transform_edt(~rendered)[mask].mean())
            return np.concatenate(out), (float(np.mean(chamfer)) if chamfer else 0.)

        def loss(x):
            e, chamfer = errors(x)
            # depth term in mm (capped: bricks, cables and the laptop are not in the sim) + arm outline mismatch in pixels
            return float(np.minimum(np.abs(e), .03).mean()*1000)+chamfer

        before, chamfer_before = errors(np.zeros(6))
        best = np.zeros(6)
        for scale in (1, .3):
            best = minimize(loss, best, method='Powell', options={'xtol': 1e-3, 'ftol': 1e-4, 'maxfev': 700},
                            bounds=[(v-.08*scale, v+.08*scale) for v in best[:3]]+[(v-.15*scale, v+.15*scale) for v in best[3:]]).x
        after, chamfer_after = errors(best)
        apply(best)
        report[ours] = {'pos': m.cam_pos[cam].round(4).tolist(), 'quat_wxyz': m.cam_quat[cam].round(5).tolist(),
                        'fovy_deg': float(m.cam_fovy[cam]), 'captures': len(poses),
                        'moved_cm': (best[:3]*100).round(1).tolist(), 'turned_deg': round(float(np.rad2deg(np.linalg.norm(best[3:]))), 2),
                        'median_abs_depth_error_mm': {'before': round(float(np.median(np.abs(before))*1000), 1),
                                                      'after': round(float(np.median(np.abs(after))*1000), 1)},
                        'arm_outline_mismatch_px_at_320_wide': {'before': round(chamfer_before, 1), 'after': round(chamfer_after, 1),
                                                                'poses_used': sum(x is not None for x in distance)}}
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
