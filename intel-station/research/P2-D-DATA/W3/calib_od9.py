#!/usr/bin/env python3
"""P2-D-DATA W3, OD-9 calibration (read-only input).
Inspection spot = where the S1 brick is put down (team eps 3-6 pattern).
Usage: python -I calib_od9.py EPISODES_JSON DATA_ROOT OUT_JSON
Writes poses (arm joints as stored) and the placed-brick image position in both RealSense views."""
import json, os, subprocess, sys
import numpy as np
from PIL import Image

EP_JSON, DATA_ROOT, OUT_JSON = sys.argv[1], sys.argv[2], sys.argv[3]
FPS = 30
CAMS = ['243722064316', '243622061421']
CAM_NAME = {c: 'realsense ' + c for c in CAMS}
ARM = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos']
TMP = os.path.join(os.path.dirname(OUT_JSON), 'od9_frames')
os.makedirs(TMP, exist_ok=True)
eps = {r['episode']: r for r in json.load(open(EP_JSON))}
ddir = [d for d in os.listdir(DATA_ROOT) if d.startswith('6d6bdce3')][0]
DROOT = os.path.join(DATA_ROOT, ddir)


def arm_pose(v):
    return None if v is None else [round(float(x), 3) for x in v]


def decode(ep, cam, t_rel, tag):
    vid = ep['video'][CAM_NAME[cam]]
    path = os.path.join(DROOT, 'videos', 'observation.images.' + CAM_NAME[cam], 'chunk-%03d' % vid['chunk'],
                        'file-%03d.mp4' % vid['file'])
    png = os.path.join(TMP, 'ep%d_%s_%s.png' % (ep['episode'], cam, tag))
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', '%.4f' % (vid['from_ts'] + t_rel), '-i', path,
                    '-frames:v', '1', '-y', png], capture_output=True)
    return np.asarray(Image.open(png).convert('RGB')).astype(float)


def sat(x):
    return x.max(axis=2) - x.min(axis=2)


def blob(mask, cell=16, radius=60):
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    H, W = mask.shape
    grid = np.zeros((H // cell + 1, W // cell + 1), int)
    np.add.at(grid, (ys // cell, xs // cell), 1)
    gy, gx = np.unravel_index(np.argmax(grid), grid.shape)
    cy, cx = gy * cell + cell / 2, gx * cell + cell / 2
    keep = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
    px, py = xs[keep].astype(float), ys[keep].astype(float)
    return {'centre_px': [round(float(px.mean()), 1), round(float(py.mean()), 1)],
            'spread_px_std': [round(float(px.std()), 1), round(float(py.std()), 1)],
            'pixels': int(keep.sum())}


def placed_brick(ep):
    """Destination = saturated pixels that appear between the first and last frame (arm at home in both)."""
    last_t = (ep['data_rows'] - 1) / FPS - 0.05
    out = {}
    for cam in CAMS:
        a = decode(ep, cam, 0.0, 'first')
        b = decode(ep, cam, last_t, 'last')
        gained = (sat(b) > 70) & (sat(a) < 40)
        lost = (sat(a) > 70) & (sat(b) < 40)
        out[cam] = {'size_wh': [int(a.shape[1]), int(a.shape[0])],
                    'gained_destination': blob(gained), 'lost_source': blob(lost)}
    return out


result = {'note': 'OD-9 calibration. Joint values as stored (arm joints in joints list). Source: episodes.json (release_arm, end_arm, start_arm) and decoded frames.',
          'joints': ARM}

# Inspection spot: put-down pose = release_arm of eps 3-6 (team pattern)
spot_eps = [3, 4, 5, 6]
pose_arr = np.array([eps[e]['release_arm'] for e in spot_eps], float)
result['spot_release_pose'] = {str(e): arm_pose(eps[e]['release_arm']) for e in spot_eps}
result['spot_release_pose_median'] = arm_pose(np.median(pose_arr, axis=0))
result['spot_release_pose_range'] = arm_pose(pose_arr.max(axis=0) - pose_arr.min(axis=0))
result['spot_release_pose_note'] = ('eps 3-6 release over the cluster per manual.json; the put-down pose is the release event, '
                                    'so the spot is the team placement region, not a marked place')

# Placed-brick image position (eps 3, 5, 6 return home; ep 4 does not)
brick = {str(e): placed_brick(eps[e]) for e in [3, 5, 6]}
result['placed_brick_image'] = brick
summary = {}
for c in CAMS:
    pts = [brick[e][c]['gained_destination']['centre_px'] for e in brick if brick[e][c]['gained_destination']]
    if not pts:
        summary[c] = None
        continue
    arr = np.array(pts, float)
    summary[c] = {'centre_px_median': [round(float(np.median(arr[:, 0])), 1), round(float(np.median(arr[:, 1])), 1)],
                  'centre_px_range_x_y': [round(float(np.ptp(arr[:, 0])), 1), round(float(np.ptp(arr[:, 1])), 1)],
                  'n_episodes': len(pts)}
result['placed_brick_image_summary'] = summary

# Drop poses from release events over each plate
result['blue_drop_pose'] = {'ep7_release_arm': arm_pose(eps[7]['release_arm']),
                            'confidence': 'moderate (manual.json: release centred over light-blue plate)'}
gp = np.array([eps[e]['release_arm'] for e in [8, 10]], float)
result['green_drop_pose'] = {'ep8_release_arm': arm_pose(eps[8]['release_arm']),
                             'ep10_release_arm': arm_pose(eps[10]['release_arm']),
                             'median': arm_pose(np.median(gp, axis=0)),
                             'confidence': 'low (plate edge, manual.json)'}

# Home pose: start poses (all 11) and end poses of eps 3, 5, 6 (returned home)
start_all = np.array([eps[e]['start_arm'] for e in eps], float)
ret = np.array([eps[e]['end_arm'] for e in [3, 5, 6]], float)
result['home_pose'] = {'median_start_all_11': arm_pose(np.median(start_all, axis=0)),
                       'end_pose_median_eps_3_5_6': arm_pose(np.median(ret, axis=0)),
                       'start_range_all_11': arm_pose(start_all.max(axis=0) - start_all.min(axis=0))}

# Hover pose ~8 cm above the spot: not derivable from joint data alone
result['hover_pose_8cm'] = {'value': None,
                            'reason': 'needs SO-101 forward kinematics (link lengths) to convert 8 cm to joint space; no link geometry used here. Verify live at slow speed.'}

json.dump(result, open(OUT_JSON, 'w'), indent=1)
print('OD9_OK', OUT_JSON)
