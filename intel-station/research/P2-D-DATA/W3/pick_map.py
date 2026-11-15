#!/usr/bin/env python3
"""P2-D-DATA W3: pick map (read-only). Usage: python -I pick_map.py EPISODES_JSON DATA_ROOT OUT_JSON
Events: grasp close (grasp_idx), release (release_idx). Pose = observation.state (6 values) at the event;
hover = pose 0.7 s (21 frames) before the event. Brick pixel (top view 243622061421):
  grasp   = saturated blob LOST between the episode start frame and grasp+1.0 s (brick left the cluster)
  release = saturated blob GAINED between the start frame and release+1.0 s (brick now at the drop place)
Plate centres from the start frame (light-blue, green masks). Fit: IDW and affine pixel->pose, leave-one-out."""
import json, os, subprocess, sys
import numpy as np
import pandas as pd
from PIL import Image

EP_JSON, DATA_ROOT, OUT_JSON = sys.argv[1], sys.argv[2], sys.argv[3]
FPS = 30
TOP = '243622061421'
SIDE = '243722064316'
TMP = os.path.join(os.path.dirname(OUT_JSON), 'pm_frames')
os.makedirs(TMP, exist_ok=True)
eps = json.load(open(EP_JSON))
ddir = [d for d in os.listdir(DATA_ROOT) if d.startswith('6d6bdce3')][0]
DROOT = os.path.join(DATA_ROOT, ddir)

# frame table (all rows, global index = row position)
df = pd.concat([pd.read_parquet(os.path.join(DROOT, 'data/chunk-000', f)) for f in
                sorted(os.listdir(os.path.join(DROOT, 'data/chunk-000')))], ignore_index=True)
STATE = np.stack(df['observation.state'].to_numpy()).astype(float)


def decode(ep, cam, t_rel, tag):
    vid = ep['video']['realsense ' + cam]
    path = os.path.join(DROOT, 'videos', 'observation.images.realsense ' + cam, 'chunk-%03d' % vid['chunk'],
                        'file-%03d.mp4' % vid['file'])
    png = os.path.join(TMP, 'ep%d_%s_%s.png' % (ep['episode'], cam, tag))
    if not os.path.exists(png):
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
    return [round(float(xs[keep].mean()), 1), round(float(ys[keep].mean()), 1)]


def plate_centres(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    blue = (b > 150) & (b - r > 60) & (g > 110) & (b - g < 60)
    green = (g > 110) & (g - r > 50) & (g - b > 30)
    out = {}
    for name, m in [('BLUE', blue), ('GREEN', green)]:
        ys, xs = np.nonzero(m)
        out[name] = [round(float(xs.mean()), 1), round(float(ys.mean()), 1), int(len(xs))] if len(xs) > 500 else None
    return out


def pixel_diff(ep, t_event, which):
    a = decode(ep, TOP, 0.0, 'start')
    b = decode(ep, TOP, t_event + 1.0, 'after%.2f' % t_event)
    if which == 'lost':
        m = (sat(a) > 70) & (sat(b) < 40)
    else:
        m = (sat(b) > 70) & (sat(a) < 40)
    return blob(m), a


events = []
plates = None
for ep in eps:
    e = ep['episode']
    g0 = ep['global_from']
    if plates is None and e == 0:
        pass
    for kind, idx in [('grasp', ep.get('grasp_idx')), ('release', ep.get('release_idx'))]:
        if idx is None:
            continue
        pose = STATE[g0 + idx].tolist()
        hover = STATE[g0 + max(idx - 21, 0)].tolist()
        t_s = idx / FPS
        px, start_img = pixel_diff(ep, t_s, 'lost' if kind == 'grasp' else 'gained')
        events.append({'episode': e, 'kind': kind, 'frame': int(idx), 'event_t_s': round(t_s, 3),
                       'pose6': [round(v, 3) for v in pose], 'hover_0p7s_pose6': [round(v, 3) for v in hover],
                       'top_pixel': px, 'verdict_note': 'pixel from frame difference; manual check needed'})
        if plates is None:
            plates = plate_centres(start_img)

# plate assignment for releases from the top view
for ev in events:
    if ev['kind'] == 'release' and ev['top_pixel'] and plates:
        d = {k: (np.hypot(ev['top_pixel'][0] - v[0], ev['top_pixel'][1] - v[1]) if v else None)
             for k, v in plates.items()}
        near = min([k for k in d if d[k] is not None], key=lambda k: d[k], default=None)
        ev['nearest_plate'] = near
        ev['plate_distance_px'] = {k: (round(float(v), 1) if v is not None else None) for k, v in d.items()}

# fit pixel(top view) -> arm pose (5 arm joints), leave-one-out, on grasp events with a pixel
G = [ev for ev in events if ev['kind'] == 'grasp' and ev['top_pixel']]


def idw(train_x, train_y, q, p=2.0):
    d = np.linalg.norm(train_x - q, axis=1) + 1e-6
    w = 1.0 / d ** p
    return (w[:, None] * train_y).sum(0) / w.sum()


def affine(train_x, train_y, q):
    A = np.hstack([train_x, np.ones((len(train_x), 1))])
    coef, *_ = np.linalg.lstsq(A, train_y, rcond=None)
    return np.append(q, 1.0) @ coef


fits = {}
if len(G) >= 3:
    X = np.array([ev['top_pixel'] for ev in G], float)
    Y = np.array([ev['pose6'][:5] for ev in G], float)
    for name, fn in [('IDW_p2', idw), ('affine', affine)]:
        errs = []
        for i in range(len(G)):
            m = np.arange(len(G)) != i
            pred = fn(X[m], Y[m], X[i])
            errs.append(np.abs(pred - Y[i]))
        errs = np.array(errs)
        fits[name] = {'n': len(G), 'LOO_mean_abs_error_per_joint': [round(float(v), 2) for v in errs.mean(0)],
                      'LOO_mean_abs_error_all_joints': round(float(errs.mean()), 2)}
    fits['note'] = 'grasp samples only; the pixel is the source of the lost brick. Leave-one-out per joint in joint units.'

result = {'events': events, 'plates_top_view_px': plates, 'fits_grasp_pixel_to_arm_pose': fits}
json.dump(result, open(OUT_JSON, 'w'), indent=1)
print('PICKMAP_OK', len(events), 'events; grasp samples', len(G))
