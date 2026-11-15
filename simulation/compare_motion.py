"""Replay a real joints.csv (t, goal0-5, pos0-5) through the sim and plot sim against real per joint."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from fit_dynamics import JOINTS, Replay, load_csv


def trace(replay, t, goal, position, delay):
    env, d = replay.env, replay.env.data
    goal, real = replay.radians(goal), replay.radians(position)
    mujoco.mj_resetData(replay.m, d)
    d.qpos[env.qadr] = real[0]
    d.ctrl[env.actuator_ids] = real[0]
    mujoco.mj_forward(replay.m, d)
    sent = np.searchsorted(t, t[:-1]-delay, side='right')-1
    steps = np.maximum(1, np.round(np.minimum(np.diff(t), .5)*600).astype(int))
    out = [real[0].copy()]
    for i in range(1, len(t)):
        d.ctrl[env.actuator_ids] = goal[max(sent[i-1], 0)]
        mujoco.mj_step(replay.m, d, nstep=steps[i-1])
        out.append(d.qpos[env.qadr].copy())
    return np.rad2deg(goal), np.rad2deg(real), np.rad2deg(np.array(out))


def main(path, output, settings):
    t, goal, position = load_csv(path)
    replay = Replay()
    fitted = replay.scales.copy()
    cases = {'default sim motors': (np.ones((5, 4)), 0.), 'current sim motors': (fitted, .07)}
    for label, file in settings:
        joints = json.loads(Path(file).read_text())['joints']
        cases[label] = (np.array([[joints[j][k] for k in ('kp', 'kv', 'damping', 'frictionloss')] for j in JOINTS]), .07)
    results = {}
    for label, (scales, delay) in cases.items():
        replay.set(scales)
        g, real, sim = trace(replay, t, goal, position, delay)
        results[label] = sim
        error = sim[:, :5]-real[:, :5]
        print(f'{label:22s} rms deg {np.sqrt((error**2).mean(0)).round(2).tolist()} mean {np.sqrt((error**2).mean(0)).mean():.2f} | worst deg {np.abs(error).max(0).round(1).tolist()}')
    print(f'{"real: command vs arm":22s} rms deg {np.sqrt(((g-real)[:, :5]**2).mean(0)).round(2).tolist()}')
    # plot: one panel per arm joint; grey command, black real, colours sim
    W, H, colours = 1500, 230, [(220, 60, 50), (40, 110, 230), (30, 160, 80)]
    image = Image.new('RGB', (W, H*5+30), 'white')
    draw = ImageDraw.Draw(image)
    for j, name in enumerate(JOINTS):
        y0 = j*H+10
        series = [g[:, j], real[:, j]]+[s[:, j] for s in results.values()]
        lo, hi = min(s.min() for s in series)-2, max(s.max() for s in series)+2
        xy = lambda s: [(40+(W-60)*t[i]/t[-1], y0+H-20-(H-40)*(s[i]-lo)/(hi-lo)) for i in range(0, len(t), 2)]
        draw.rectangle([40, y0+10, W-20, y0+H-20], outline=(200, 200, 200))
        draw.line(xy(g[:, j]), fill=(170, 170, 170), width=1)
        for k, s in enumerate(results.values()):
            draw.line(xy(s[:, j]), fill=colours[k % 3], width=2)
        draw.line(xy(real[:, j]), fill=(0, 0, 0), width=2)
        draw.text((46, y0+12), f'{name}  ({lo:.0f} to {hi:.0f} deg)', fill=(0, 0, 0))
    legend = 'black: real arm   grey: command   ' + '   '.join(f'{("red", "blue", "green")[k % 3]}: {label}' for k, label in enumerate(results))
    draw.text((46, H*5+12), legend+f'   |   {t[-1]:.0f} s', fill=(0, 0, 0))
    image.save(output, quality=88)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], [x.split('=', 1) for x in sys.argv[3:]])
