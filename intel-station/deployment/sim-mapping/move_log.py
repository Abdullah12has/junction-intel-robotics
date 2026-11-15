"""Slow, guarded pass through the planned waypoints, logging commanded and measured positions at ~25 Hz.

Same guards as move_capture.py: port must be unowned; position mode only; <= 8 ticks per joint per 40 ms step;
abort-and-hold on > 300 ticks lag for 0.6 s, > 60 C, or /tmp/arm_stop. RAM registers only. Ends at rest, torque off.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
plan = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2])
port = '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00'
if subprocess.run(['fuser', port], capture_output=True).stdout.strip(): raise SystemExit('Follower port is owned; not opening')
if os.path.exists('/tmp/arm_stop'): raise SystemExit('/tmp/arm_stop exists')
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
N = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
LO, HI = np.array(plan['tick_min'])+20, np.array(plan['tick_max'])-20
bus = FeetechMotorsBus(port, {n: Motor(i+1, 'sts3215', MotorNormMode.RANGE_M100_100) for i, n in enumerate(N)})
bus.connect(handshake=False)
read = lambda reg: np.array([bus.sync_read(reg, normalize=False)[n] for n in N], float)
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
rows = []
class Abort(Exception): pass

def step(goal, label, i=0, lag=[0]):
    t0 = time.monotonic()
    bus.sync_write('Goal_Position', {n: int(round(g)) for n, g in zip(N, goal)}, normalize=False)
    pos = read('Present_Position'); rows.append([time.time(), *np.round(goal), *pos])
    lag[0] = lag[0]+1 if np.abs(goal-pos).max() > 300 else 0
    if lag[0] > 15 or (i % 10 == 0 and os.path.exists('/tmp/arm_stop')):
        hold = read('Present_Position'); bus.sync_write('Goal_Position', {n: int(h) for n, h in zip(N, hold)}, normalize=False)
        raise Abort(f'{label}: stopped, lag {np.abs(goal-pos).max():.0f} ticks; holding position, torque ON')
    time.sleep(max(0, .04-(time.monotonic()-t0)))

def move(target, label, hold_s=1.5):
    target = np.array(target, float); start = np.array(rows[-1][1:7]) if rows else read('Present_Position')
    steps = int(np.ceil(np.max(np.abs(target-start))/8)) or 1
    for i in range(1, steps+1):
        s = i/steps; step(start+(target-start)*(s*s*(3-2*s)), label, i)
    for i in range(int(hold_s/.04)): step(target, label, i+1)
    log(f'{label}: residual ticks {np.round(read("Present_Position")-target).astype(int).tolist()}')

status = 'ok'
try:
    mode, temp = read('Operating_Mode'), read('Present_Temperature')
    log('mode', mode.astype(int).tolist(), 'temp C', temp.astype(int).tolist())
    if np.any(mode != 0) or temp.max() > 55: raise SystemExit('Unexpected mode or temperature; nothing moved')
    start = read('Present_Position')
    if np.abs(start-np.array(plan['rest_ticks'])).max() > 60: raise SystemExit(f'Arm is not at the planned rest pose ({start.astype(int).tolist()}); nothing moved')
    ways = [(w['name'], np.clip(w['ticks'], LO, HI)) for w in plan['waypoints']]
    bus.sync_write('Goal_Position', {n: int(s) for n, s in zip(N, start)}, normalize=False)
    bus.enable_torque(); log('torque on')
    try:
        for name, ticks in ways:
            move(ticks, name)
            if name == 'centre':          # gripper cycles in the air: closed, wide open, closed, back to the planned opening
                for g in (2060, 2700, 2060, ticks[5]):
                    t = ticks.copy(); t[5] = g; move(t, f'gripper to {int(g)}', hold_s=1.0)
        for name, ticks in reversed(ways[:-1]): move(ticks, 'back to '+name)
        move(np.array(plan['rest_ticks']), 'rest')
        bus.disable_torque(); log('at rest, torque off')
    except Abort as e:
        status = 'ABORTED: '+str(e); log(status)
finally:
    log('final ticks', read('Present_Position').astype(int).tolist(), 'torque', read('Torque_Enable').astype(int).tolist(), 'temp', read('Present_Temperature').astype(int).tolist())
    bus.disconnect(disable_torque=False)
    np.savetxt(out, np.array(rows), delimiter=',', header='t,'+','.join(f'goal{i}' for i in range(6))+','+','.join(f'pos{i}' for i in range(6)), comments='', fmt='%.4f')
print('RESULT', status, len(rows), 'rows')
