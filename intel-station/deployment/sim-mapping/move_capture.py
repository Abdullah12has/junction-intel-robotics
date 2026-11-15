"""Slow, guarded move through planned waypoints with a depth capture at each. Returns to rest and releases torque.

Guards: port must be unowned; position mode only; <= 8 ticks per joint per 40 ms step (about 18 deg/s);
abort-and-hold if any joint lags its goal by > 300 ticks for 0.6 s, a servo exceeds 60 C, or /tmp/arm_stop exists.
Writes only RAM registers (Goal_Position, Torque_Enable/Lock via the LeRobot bus). No calibration or EEPROM writes.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
plan = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=False)
port = '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00'
if subprocess.run(['fuser', port], capture_output=True).stdout.strip(): raise SystemExit('Follower port is owned; not opening')
if os.path.exists('/tmp/arm_stop'): raise SystemExit('/tmp/arm_stop exists')
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
import pyrealsense2 as rs
N = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
LO, HI = np.array(plan['tick_min'])+20, np.array(plan['tick_max'])-20
bus = FeetechMotorsBus(port, {n: Motor(i+1, 'sts3215', MotorNormMode.RANGE_M100_100) for i, n in enumerate(N)})
bus.connect(handshake=False)
read = lambda reg: np.array([bus.sync_read(reg, normalize=False)[n] for n in N], float)
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)

class Abort(Exception): pass

def move(target, label):
    target = np.array(target, float); start = read('Present_Position'); goal = start.copy()
    steps = int(np.ceil(np.max(np.abs(target-start))/8)) or 1; lag = 0
    for i in range(1, steps+1):
        t0 = time.monotonic(); s = i/steps; s = s*s*(3-2*s)
        goal = start+(target-start)*s
        bus.sync_write('Goal_Position', {n: int(round(g)) for n, g in zip(N, goal)}, normalize=False)
        pos = read('Present_Position'); err = np.abs(goal-pos).max()
        lag = lag+1 if err > 300 else 0
        if lag > 15 or (i % 10 == 0 and os.path.exists('/tmp/arm_stop')):
            hold = read('Present_Position'); bus.sync_write('Goal_Position', {n: int(h) for n, h in zip(N, hold)}, normalize=False)
            raise Abort(f'{label}: stopped at step {i}/{steps}, max lag {err:.0f} ticks; holding position, torque ON')
        time.sleep(max(0, .04-(time.monotonic()-t0)))
    time.sleep(1.2); pos = read('Present_Position')
    log(f'{label}: arrived, residual ticks {np.round(pos-target).astype(int).tolist()}')
    return pos

def capture(name):
    ticks = read('Present_Position'); arrays, intr = {}, {}
    for cam, serial in (('overview', '243622061421'), ('front', '243722064316')):
        pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30); cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipe.start(cfg); align = rs.align(rs.stream.color)
        try:
            for _ in range(25): frames = align.process(pipe.wait_for_frames(5000))
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            arrays['color_'+cam] = np.asanyarray(frames.get_color_frame().get_data()).copy()
            arrays['depth_'+cam] = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32)*scale
            i = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
            intr[cam] = {'fx': i.fx, 'fy': i.fy, 'ppx': i.ppx, 'ppy': i.ppy, 'width': i.width, 'height': i.height}
        finally: pipe.stop()
    after = read('Present_Position'); d = out/name; d.mkdir()
    np.savez_compressed(d/'frames.npz', **arrays)
    (d/'pose.json').write_text(json.dumps({'ticks': dict(zip(N, ticks.astype(int).tolist())), 'ticks_after': after.astype(int).tolist(), 'intrinsics': intr}, indent=1))
    log(f'captured {name}; drift during capture {np.abs(after-ticks).max():.0f} ticks')

status = 'ok'
try:
    mode, pgain, temp, volt = read('Operating_Mode'), read('P_Coefficient'), read('Present_Temperature'), read('Present_Voltage')
    log('mode', mode.astype(int).tolist(), 'P', pgain.astype(int).tolist(), 'temp C', temp.astype(int).tolist(), 'volt', (volt/10).tolist())
    if np.any(mode != 0) or temp.max() > 55: raise SystemExit('Unexpected mode or temperature; nothing moved')
    start = read('Present_Position')
    if np.abs(start-np.array(plan['rest_ticks'])).max() > 60: raise SystemExit(f'Arm is not at the planned rest pose ({start.astype(int).tolist()}); nothing moved')
    ways = [(w['name'], np.clip(w['ticks'], LO, HI)) for w in plan['waypoints']]
    bus.sync_write('Goal_Position', {n: int(s) for n, s in zip(N, start)}, normalize=False)
    bus.enable_torque(); log('torque on, holding rest')
    try:
        for k, (name, ticks) in enumerate(ways):
            move(ticks, name)
            if k > 0:
                try: capture(f'{k:02d}-'+name.replace(' ', '-'))
                except Exception as e: log('capture failed, continuing:', repr(e)[:150])
            if read('Present_Temperature').max() > 60: log('servo above 60 C; returning early'); break
        for name, ticks in reversed(ways[:k]): move(ticks, 'back to '+name)
        move(np.array(plan['rest_ticks']), 'rest')
        bus.disable_torque(); log('at rest, torque off')
    except Abort as e:
        status = 'ABORTED: '+str(e); log(status)
finally:
    log('final ticks', read('Present_Position').astype(int).tolist(), 'torque', read('Torque_Enable').astype(int).tolist(), 'temp', read('Present_Temperature').astype(int).tolist())
    bus.disconnect(disable_torque=False)
print('RESULT', status)
