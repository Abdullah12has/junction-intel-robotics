"""One bounded closed-loop SmolVLA trial on the real SO-101 follower (Intel laptop).

Run with the motion-env Python (it has the Studio SO-101 driver). The policy runs in a
separate LeRobot process (policy_worker.py over stdio, CPU). Both RealSense colour streams
are read with ffmpeg/V4L2. Safety: actions are clipped to the calibrated normalized range,
every joint is rate-limited to HARD_MAX_SPEED units/s, the trial stops after --seconds or
when the STOP file exists, and the arm then holds its present pose with torque on.
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import threading
from queue import Queue, Empty
from visual_servo import bounded_target, validate_chunk
from policy_action_bounds import PolicyActionBounds
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home()/'intel-junction-2026/app/motion'))
from so101_motion import HARD_MAX_SPEED, claim_devices, device_ids_for, open_follower_held, resolve_arm  # noqa: E402

W, H = 640, 480
LOW = np.array([-99.9]*5+[0.1])
HIGH = np.array([99.9]*5+[99.9])


class Camera:
    def __init__(self, node, input_format="yuyv422"):
        node = str(Path(node).resolve(strict=True))
        owner = subprocess.run(['fuser', node], capture_output=True)
        if owner.returncode != 1:
            raise RuntimeError('Camera is owned or ownership check failed: '+node)
        self.p = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'v4l2', '-input_format', input_format,
                                   '-video_size', f'{W}x{H}', '-framerate', '30', '-i', node, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.frame, self.stamp = None, 0.
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        while True:
            buf = self.p.stdout.read(W*H*3)
            if len(buf) < W*H*3:
                return
            self.frame, self.stamp = np.frombuffer(buf, np.uint8).reshape(H, W, 3), time.monotonic()

    def latest(self):
        if self.frame is None or time.monotonic()-self.stamp > .5:
            raise RuntimeError('camera frame missing or stale')
        return self.frame

    def close(self):
        if self.p.poll() is None:
            self.p.terminate()
            try: self.p.wait(timeout=2)
            except subprocess.TimeoutExpired: self.p.kill(); self.p.wait()
        self.p.stdout.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='pick up all the blocks and put them on the blue plate')
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--max-speed', type=float, default=HARD_MAX_SPEED, help="units/s; capped at the driver's own servo estimate")
    p.add_argument('--root', type=Path, default=Path.home()/'junction-cell/real-vla')
    p.add_argument('--policy-python', default=str(Path.home()/'intel-junction-2026/envs/lerobot-env/bin/python'))
    p.add_argument('--policy-stdio', action='store_true')
    p.add_argument('--policy-device', default='xpu')
    p.add_argument('--execute-steps', type=int, default=10)
    p.add_argument('--inference-timeout', type=float, default=5.)
    p.add_argument('--camera1', required=True, help='Verified overhead stable /dev/v4l/by-id path')
    p.add_argument('--camera2', required=True, help='Verified fixed-close stable /dev/v4l/by-id path')
    a = p.parse_args()
    if not 1 <= a.execute_steps <= 50 or not 0 < a.seconds <= 60 or not 0 < a.inference_timeout <= 10 or not np.isfinite(a.max_speed) or a.max_speed <= 0:
        p.error('Invalid trial limits')
    if Path(a.camera1).resolve() == Path(a.camera2).resolve():
        p.error('Camera inputs must be distinct')
    out = a.root/time.strftime('trial-%Y%m%dT%H%M%S')
    out.mkdir()
    stop = a.root/'STOP'
    if stop.exists():
        raise RuntimeError('STOP file exists; no devices opened')
    model = a.root/'model'
    if a.policy_stdio:   # the worker runs elsewhere (Mac) and is piped to our stdin/stdout; logs go to stderr
        worker, to_worker, from_worker = None, sys.stdout, sys.stdin
        sys.stdout = sys.stderr
    else:
        worker = subprocess.Popen([a.policy_python, '-u', str(a.root/'policy_worker.py'), '--checkpoint', str(model/'pretrained_model'),
                                   '--contract', str(model/'contract.json'), '--device', a.policy_device],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=(out/'worker.err').open('w'), text=True, bufsize=1)
        to_worker, from_worker = worker.stdin, worker.stdout

    lines = Queue()
    def read_responses():
        for line in from_worker:
            lines.put(line)
        lines.put(None)
    threading.Thread(target=read_responses, daemon=True).start()
    def response(timeout, started=None):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if stop.exists() or (started is not None and time.monotonic()-started >= a.seconds):
                raise RuntimeError('Trial stopped while awaiting inference')
            try:
                line = lines.get(timeout=min(.1, max(.001, deadline-time.monotonic())))
            except Empty:
                continue
            if line is None:
                raise RuntimeError('Policy worker exited')
            if line.startswith('VLA_RESPONSE '):
                r = json.loads(line[13:])
                if 'error' in r:
                    raise RuntimeError(r['error'])
                return r
        raise TimeoutError('Policy response deadline exceeded')

    ready = response(180)
    contract = json.loads((model/'contract.json').read_text())
    if not ready.get('ready') or ready.get('action_units') != 'so101_station_normalized_v1' or any(ready.get(k) != contract[k] for k in ('action_units','mapping_sha256')):
        raise RuntimeError('Policy coordinate contract mismatch')
    print('policy ready', ready['checkpoint'], flush=True)
    cams = {}
    try:
        for key, node in {'camera1': a.camera1, 'camera2': a.camera2}.items():
            cams[key] = Camera(node, "mjpeg" if key == "camera2" else "yuyv422")
        time.sleep(2)
        log = {'execute_steps': a.execute_steps, 'camera_paths': [a.camera1,a.camera2], 'task': a.task, 'seconds': a.seconds, 'max_speed_units_s': HARD_MAX_SPEED, 'chunks': [], 'samples': []}
        port, cal = resolve_arm(studio_export=str(Path.home()/'intel-junction-2026/app/motion/project_robots.json'), name='SO101 Follower', follower=True)
        with claim_devices(device_ids_for(port), ports=(port,)):
            drv = open_follower_held(port, cal)
            try:
                goal = np.asarray(drv.get_observation().joint_positions, float)
                log['initial_state'] = goal.tolist()
                speed = min(a.max_speed, float(drv.max_speed), HARD_MAX_SPEED)
                log['max_speed_units_s'] = speed
                print(f'speed limit {speed:.0f} units/s', flush=True)
                start, first = time.monotonic(), True
                action_bounds = PolicyActionBounds()
                while time.monotonic()-start < a.seconds and not stop.exists():
                    state = np.asarray(drv.get_observation().joint_positions, np.float32)
                    buf = io.BytesIO()
                    np.savez_compressed(buf, state=state, **{k: c.latest() for k, c in cams.items()})
                    (out/f"observation-{len(log['chunks']):04d}.npz").write_bytes(buf.getvalue())
                    to_worker.write(json.dumps({'observation': base64.b64encode(buf.getvalue()).decode(), 'task': a.task, 'reset': first,
                                                   'seed': 42, 'action_units': ready['action_units'], 'mapping_sha256': ready['mapping_sha256']})+'\n')
                    to_worker.flush()
                    r, first = response(a.inference_timeout, start), False
                    chunk = validate_chunk(r['actions'])
                    bounded, projection = action_bounds.prepare(chunk[:a.execute_steps])
                    log['chunks'].append({'t': time.monotonic()-start, 'state': state.tolist(), 'inference_s': r['inference_seconds'], 'projection': projection})
                    print(f"t={time.monotonic()-start:5.1f}s inference {r['inference_seconds']:.1f}s state {np.round(state, 1).tolist()}", flush=True)
                    for raw, target in zip(chunk[:a.execute_steps], bounded):
                        if time.monotonic()-start >= a.seconds or stop.exists():
                            break
                        tick = time.monotonic()
                        for camera in cams.values(): camera.latest()
                        measured = np.asarray(drv.get_observation().joint_positions, float)
                        if measured.shape != (6,) or not np.isfinite(measured).all() or np.any(measured < LOW) or np.any(measured > HIGH):
                            raise RuntimeError('Measured joints outside calibrated range')
                        if np.max(np.abs(measured-goal)) > 10:
                            raise RuntimeError('Follower tracking error exceeds 10 normalized units')
                        action_bounds.observe(measured, raw, tick)
                        goal = bounded_target(target, goal, speed, 1/30)
                        drv.send_action(goal.astype(np.float32))
                        log['samples'].append({'t': tick-start, 'raw': raw.tolist(), 'goal': goal.tolist()})
                        time.sleep(max(0., 1/30-(time.monotonic()-tick)))
                log['stopped_by'] = 'STOP file' if stop.exists() else 'time limit'
            except Exception as exc:
                log['error'] = str(exc)
                raise
            finally:
                try:
                    present = np.asarray(drv._read_joint_positions(), dtype=np.int32)
                    drv._write_joint_positions(present)
                    log['final_state'] = np.asarray(drv.get_observation().joint_positions, float).tolist()
                    log['hold_confirmed'] = True
                except Exception as hold_error:
                    log['hold_error'] = str(hold_error)
                finally:
                    try: drv.disconnect()
                    finally: (out/'trial.json').write_text(json.dumps(log, indent=1))
                print('trial ended, inspect hold status in', out/'trial.json', flush=True)

    finally:
        for camera in cams.values(): camera.close()
        if worker is not None:
            worker.terminate()
            try: worker.wait(timeout=3)
            except subprocess.TimeoutExpired: worker.kill(); worker.wait()


if __name__ == '__main__':
    main()
