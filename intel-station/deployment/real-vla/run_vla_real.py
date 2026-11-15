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
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home()/'intel-junction-2026/app/motion'))
from so101_motion import HARD_MAX_SPEED, claim_devices, device_ids_for, open_follower_held, resolve_arm  # noqa: E402

W, H = 640, 480
CAMERAS = {'camera1': '/dev/video8', 'camera2': '/dev/video14'}   # overhead, close (colour nodes)
LOW = np.array([-99.9]*5+[0.1])
HIGH = np.array([99.9]*5+[99.9])


class Camera:
    def __init__(self, node):
        self.p = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'v4l2', '-input_format', 'yuyv422',
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
        self.p.kill()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='pick up all the blocks and put them on the blue plate')
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--max-speed', type=float, default=HARD_MAX_SPEED, help="units/s; capped at the driver's own servo estimate")
    p.add_argument('--root', type=Path, default=Path.home()/'junction-cell/real-vla')
    p.add_argument('--policy-python', default=str(Path.home()/'intel-junction-2026/envs/lerobot-env/bin/python'))
    p.add_argument('--policy-stdio', action='store_true')
    a = p.parse_args()
    out = a.root/time.strftime('trial-%Y%m%dT%H%M%S')
    out.mkdir()
    stop = a.root/'STOP'
    model = a.root/'model'
    if a.policy_stdio:   # the worker runs elsewhere (Mac) and is piped to our stdin/stdout; logs go to stderr
        worker, to_worker, from_worker = None, sys.stdout, sys.stdin
        sys.stdout = sys.stderr
    else:
        worker = subprocess.Popen([a.policy_python, '-u', str(a.root/'policy_worker.py'), '--checkpoint', str(model/'pretrained_model'),
                                   '--contract', str(model/'contract.json'), '--device', 'cpu'],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=(out/'worker.err').open('w'), text=True, bufsize=1)
        to_worker, from_worker = worker.stdin, worker.stdout

    def response():
        for line in from_worker:
            if line.startswith('VLA_RESPONSE '):
                r = json.loads(line[13:])
                if 'error' in r:
                    raise RuntimeError(r['error'])
                return r
        raise RuntimeError('policy worker exited')

    ready = response()
    print('policy ready', ready['checkpoint'], flush=True)
    cams = {k: Camera(v) for k, v in CAMERAS.items()}
    time.sleep(2)
    log = {'task': a.task, 'seconds': a.seconds, 'max_speed_units_s': HARD_MAX_SPEED, 'chunks': [], 'samples': []}
    port, cal = resolve_arm(studio_export=str(Path.home()/'intel-junction-2026/app/motion/project_robots.json'), name='SO101 Follower', follower=True)
    with claim_devices(device_ids_for(port), ports=(port,)):
        drv = open_follower_held(port, cal)
        try:
            goal = np.asarray(drv.get_observation().joint_positions, float)
            log['initial_state'] = goal.tolist()
            speed = min(a.max_speed, float(drv.max_speed))
            log['max_speed_units_s'] = speed
            print(f'speed limit {speed:.0f} units/s', flush=True)
            step_limit, start, first = speed/30, time.monotonic(), True
            while time.monotonic()-start < a.seconds and not stop.exists():
                state = np.asarray(drv.get_observation().joint_positions, np.float32)
                buf = io.BytesIO()
                np.savez_compressed(buf, state=state, **{k: c.latest() for k, c in cams.items()})
                to_worker.write(json.dumps({'observation': base64.b64encode(buf.getvalue()).decode(), 'task': a.task, 'reset': first,
                                               'seed': 42, 'action_units': ready['action_units'], 'mapping_sha256': ready['mapping_sha256']})+'\n')
                to_worker.flush()
                r, first = response(), False
                chunk = np.asarray(r['actions'], float)
                log['chunks'].append({'t': time.monotonic()-start, 'state': state.tolist(), 'inference_s': r['inference_seconds']})
                print(f"t={time.monotonic()-start:5.1f}s inference {r['inference_seconds']:.1f}s state {np.round(state, 1).tolist()}", flush=True)
                for raw in chunk:
                    if time.monotonic()-start >= a.seconds or stop.exists():
                        break
                    tick = time.monotonic()
                    goal = goal+np.clip(np.clip(raw, LOW, HIGH)-goal, -step_limit, step_limit)
                    drv.send_action(goal.astype(np.float32))
                    log['samples'].append({'t': tick-start, 'raw': raw.tolist(), 'goal': goal.tolist()})
                    time.sleep(max(0., 1/30-(time.monotonic()-tick)))
            log['stopped_by'] = 'STOP file' if stop.exists() else 'time limit'
        finally:
            present = np.asarray(drv._read_joint_positions(), dtype=np.int32)
            drv._write_joint_positions(present)              # hold where it is
            log['final_state'] = np.asarray(drv.get_observation().joint_positions, float).tolist()
            drv.disconnect()                                 # follower: torque stays on
            for c in cams.values():
                c.close()
            to_worker.close()
            worker and worker.terminate()
            (out/'trial.json').write_text(json.dumps(log, indent=1))
            print('arm holding, log', out/'trial.json', flush=True)


if __name__ == '__main__':
    main()
