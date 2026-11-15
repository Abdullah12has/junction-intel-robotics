"""Read-only: follower ticks, torque flags, one colour frame per RealSense. No writes."""
import json, subprocess, sys, time
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
port = '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00'
if subprocess.run(['fuser', port], capture_output=True).stdout.strip():
    raise SystemExit('Follower port is owned; not opening')
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
bus = FeetechMotorsBus(port, {n: Motor(i+1, 'sts3215', MotorNormMode.RANGE_M100_100) for i, n in enumerate(names)})
bus.connect(handshake=False)
try:
    samples = []
    for _ in range(5):
        samples.append(bus.sync_read('Present_Position', normalize=False)); time.sleep(.05)
    torque = bus.sync_read('Torque_Enable', normalize=False)
finally:
    bus.disconnect(disable_torque=False)
result = {'ticks': {n: [int(s[n]) for s in samples] for n in names}, 'torque_enable': {n: int(torque[n]) for n in names}}
try:
    import numpy as np, pyrealsense2 as rs
    from PIL import Image
    for serial in ('243622061421', '243722064316'):
        pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        pipe.start(cfg)
        try:
            for _ in range(20): frame = pipe.wait_for_frames(5000).get_color_frame()
            Image.fromarray(np.asanyarray(frame.get_data())).save(out/f'{serial}.jpg', quality=85)
        finally: pipe.stop()
    result['cameras'] = 'ok'
except Exception as e:
    result['cameras'] = repr(e)
(out/'pose.json').write_text(json.dumps(result, indent=1))
print(json.dumps(result))
