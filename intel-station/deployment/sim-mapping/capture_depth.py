"""Read-only: follower ticks plus colour and depth (aligned to colour) from both D415s. No commands."""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
port = '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00'
if subprocess.run(['fuser', port], capture_output=True).stdout.strip(): raise SystemExit('Follower port is owned; not opening')
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
bus = FeetechMotorsBus(port, {n: Motor(i+1, 'sts3215', MotorNormMode.RANGE_M100_100) for i, n in enumerate(names)})
bus.connect(handshake=False)
try:
    ticks = bus.sync_read('Present_Position', normalize=False); torque = bus.sync_read('Torque_Enable', normalize=False)
finally:
    bus.disconnect(disable_torque=False)
result = {'ticks': {n: int(ticks[n]) for n in names}, 'torque_enable': {n: int(torque[n]) for n in names}, 'intrinsics': {}}
import pyrealsense2 as rs
arrays = {}
for name, serial in (('overview', '243622061421'), ('front', '243722064316')):
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30); cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg); align = rs.align(rs.stream.color)
    try:
        for _ in range(30): frames = align.process(pipe.wait_for_frames(5000))
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        arrays['color_'+name] = np.asanyarray(frames.get_color_frame().get_data()).copy()
        arrays['depth_'+name] = (np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32)*scale)
        i = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
        result['intrinsics'][name] = {'fx': i.fx, 'fy': i.fy, 'ppx': i.ppx, 'ppy': i.ppy, 'width': i.width, 'height': i.height}
    finally: pipe.stop()
np.savez_compressed(out/'frames.npz', **arrays); (out/'pose.json').write_text(json.dumps(result, indent=1)); print(json.dumps(result))
