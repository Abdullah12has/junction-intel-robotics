"""Read-only: leader and follower normalized positions from raw ticks + Studio calibration."""
import json, subprocess, urllib.request
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
robots = json.load(urllib.request.urlopen('http://127.0.0.1:7860/api/projects/ab767473-d6fd-47ff-812e-8239c455dabb/robots', timeout=5))
out = {}
for r in robots:
    serial = r['payload']['serial_number']; port = f'/dev/serial/by-id/usb-1a86_USB_Single_Serial_{serial}-if00'
    if subprocess.run(['fuser', port], capture_output=True).stdout.strip(): raise SystemExit(port + ' is owned')
    bus = FeetechMotorsBus(port, {n: Motor(i+1, 'sts3215', MotorNormMode.RANGE_M100_100) for i, n in enumerate(names)})
    bus.connect(handshake=False)
    try: t = bus.sync_read('Present_Position', normalize=False)
    finally: bus.disconnect(disable_torque=False)
    cal = r['payload']['calibration']; norm = {}
    for n in names:
        lo, hi = cal[n]['range_min'], cal[n]['range_max']; f = (t[n]-lo)/(hi-lo)
        norm[n] = round(f*100 if n == 'gripper' else f*200-100, 1)
    out[r['name']] = norm
print(json.dumps(out, indent=1))
