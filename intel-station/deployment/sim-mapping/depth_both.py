"""Read-only depth quality test: each D415 alone, then both streaming together, with emitter on. No arm access."""
import numpy as np, pyrealsense2 as rs
S = {'overview': '243622061421', 'front': '243722064316'}
def start(serial):
    p = rs.pipeline(); c = rs.config(); c.enable_device(serial); c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = p.start(c); return p, prof.get_device().first_depth_sensor().get_depth_scale()
def stats(p, scale):
    for _ in range(40): f = p.wait_for_frames(5000)
    zs = []
    for _ in range(10): zs.append(np.asanyarray(p.wait_for_frames(5000).get_depth_frame().get_data()).astype(np.float32)*scale)
    z = np.stack(zs); valid = (z > .15) & (z < 1.5); allv = valid.all(0)
    return f'valid {valid.mean()*100:.0f}%, frame-to-frame noise {np.median(z.std(0)[allv])*1000:.2f} mm'
for name, serial in S.items():
    p, s = start(serial)
    try: print('alone   ', name, stats(p, s))
    finally: p.stop()
pipes = {n: start(s) for n, s in S.items()}
try:
    for n, (p, s) in pipes.items(): print('together', n, stats(p, s))
finally:
    for p, _ in pipes.values(): p.stop()
