import sys, json, numpy as np
sys.path.insert(0, '/home/intel/intel-junction-2026/app')
from pick_v2.kinematics import Chain, FollowerCalibration
U = '/home/intel/intel-junction-2026/repos/edge-developer-kit-reference-scripts/samples/ai/edge-ai-demo-studio/workers/services/robotics-ai/platforms/so101/urdf/so101.urdf'
ch = Chain(U)
cal = FollowerCalibration.from_studio_json('/home/intel/intel-junction-2026/app/motion/project_robots.json', 'SO101_Follower')
def tip(n):
    T = ch.fk(cal.rad_from_norm(n[:5])); return T[:3, 3] * 1000, T[:3, :3]
base = [-12.62, 30.25, -19.24, 69.73, -45.73]
p0, R0 = tip(base)
print('base tip mm', p0.round(1), 'tool z axis', R0[:, 2].round(2))
for j, nm in enumerate(['pan', 'sl', 'el', 'wf', 'roll']):
    q = list(base); q[j] += 3
    p, _ = tip(q); d = (p - p0) / 3
    print(nm, 'per unit mm', d.round(2), '|xy|', round(float(np.hypot(d[0], d[1])), 2))
for nm, q in [('grasp', [-12.62, 30.0, 5.47, 50.33, -57.25]), ('desc1', [-12.62, 30.0, -7.83, 63.23, -57.25]), ('hover_al', [-12.62, 30.0, -19.33, 74.43, -57.25]), ('park', [-12.62, 18.5, -19.69, 76.85, -45.6]), ('home', [-6, -98, 96.6, 66, -45])]:
    p, R = tip(q); print(nm, p.round(1), 'toolz', R[:, 2].round(2))
