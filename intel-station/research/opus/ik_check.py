import sys, math, time, numpy as np
sys.path.insert(0, '/home/intel/intel-junction-2026/app')
from pick_v2.kinematics import Chain, FollowerCalibration, solve_top_down
U = '/home/intel/intel-junction-2026/repos/edge-developer-kit-reference-scripts/samples/ai/edge-ai-demo-studio/workers/services/robotics-ai/platforms/so101/urdf/so101.urdf'
ch = Chain(U); cal = FollowerCalibration.from_studio_json('/home/intel/intel-junction-2026/app/motion/project_robots.json', 'SO101_Follower')
lo, hi = cal.rad_bounds(ch, margin_units=3)
def fk(n): return ch.fk(cal.rad_from_norm(n[:5]))
for nm, q in [('hover_al', [-12.62, 30.0, -19.33, 74.43, -57.25]), ('grasp', [-12.62, 30.0, 5.47, 50.33, -57.25])]:
    T = fk(q); p = T[:3, 3]; psi = math.atan2(T[1, 0], T[0, 0])
    t0 = time.time(); r = solve_top_down(ch, cal, p[0], p[1], p[2], psi, lo, hi, q_ref=cal.rad_from_norm(q), warm_start=True)
    n = cal.norm_from_rad(r.q)
    print(nm, 'fk', (p*1000).round(1), 'psi', round(math.degrees(psi),1), 'ik', np.round(n,2), 'err mm', round(r.pos_err_m*1000,2), 'rot', round(r.rot_err_rad,3), 'ms', round((time.time()-t0)*1000))
    r2 = solve_top_down(ch, cal, p[0], p[1], p[2], psi, lo, hi)
    print('   cold', np.round(cal.norm_from_rad(r2.q),2), 'err mm', round(r2.pos_err_m*1000,2), 'rot', round(r2.rot_err_rad,3))
# reachable grid at hover height: TCP z = mat(-38.3) + 57.7
z = (-38.3 + 57.7) / 1000
for x in (0.18, 0.22, 0.26, 0.30, 0.34):
    row = []
    for y in (-0.12, -0.06, 0.0, 0.06, 0.12):
        r = solve_top_down(ch, cal, x, y, z, 0.0, lo, hi)
        row.append('%5.1f' % (r.pos_err_m * 1000))
    print('x=%.2f' % x, ' '.join(row))
