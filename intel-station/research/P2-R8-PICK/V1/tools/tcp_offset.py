# V1 verifier: does a constant jaw-tip offset in the gripper frame explain the grasp residuals? (identity joint convention)
# Offset o (metres, in the gripper_frame_link axes) is added as TCP = p + R @ o. Scan o_x, o_y in [-3 cm, +3 cm]; o_z = 0.
import itertools, json, sys
import numpy as np
sys.path.insert(0, "/home/intel/intel-junction-2026/work/P2-R8-PICK/V1/tools")
from fit import LABELS, R, rows, fk5, fit_affine, apply_affine, inv_affine  # noqa: E402

OUT = "/home/intel/intel-junction-2026/work/P2-R8-PICK/V1"
EPS = sorted(LABELS)
PX = [LABELS[e] for e in EPS]


def tcp_off(u, o):
    T = fk5(u)
    return T[:3, 3] + T[:3, :3] @ np.asarray(o, float)


def loo_errs(xy):
    errs = []
    for k in range(len(EPS)):
        idx = [i for i in range(len(EPS)) if i != k]
        M = fit_affine([PX[i] for i in idx], [xy[i] for i in idx])
        errs.append(1000 * np.linalg.norm(apply_affine(M, PX[k]) - xy[k]))
    return errs


grid = np.arange(-0.03, 0.0301, 0.003)
out = []
for a, b in itertools.product(grid, grid):
    o = [a, b, 0.0]
    xy = [tcp_off(R[e]["state_grasp"][:5], o)[:2] for e in EPS]
    errs = loo_errs(xy)
    M = fit_affine(PX, xy)
    ins = [1000 * np.linalg.norm(apply_affine(M, PX[i]) - xy[i]) for i in range(len(EPS))]
    home = tcp_off(rows["known_poses"]["home_facts_13:24"]["normalized"][:5], o)[:2]
    hp = inv_affine(M, home)
    out.append({"o_xy_cm": [round(a * 100, 1), round(b * 100, 1)], "loo_median_mm": round(float(np.median(errs)), 2),
                "loo_max_mm": round(float(np.max(errs)), 2), "in_sample_median_mm": round(float(np.median(ins)), 2),
                "home_px": [round(float(hp[0]), 1), round(float(hp[1]), 1)]})
out.sort(key=lambda r: r["loo_median_mm"])
ident = [r for r in out if r["o_xy_cm"] == [0.0, 0.0]][0]
print("identity", ident)
for r in out[:12]:
    print(r)
json.dump({"identity": ident, "top": out[:40], "n": len(out)}, open(OUT + "/tcp_offset_search.json", "w"), indent=1)
