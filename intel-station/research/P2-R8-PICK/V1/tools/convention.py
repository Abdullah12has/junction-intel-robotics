# V1 verifier: convention check. Searches joint signs (5 body joints) and 0/+-90 deg offsets (lift, elbow, wrist_flex)
# against the verified grasp pairs (affine LOO) and checks home and plate poses. The identity convention is the URDF/physicalai one.
import itertools, json, math, sys
import numpy as np
sys.path.insert(0, "/home/intel/intel-junction-2026/work/P2-R8-PICK/V1/tools")
from fit import LABELS, R, rows, joints, chain, fit_affine, apply_affine, inv_affine, fk, q_of, BODY  # noqa: E402

OUT = "/home/intel/intel-junction-2026/work/P2-R8-PICK/V1"
EPS = sorted(LABELS)
PX = [LABELS[e] for e in EPS]
KNOWN = ["home_facts_13:24", "blue_ep7_release_facts", "green_ep8_release_facts", "green_ep10_release_facts"]


def tcp_xy(u, s, off):
    q0 = q_of(np.asarray(u[:5], float))
    q = {n: s[j] * q0[n] + off[j] for j, n in enumerate(BODY)}
    return fk(joints, chain, q)[:2, 3]


def loo(xy_list):
    errs = []
    for k in range(len(EPS)):
        idx = [i for i in range(len(EPS)) if i != k]
        M = fit_affine([PX[i] for i in idx], [xy_list[i] for i in idx])
        errs.append(1000 * np.linalg.norm(apply_affine(M, PX[k]) - xy_list[k]))
    return errs


results = []
offs_choices = [-math.pi / 2, 0.0, math.pi / 2]
for s in itertools.product([1.0, -1.0], repeat=5):
    for o_l, o_e, o_w in itertools.product(offs_choices, repeat=3):
        off = [0.0, o_l, o_e, o_w, 0.0]
        xy = [tcp_xy(R[e]["state_grasp"], s, off) for e in EPS]
        xy = [np.asarray(v) for v in xy]
        errs = loo(xy)
        M = fit_affine(PX, xy)
        resid = [1000 * np.linalg.norm(apply_affine(M, PX[i]) - xy[i]) for i in range(len(EPS))]
        home = tcp_xy(rows["known_poses"][KNOWN[0]]["normalized"], s, off)
        results.append({"signs": list(s), "offsets_deg": [round(math.degrees(v)) for v in off],
                        "loo_median_mm": round(float(np.median(errs)), 2), "loo_max_mm": round(float(np.max(errs)), 2),
                        "in_sample_median_mm": round(float(np.median(resid)), 2),
                        "home_px_affine": [round(float(v), 1) for v in inv_affine(M, home)]})
results.sort(key=lambda r: r["loo_median_mm"])
ident = [r for r in results if r["signs"] == [1.0] * 5 and r["offsets_deg"] == [0, 0, 0, 0, 0]][0]
print("identity", ident)
print("n combos", len(results))
for r in results[:12]:
    print(r)
json.dump({"identity": ident, "top": results[:30], "n": len(results)}, open(OUT + "/convention_search.json", "w"), indent=1)
