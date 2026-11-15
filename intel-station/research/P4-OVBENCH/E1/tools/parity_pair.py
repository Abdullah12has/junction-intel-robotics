# Read-only parity between two raw npz outputs. Usage: parity_pair.py REF_TAG REF_DEV CAND_TAG CAND_DEV
import os
import sys

import numpy as np

E = '/home/intel/intel-junction-2026/work/P4-OVBENCH/E1'
ta, da, tb, db = sys.argv[1:5]


def load(tag, dev):
    z = np.load(os.path.join(E, 'raw', '%s__%s__out.npz' % (tag, dev)))
    return {k: z[k].astype(np.float64) for k in z.files}


A = load(ta, da)
B = load(tb, db)
keys = sorted(set(A) & set(B))
ref = max(float(np.abs(A[k]).max()) for k in keys)
n = 0
s = 0.0
mx = 0.0
gt = 0
for k in keys:
    d = np.abs(A[k] - B[k])
    n += d.size
    s += float(d.sum())
    mx = max(mx, float(d.max()))
    gt += int((d > 0.01 * ref).sum())
print('ref=%s-%s cand=%s-%s keys=%d elements=%d ref_absmax=%.6g max_abs=%.6g mean_abs=%.6g n_gt_1pct=%d'
      % (ta, da, tb, db, len(keys), n, ref, mx, s / n, gt))
