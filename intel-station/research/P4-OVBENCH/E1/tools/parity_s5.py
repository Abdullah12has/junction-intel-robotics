# Read-only: pairwise parity of raw ACT outputs. The reference is the first member of each pair. No device is touched.
import os

import numpy as np

E = '/home/intel/intel-junction-2026/work/P4-OVBENCH/E1'


def load(tag, dev):
    z = np.load(os.path.join(E, 'raw', '%s__%s__out.npz' % (tag, dev)))
    return {k: z[k].astype(np.float64) for k in z.files}


PAIRS = [
    (('act_fp32', 'CPU'), ('act_fp16', 'CPU')),
    (('act_fp32', 'CPU'), ('act_fp16_cpu_f32', 'CPU')),
    (('act_fp16', 'CPU'), ('act_fp16_cpu_f32', 'CPU')),
    (('act_fp32', 'CPU'), ('act_fp16', 'GPU')),
    (('act_fp32', 'CPU'), ('act_fp16', 'NPU')),
    (('act_fp32', 'CPU'), ('act_fp32', 'GPU')),
]
print('ref\tcand\tkeys\telements\tref_absmax\tmax_abs\tmean_abs\tn_gt_1pct_of_ref_absmax')
for (ta, da), (tb, db) in PAIRS:
    try:
        A = load(ta, da)
        B = load(tb, db)
    except FileNotFoundError as e:
        print('%s-%s\t%s-%s\tmissing file: %s' % (ta, da, tb, db, e.filename))
        continue
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
    print('%s-%s\t%s-%s\t%d\t%d\t%.6g\t%.6g\t%.6g\t%d' % (ta, da, tb, db, len(keys), n, ref, mx, s / n, gt))
