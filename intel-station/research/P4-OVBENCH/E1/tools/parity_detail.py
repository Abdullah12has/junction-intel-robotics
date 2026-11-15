# Read-only comparison of the raw output npz files written by the bench runs. No device is touched.
import glob
import os
import re

import numpy as np

E = '/home/intel/intel-junction-2026/work/P4-OVBENCH/E1'
DEV = re.compile(r'(CPU|GPU|NPU)', re.I)
files = sorted(glob.glob(os.path.join(E, 'raw', '*__out.npz')))
lines = []


def emit(s):
    print(s)
    lines.append(s)


emit('npz files: %d' % len(files))
groups = {}
for f in files:
    b = os.path.basename(f)[:-len('__out.npz')]
    m = DEV.search(b)
    if m is None:
        emit('skipped (no device token): %s' % os.path.basename(f))
        continue
    tag = (b[:m.start()] + '_' + b[m.end():]).strip('_')
    groups.setdefault(tag, {})[m.group(1).upper()] = f
    emit('file %s tag=%s device=%s' % (os.path.basename(f), tag, m.group(1).upper()))


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


for tag in sorted(groups):
    devs = groups[tag]
    data = {d: load(p) for d, p in devs.items()}
    for d in sorted(data):
        for k, a in sorted(data[d].items()):
            if a.dtype.kind not in 'fiub':
                emit('ARRAY\t%s\t%s\t%s\t%s\t%s' % (tag, d, k, a.shape, a.dtype))
                continue
            af = a.astype(np.float64)
            nan = int(np.isnan(af).sum())
            inf = int(np.isinf(af).sum())
            fin = af[np.isfinite(af)]
            amax = float(np.abs(fin).max()) if fin.size else float('nan')
            emit('ARRAY\t%s\t%s\t%s\t%s\t%s\tnan=%d\tinf=%d\tabsmax=%.6g' % (tag, d, k, a.shape, a.dtype, nan, inf, amax))
    names = sorted(devs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = names[i], names[j]
            for k in sorted(set(data[A]) & set(data[B])):
                x = data[A][k].astype(np.float64)
                y = data[B][k].astype(np.float64)
                if x.shape != y.shape:
                    emit('PAIR\t%s\t%s-%s\t%s\tshape mismatch %s %s' % (tag, A, B, k, x.shape, y.shape))
                    continue
                d = np.abs(x - y)
                bad = ~np.isfinite(d)
                n_bad = int(bad.sum())
                d = np.where(bad, np.inf, d)
                idx = np.unravel_index(int(np.argmax(d)), d.shape)
                fin = x[np.isfinite(x)]
                scale = float(np.abs(fin).max()) if fin.size else float('nan')
                thr = 0.01 * scale if scale == scale else float('nan')
                n_gt = int((d > thr).sum()) if thr == thr else -1
                mean_abs = float(np.mean(np.where(bad, 0.0, d)))
                emit('PAIR\t%s\t%s-%s\t%s\tn=%d\tmax_abs=%.6g\tmean_abs=%.6g\tn_nonfinite_diff=%d\tn_gt_1pct=%d\tref_absmax=%.6g\targmax=%s\tref=%.6g\tother=%.6g' % (
                    tag, A, B, k, x.size, float(np.max(d)), mean_abs, n_bad, n_gt, scale,
                    list(int(v) for v in idx), float(x[idx]), float(y[idx])))
open(os.path.join(E, 'parity_detail.tsv'), 'w').write('\n'.join(lines) + '\n')
