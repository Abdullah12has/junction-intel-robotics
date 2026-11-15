# Read-only parity of chosen raw npz output pairs (CPU is the reference). No device is touched.
import glob
import json
import os

import numpy as np

E = '/home/intel/intel-junction-2026/work/P4-OVBENCH/E1'
THR = 534.8227233886719  # pre_close DEFECT threshold (threshold.json): DEFECT when pred_score > threshold
PAIRS = [
    ('act_fp16', 'CPU', 'act_fp32', 'CPU'),
    ('act_fp32', 'CPU', 'act_fp32', 'GPU'),
    ('act_fp32', 'CPU', 'act_fp16', 'GPU'),
    ('act_fp32', 'CPU', 'act_fp16', 'NPU'),
    ('anom_pre_close', 'CPU', 'anom_pre_close_f32hint', 'GPU'),
    ('anom_pre_close', 'CPU', 'anom_pre_close', 'GPU'),
    ('anom_pre_close', 'CPU', 'anom_pre_close', 'NPU'),
    ('anom_patchcore', 'CPU', 'anom_patchcore', 'GPU'),
    ('anom_patchcore', 'CPU', 'anom_patchcore', 'NPU'),
    ('anom_padim', 'CPU', 'anom_padim', 'GPU'),
    ('anom_padim', 'CPU', 'anom_padim', 'NPU'),
]
cache = {}
lines = []


def emit(s):
    print(s)
    lines.append(s)


def get(tag, dev):
    key = (tag, dev)
    if key not in cache:
        p = os.path.join(E, 'raw', '%s__%s__out.npz' % (tag, dev))
        if os.path.exists(p):
            z = np.load(p)
            cache[key] = {k: z[k].astype(np.float64) for k in z.files}
        else:
            cache[key] = None
    return cache[key]


emit('pair\tkeys\telements\tmax_abs\tworst_key\tmean_abs_per_elem\tnonfinite_diff\tn_gt_1pct_of_ref_absmax\tscores\tscore_flips_at_thr\tscore_nonfinite\tscore_max_abs')
for ta, da, tb, db in PAIRS:
    A = get(ta, da)
    B = get(tb, db)
    label = '%s-%s vs %s-%s' % (ta, da, tb, db)
    if A is None or B is None:
        emit('%s\tmissing npz' % label)
        continue
    keys = sorted(set(A) & set(B))
    refmax = 0.0
    for k in keys:
        x = A[k]
        if np.isfinite(x).any():
            refmax = max(refmax, float(np.abs(x[np.isfinite(x)]).max()))
    thr_gt = 0.01 * refmax
    n_tot = 0
    sum_abs = 0.0
    max_abs = -1.0
    worst = None
    nonfin = 0
    gt = 0
    n_scores = 0
    flips = 0
    score_nonfin = 0
    score_max = 0.0
    for k in keys:
        x = A[k]
        y = B[k]
        if x.shape != y.shape:
            emit('%s\tshape mismatch at %s %s %s' % (label, k, x.shape, y.shape))
            continue
        d = np.abs(x - y)
        bad = ~np.isfinite(d)
        nonfin += int(bad.sum())
        d2 = np.where(bad, np.inf, d)
        n_tot += x.size
        sum_abs += float(np.where(bad, 0.0, d).sum())
        m = float(d2.max()) if d2.size else 0.0
        if m > max_abs:
            max_abs = m
            worst = k
        gt += int((d2 > thr_gt).sum())
        if x.ndim == 1 and x.size <= 4 and np.isfinite(x).any() and float(np.abs(x[np.isfinite(x)]).max()) > 10:
            n_scores += x.size
            ny = ~np.isfinite(y)
            score_nonfin += int(ny.sum())
            dx = x > THR
            dy = np.where(ny, ~dx, y > THR)
            flips += int((dx != dy).sum())
            score_max = max(score_max, float(d2.max()))
    emit('%s\t%d\t%d\t%.6g\t%s\t%.6g\t%d\t%d\t%d\t%d\t%d\t%.6g' % (
        label, len(keys), n_tot, max_abs, worst, sum_abs / max(n_tot, 1), nonfin, gt,
        n_scores, flips, score_nonfin, score_max))
for p in sorted(glob.glob(os.path.join(E, 'raw', 'act_fp32*.json'))):
    o = json.load(open(p))
    keep = {k: o[k] for k in o if k in ('device', 'gpu_hint', 'compile_cold_s', 'first_infer_s', 'latency_ms', 'async', 'busy', 'device_name')}
    emit('JSON %s %s' % (os.path.basename(p), json.dumps(keep)[:900]))
open(os.path.join(E, 'parity_focus.tsv'), 'w').write('\n'.join(lines) + '\n')
