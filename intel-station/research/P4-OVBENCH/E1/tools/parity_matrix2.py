# Read-only parity matrix v2. Reference = CPU of the same IR on the same bench inputs.
# Per output: max abs diff, elements above 1% of the reference absmax, non-finite count, and the fraction of exact 0 and exact 1 values
# on each side (a saturation check). Decisions: pred_score > 534.8227 for pre_close (production threshold); pred_label > 0.5 for
# PatchCore and PaDiM (the IR's own 0/1 decision; their outputs are normalized to [0, 1], so the P4-FUSION threshold does not apply).
# An output passes if every value is finite and max abs diff <= 1% of the reference absmax. A row passes if every output passes and
# every decision agrees.
import json
import os
import re

import numpy as np

E = '/home/intel/intel-junction-2026/work/P4-OVBENCH/E1'
RAW = os.path.join(E, 'raw')
THR_PRECLOSE = 534.8227233886719
KEY = re.compile(r's(\d+)_o(\d+)$')
PAIRS = [
    ('anom_pre_close', 'CPU', 'anom_pre_close', 'GPU'),
    ('anom_pre_close', 'CPU', 'anom_pre_close_f32hint', 'GPU'),
    ('anom_pre_close', 'CPU', 'anom_pre_close', 'NPU'),
    ('anom_patchcore', 'CPU', 'anom_patchcore', 'GPU'),
    ('anom_patchcore', 'CPU', 'anom_patchcore_f32hint', 'GPU'),
    ('anom_patchcore', 'CPU', 'anom_patchcore', 'NPU'),
    ('anom_padim', 'CPU', 'anom_padim', 'GPU'),
    ('anom_padim', 'CPU', 'anom_padim_f32hint', 'GPU'),
    ('anom_padim', 'CPU', 'anom_padim', 'NPU'),
    ('anom_padim', 'CPU', 'anom_padim_b1', 'CPU'),
    ('anom_padim_b1', 'CPU', 'anom_padim_b1', 'GPU'),
    ('anom_padim_b1', 'CPU', 'anom_padim_b1', 'NPU'),
]


def load(tag, dev):
    npz = os.path.join(RAW, '%s__%s__out.npz' % (tag, dev))
    js = os.path.join(RAW, '%s__%s.json' % (tag, dev))
    if not (os.path.exists(npz) and os.path.exists(js)):
        return None, None
    z = np.load(npz)
    out = {}
    for k in z.files:
        m = KEY.match(k)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = z[k].astype(np.float64)
    return out, json.load(open(js)).get('output_names')


def fmt(x):
    return 'inf' if x == float('inf') else '%.6g' % x


print('ref\tcand\tout\tname\tshape\tref_absmax\tmax_abs\tn_gt_1pct\tnonfinite\tcpu_frac0\tcpu_frac1\tdev_frac0\tdev_frac1\toutput_pass\tdecision_rule\tagree')
for rt, rd, ct, cd in PAIRS:
    C, names = load(rt, rd)
    D, _ = load(ct, cd)
    label_r = '%s-%s' % (rt, rd)
    label_c = '%s-%s' % (ct, cd)
    if C is None or D is None:
        print('%s\t%s\tmissing' % (label_r, label_c))
        continue
    row_pass = True
    for j in sorted({jj for (_, jj) in C}):
        keys = sorted(i for (i, jj) in C if jj == j and (i, j) in D)
        if not keys:
            continue
        a = np.concatenate([C[(i, j)].ravel() for i in keys])
        b = np.concatenate([D[(i, j)].ravel() for i in keys])
        name = names[j] if names and j < len(names) else str(j)
        fa = np.isfinite(a)
        ref = float(np.abs(a[fa]).max())
        tol = 0.01 * ref
        nonfin = int((~np.isfinite(b)).sum()) + int((~fa).sum())
        diff = np.abs(a - b)
        fin = np.isfinite(diff)
        max_abs = float(diff[fin].max()) if fin.any() else float('inf')
        if nonfin:
            max_abs = float('inf')
        n_gt = int((~(diff <= tol)).sum())
        out_pass = (nonfin == 0) and (max_abs <= tol)
        row_pass = row_pass and out_pass
        rule = '-'
        agree = '-'
        if rt.startswith('anom_pre_close') and name == 'pred_score':
            rule = 'pred_score > %.4f' % THR_PRECLOSE
            da, db = a > THR_PRECLOSE, b > THR_PRECLOSE
        elif name == 'pred_label':
            rule = 'pred_label > 0.5'
            da, db = a > 0.5, b > 0.5
        if rule != '-':
            ok = (da == db) & np.isfinite(b)
            agree = '%d/%d' % (int(ok.sum()), int(ok.size))
            row_pass = row_pass and int(ok.sum()) == int(ok.size)
        print('%s\t%s\t%d\t%s\t%s\t%s\t%s\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%s\t%s\t%s' % (
            label_r, label_c, j, name, list(C[(keys[0], j)].shape), '%.6g' % ref, fmt(max_abs), n_gt, nonfin,
            float(np.mean(a == 0)), float(np.mean(a == 1)), float(np.mean(b == 0)), float(np.mean(b == 1)),
            'PASS' if out_pass else 'FAIL', rule, agree))
    print('ROW %s vs %s: %s' % (label_r, label_c, 'PASS' if row_pass else 'FAIL'))
