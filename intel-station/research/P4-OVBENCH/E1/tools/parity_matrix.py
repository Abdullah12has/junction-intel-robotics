# Read-only parity matrix for the anomaly models over the raw bench outputs. Reference = CPU of the same model.
# Decision agreement uses the per-sample score (value for a scalar output, max over the map otherwise) against a threshold.
import json
import os
import re

import numpy as np

E = '/home/intel/intel-junction-2026/work/P4-OVBENCH/E1'
RAW = os.path.join(E, 'raw')
THR = {  # pre_close: production threshold (threshold.json). PatchCore and PaDiM: P4-FUSION device_check.json threshold_all_good (not calibrated on these IRs).
    'anom_pre_close': 534.8227233886719,
    'anom_patchcore': 583.21728515625,
    'anom_padim': 468.472900390625,
    'anom_padim_b1': 468.472900390625,
}
PAIRS = [
    ('anom_pre_close', 'CPU', 'anom_pre_close', 'GPU'),
    ('anom_pre_close', 'CPU', 'anom_pre_close_f32hint', 'GPU'),
    ('anom_pre_close', 'CPU', 'anom_pre_close', 'NPU'),
    ('anom_patchcore', 'CPU', 'anom_patchcore', 'GPU'),
    ('anom_patchcore', 'CPU', 'anom_patchcore', 'NPU'),
    ('anom_padim', 'CPU', 'anom_padim', 'GPU'),
    ('anom_padim', 'CPU', 'anom_padim', 'NPU'),
    ('anom_padim_b1', 'CPU', 'anom_padim_b1', 'GPU'),
    ('anom_padim_b1', 'CPU', 'anom_padim_b1', 'NPU'),
]
KEY = re.compile(r's(\d+)_o(\d+)$')


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
    names = json.load(open(js)).get('output_names')
    return out, names


for tag in ['anom_pre_close', 'anom_patchcore', 'anom_padim', 'anom_padim_b1']:
    C, names = load(tag, 'CPU')
    if C is None:
        print('%s: no CPU output' % tag)
        continue
    js_idx = sorted({j for (_, j) in C})
    print('== %s CPU outputs: %s' % (tag, ', '.join('%d:%s%s' % (j, (names or ['?'] * 9)[j] if j < len(names or []) else '?',
                                                                 list(C[min(C)].shape) if j == js_idx[0] else '') for j in js_idx)))
    for j in js_idx:
        a0 = C[sorted(k for k in C if k[1] == j)[0]]
        print('   out %d name=%s shape=%s' % (j, (names or ['?'])[j] if names and j < len(names) else '?', list(a0.shape)))
print('ref_tag\tref_dev\tcand_tag\tcand_dev\tout\tname\tshape\tref_absmax\tmax_abs\tmean_abs\tn_gt_1pct\tnonfinite\tscore_scale\tthr\tcpu_flagged\tdev_flagged\tdecision_agree')
for rt, rd, ct, cd in PAIRS:
    C, names = load(rt, rd)
    D, _ = load(ct, cd)
    if C is None or D is None:
        print('%s-%s vs %s-%s\tmissing' % (rt, rd, ct, cd))
        continue
    thr = THR[rt]
    for j in sorted({jj for (_, jj) in C}):
        keys = sorted(i for (i, jj) in C if jj == j and (i, j) in D)
        if not keys:
            continue
        a = [C[(i, j)] for i in keys]
        b = [D[(i, j)] for i in keys]
        ref = max(float(np.abs(x[np.isfinite(x)]).max()) for x in a)
        mx = 0.0
        sm = 0.0
        n = 0
        gt = 0
        nonfin = 0
        for x, y in zip(a, b):
            d = np.abs(x - y)
            nonfin += int((~np.isfinite(y)).sum())
            d = np.where(np.isfinite(d), d, np.inf)
            mx = max(mx, float(d.max()))
            sm += float(np.where(np.isfinite(d), d, 0.0).sum())
            n += d.size
            gt += int((d > 0.01 * ref).sum())
        cs = [float(np.max(x)) for x in a]
        ds = [float(np.max(y)) if np.isfinite(y).any() else float('nan') for y in b]
        scale = max(cs) > 10
        if scale:
            agree = sum(1 for p, q in zip(cs, ds) if np.isfinite(q) and ((p > thr) == (q > thr)))
            cf = sum(1 for p in cs if p > thr)
            df = sum(1 for q in ds if np.isfinite(q) and q > thr)
            dec = '%d/%d' % (agree, len(keys))
        else:
            cf = df = 0
            dec = 'n/a (no score-scale values)'
        nm = names[j] if names and j < len(names) else str(j)
        print('%s-%s\t%s-%s\t%d\t%s\t%s\t%.6g\t%.6g\t%.6g\t%d\t%d\t%s\t%.4f\t%d\t%d\t%s' % (
            rt, rd, ct, cd, j, nm, list(a[0].shape), ref, mx, sm / max(n, 1), gt, nonfin, scale, thr, cf, df, dec))
