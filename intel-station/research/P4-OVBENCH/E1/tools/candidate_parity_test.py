# S9 item 11 test: GPU + f32-hint candidate of the pre_close scorer against CPU on the recorded snapshots.
# It replays the candidate config's outputs from the existing bench run (same IR, GPU with INFERENCE_PRECISION_HINT=f32):
# raw/anom_pre_close_f32hint__GPU__out.npz against raw/anom_pre_close__CPU__out.npz. It does not import infer.py.
# Output rule (as parity_matrix2.py): every value finite and max abs diff <= 1% of the CPU absmax, for each output.
# Decision rule: pred_score > threshold, at the production threshold and at the CPU recalculated threshold (if present).
# Also checks that the patched source carries the candidate config lines.
# Exit: 0 pass; 1 parity, decision or static check failed; 2 input error; 3 self-test did not detect the injected flip.
import argparse
import json
import os
import re

import numpy as np

E = "/home/intel/intel-junction-2026/work/P4-OVBENCH/E1"
RAW = os.path.join(E, "raw")
PATCHED = os.path.join(E, "app_copy", "anomaly", "anom3", "infer.py")
THR_PROD = 534.8227233886719
KEY = re.compile(r"s(\d+)_o(\d+)$")


def load(tag, dev):
    z = np.load(os.path.join(RAW, "%s__%s__out.npz" % (tag, dev)))
    out = {}
    for k in z.files:
        m = KEY.match(k)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = z[k].astype(np.float64)
    with open(os.path.join(RAW, "%s__%s.json" % (tag, dev)), encoding="utf-8") as fh:
        names = json.load(fh).get("output_names")
    return out, names


def flips(cpu, gpu, thr):
    # Decision flips between CPU and GPU at thr; a non-finite GPU value counts as a failure.
    return int(np.sum((cpu > thr) != (gpu > thr))) + int(np.sum(~np.isfinite(gpu)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-flip", action="store_true")
    ap.add_argument("--thr-cpu-json", default=os.path.join(E, "threshold_cpu_recalc.json"))
    args = ap.parse_args()
    C, names = load("anom_pre_close", "CPU")
    G, names_g = load("anom_pre_close_f32hint", "GPU")
    if not C or not G or not names or names != names_g or "pred_score" not in names:
        print("INPUT ERROR: npz keys or output names missing")
        return 2
    j_score = names.index("pred_score")
    keys = sorted(i for (i, j) in C if j == j_score and (i, j_score) in G)
    a_s = np.concatenate([C[(i, j_score)].ravel() for i in keys])
    b_s = np.concatenate([G[(i, j_score)].ravel() for i in keys])
    print("recorded snapshot calls: %d; images (pred_score values): %d" % (len(keys), a_s.size))
    if args.selftest_flip:
        idx = int(np.argmin(np.abs(a_s - THR_PROD)))
        b2 = b_s.copy()
        b2[idx] = THR_PROD + 1.0 if a_s[idx] <= THR_PROD else THR_PROD - 1.0
        det = flips(a_s, b2, THR_PROD)
        if det > 0:
            print("SELFTEST OK: injected decision flip (image %d) detected, %d flip(s)" % (idx, det))
            return 0
        print("SELFTEST FAILED: injected flip not detected")
        return 3
    parity_ok = True
    print("output names: %s" % (names,))
    for j, name in enumerate(names):
        a = np.concatenate([C[(i, j)].ravel() for i in keys])
        b = np.concatenate([G[(i, j)].ravel() for i in keys])
        fa = np.isfinite(a)
        ref = float(np.abs(a[fa]).max())
        tol = 0.01 * ref
        nonfin = int((~np.isfinite(b)).sum()) + int((~fa).sum())
        d = np.abs(a - b)
        fin = np.isfinite(d)
        mx = float(d[fin].max()) if fin.any() else float("inf")
        if nonfin:
            mx = float("inf")
        passed = nonfin == 0 and mx <= tol
        parity_ok = parity_ok and passed
        print("parity %d %-11s n=%d ref_absmax=%.6g tol=%.3g max_abs=%.3g nonfinite=%d %s" % (j, name, a.size, ref, tol, mx, nonfin, "PASS" if passed else "FAIL"))
    thrs = [("production threshold.json", THR_PROD)]
    try:
        with open(args.thr_cpu_json, encoding="utf-8") as fh:
            tc = json.load(fh).get("threshold_cpu_recalc")
        if tc is not None:
            thrs.append(("CPU recalculated threshold", float(tc)))
    except (OSError, ValueError):
        print("no CPU recalculated threshold file")
    dec_ok = True
    for label, thr in thrs:
        nf = flips(a_s, b_s, thr)
        dec_ok = dec_ok and nf == 0
        print("decision at %s (%.6f): CPU DEFECT calls %d of %d; CPU vs GPU f32 flips %d" % (label, thr, int(np.sum(a_s > thr)), a_s.size, nf))
    k = int(np.argmin(np.abs(a_s - THR_PROD)))
    print("closest CPU score to production threshold: %.4f, margin %.4f" % (a_s[k], abs(a_s[k] - THR_PROD)))
    print("CPU pred_score: %s" % np.round(a_s, 3).tolist())
    print("GPU f32 pred_score: %s" % np.round(b_s, 3).tolist())
    print("pred_score max abs CPU vs GPU f32: %.4g" % float(np.max(np.abs(a_s - b_s))))
    with open(PATCHED, encoding="utf-8") as fh:
        src = fh.read()
    static = [
        ('DEVICES = ("GPU", "CPU")', 'DEVICES = ("GPU", "CPU")' in src),
        ('guard allows GPU: d not in ("CPU", "GPU")', 'd not in ("CPU", "GPU")' in src),
        ('f32 hint on GPU compile', '{"INFERENCE_PRECISION_HINT": "f32"} if dev == "GPU"' in src),
    ]
    static_ok = all(v for _, v in static)
    for label, v in static:
        print("static check (patched app_copy infer.py) %s: %s" % (label, "present" if v else "MISSING"))
    print("RESULT parity %s, decisions %s, static config %s" % ("PASS" if parity_ok else "FAIL", "PASS" if dec_ok else "FAIL", "PASS" if static_ok else "FAIL"))
    passed = parity_ok and dec_ok and static_ok
    print("TEST PASS" if passed else "TEST FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
