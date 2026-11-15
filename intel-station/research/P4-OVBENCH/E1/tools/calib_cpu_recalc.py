# S9 item 5: threshold provenance for pre_close and a CPU recalculation on the same 19-image calibration set.
# Reads R/app (imports only; -B writes no bytecode). Writes only under the E1 folder.
import csv
import importlib
import json
import os
import sys

import numpy as np

R = "/home/intel/intel-junction-2026"
E = R + "/work/P4-OVBENCH/E1"
APP = R + "/app"
ANOM = APP + "/anomaly/anom3"
VIEW = ANOM + "/models/v000/pre_close"
THR_STORED = 534.8227233886719


def pick_threshold(s_good, s_def):
    # Copy of anomaly/anom3/common.py:128-140 (same candidates, same tie rule).
    g = np.asarray(s_good, dtype=float)
    d = np.asarray(s_def, dtype=float)
    vals = np.unique(np.concatenate([g, d]))
    mids = (vals[:-1] + vals[1:]) / 2.0
    cands = np.concatenate([[vals[0] - 1e-9], mids, [vals[-1] + 1e-9]])
    best_t, best_ba = None, -1.0
    for t in cands:
        ba = 0.5 * (float(np.mean(d > t)) + float(np.mean(g <= t)))
        if ba > best_ba + 1e-12 or (abs(ba - best_ba) <= 1e-12 and t > best_t):
            best_t, best_ba = float(t), float(ba)
    return best_t, best_ba


def rates(s_good, s_def, t):
    g = np.asarray(s_good, dtype=float)
    d = np.asarray(s_def, dtype=float)
    tnr = float(np.mean(g <= t))
    tpr = float(np.mean(d > t))
    return tnr, tpr, 0.5 * (tnr + tpr)


def call(score, thr):
    return "DEFECT" if score > thr else "GOOD"


def main():
    with open(VIEW + "/eval_list.tsv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    paths = [r["path"] for r in rows]
    labels = [r["label"] for r in rows]
    preds = [r["pred"] for r in rows]
    gpu = np.array([float(r["score"]) for r in rows])
    is_good = np.array([lab == "GOOD" for lab in labels])
    is_def = ~is_good
    on_disk = [os.path.exists(p) for p in paths]
    print("eval rows %d (GOOD %d, DEFECT %d)" % (len(rows), int(is_good.sum()), int(is_def.sum())))
    print("stored scores (eval_list.tsv, device GPU FP16 per threshold.json) finite: %d of %d" % (int(np.isfinite(gpu).sum()), len(gpu)))
    print("calibration images on disk: %d of %d" % (sum(on_disk), len(paths)))
    thr_g, ba_g = pick_threshold(gpu[is_good], gpu[is_def])
    print("recompute from stored scores: threshold %.10g, stored %.10g, diff %.2e, balanced acc %.6f" % (thr_g, THR_STORED, thr_g - THR_STORED, ba_g))
    tnr, tpr, ba = rates(gpu[is_good], gpu[is_def], THR_STORED)
    print("stored scores at stored threshold: tnr %.4f tpr %.4f balanced acc %.6f" % (tnr, tpr, ba))
    mism = sum(int((s > THR_STORED) != (p == "DEFECT")) for s, p in zip(gpu, preds))
    print("stored pred column vs (score > stored threshold): mismatches %d" % mism)
    cpu, err = None, None
    if all(on_disk):
        try:
            sys.path[:0] = [APP, ANOM]
            try:
                inf = importlib.import_module("anomaly.anom3.infer")
            except Exception as ex1:
                print("import anomaly.anom3.infer failed (%r); trying infer" % (ex1,))
                inf = importlib.import_module("infer")
            scores, dev_c = inf.score_files(VIEW, paths, devices=("CPU",))
            print("score_files device returned: %s" % (dev_c,))
            cpu = np.asarray(scores, dtype=float).ravel()
            if cpu.size != len(paths):
                err = "score_files returned %d results for %d images" % (cpu.size, len(paths))
                cpu = None
        except Exception as ex:
            err = "CPU scoring failed: %r" % (ex,)
    else:
        err = "calibration images missing on disk"
    print("CPU scoring with production infer.score_files (devices=CPU): %s" % ("ok" if cpu is not None else err))
    summary = {
        "stored_threshold_gpu_fp16": THR_STORED,
        "stored_balanced_accuracy": ba_g,
        "n_good": int(is_good.sum()),
        "n_defect": int(is_def.sum()),
        "in_sample": True,
        "rule": "pick_threshold (anomaly/anom3/common.py:128-140): max balanced accuracy, ties to higher threshold",
        "cpu_scorer": "anomaly/anom3/infer.py score_files devices=('CPU',)",
        "error": err,
    }
    if cpu is not None and np.isfinite(cpu).all():
        thr_c, ba_c = pick_threshold(cpu[is_good], cpu[is_def])
        tnr_c, tpr_c, _ = rates(cpu[is_good], cpu[is_def], thr_c)
        tnr_s, tpr_s, ba_s = rates(cpu[is_good], cpu[is_def], THR_STORED)
        diff = np.abs(cpu - gpu)
        print("GPU FP16 vs CPU per-image score: max abs diff %.6g, max rel diff %.4g" % (float(diff.max()), float((diff / np.abs(cpu)).max())))
        print("CPU recalculated threshold %.10g: balanced acc %.6f, tnr %.4f, tpr %.4f" % (thr_c, ba_c, tnr_c, tpr_c))
        print("CPU scores at stored threshold: tnr %.4f tpr %.4f balanced acc %.6f" % (tnr_s, tpr_s, ba_s))
        print("decision flips at stored threshold, GPU FP16 vs CPU: %d" % int(np.sum((gpu > THR_STORED) != (cpu > THR_STORED))))
        print("decision flips on CPU, stored vs recalculated threshold: %d" % int(np.sum((cpu > THR_STORED) != (cpu > thr_c))))
        with open(E + "/calibration_gpu_vs_cpu.tsv", "w", encoding="utf-8") as fh:
            fh.write("image\tlabel\tgpu_fp16_score\tcpu_score\tabs_diff\tpred_gpu_stored\tpred_cpu_stored\tpred_cpu_recalc\n")
            for i, p in enumerate(paths):
                line = "%s\t%s\t%.6f\t%.6f\t%.3g\t%s\t%s\t%s" % (os.path.basename(p), labels[i], gpu[i], cpu[i], diff[i], call(gpu[i], THR_STORED), call(cpu[i], THR_STORED), call(cpu[i], thr_c))
                fh.write(line + "\n")
                print(line)
        summary.update({
            "threshold_cpu_recalc": thr_c,
            "balanced_accuracy_cpu_recalc": ba_c,
            "tnr_cpu_recalc": tnr_c,
            "tpr_cpu_recalc": tpr_c,
            "max_abs_gpu_fp16_vs_cpu": float(diff.max()),
        })
    with open(E + "/threshold_cpu_recalc.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote calibration_gpu_vs_cpu.tsv and threshold_cpu_recalc.json")


if __name__ == "__main__":
    main()
