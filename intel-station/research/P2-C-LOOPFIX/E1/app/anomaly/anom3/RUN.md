# anom3 run book (P2-B-ANOM3, builder E1)

PatchCore per view (pre_close, wrist_hover, held_close, station_close), trained on GOOD crops only (anomalib 2.7.0).
Export: OpenVINO IR, FP16. Training and scoring are CPU only (coordinator update 18:32Z: no GPU jobs on this laptop;
infer.DEVICES is ("CPU",) and ViewModel refuses any other device). Every command runs under nice -n 19 with
OMP_NUM_THREADS=4. Fit budget: 180 s per view (fit_within_budget in metrics.json).

Setup, once per shell:

    PROJ=/home/intel/intel-junction-2026
    . $PROJ/envs/env.sh
    APP=$PROJ/app/anomaly/anom3

1. Data check, trains nothing. Prints per view the train GOOD, held-out GOOD and held-out DEFECT counts, the episode
   groups (RUN/ep_NNNN), the number of crops with success "missing", and the excluded labels and skip reasons:

    nice -n 19 env OMP_NUM_THREADS=4 $PY -I -B $APP/train.py --dry-run

2. Train a new version into models/VNNN/<view>/ (leave --version out to take the next free number):

    nice -n 19 env OMP_NUM_THREADS=4 $PY -I -B $APP/train.py --version VNNN > $PROJ/work/P2-B-ANOM3/E1/train_VNNN.log 2>&1

   Options: --views pre_close (one view), --label-sources opus (Opus labels only), --source folder --folder-root DIR
   (legacy layout, smoke only). A view that is skipped says why in models/VNNN/summary.json.

3. Compare the new version with a baseline on the same held-out crops (writes models/VNNN/eval_vs_BASELINE.json, or
   the file given with --out):

    nice -n 19 env OMP_NUM_THREADS=4 $PY -I -B $APP/eval.py --candidate VNNN --baseline vNNN

   Prints per view: n GOOD, n DEFECT, AUROC and balanced accuracy for both, and PROMOTE or KEEP with the reason.
   Without a baseline use --baseline none (PROMOTE needs AUROC >= 0.70 and at least 5 held-out of each class).

4. Promote, Opus only, after the verdict (eval.py never moves the symlink):

    ln -sfn VNNN $APP/models/current

5. Score BGR crops (any size) with a version (default: current):

    nice -n 19 env OMP_NUM_THREADS=4 $PY -I -B $APP/infer.py --view pre_close --image CROP.png --image CROP2.png

   From Python: import infer; infer.score(view, bgr) and infer.score_batch(view, [bgr, ...]) return
   {"view", "score", "threshold", "label", "latency_ms", "device", "version"}; label is DEFECT when score > threshold.

Retrain cadence (Opus): every 15 to 20 new cycles. Count new episodes with
   ls $PROJ/logs/sort/*/ep_*/episode.json | wc -l
then run steps 1, 2 and 3, apply step 4 only on a PROMOTE, and log the verdict in the Opus STATUS.

Data rules (common.py):
- Labels come only from episode.json (label GOOD or DEFECT; label_source det2 or opus_visual). Nothing is relabelled.
- Skipped and counted: no label; label not GOOD or DEFECT; success false; plate not matching the label
  (GOOD must be blue, DEFECT green). An episode with no success key is kept and its crops are counted as success
  "missing": these are carry-over episodes that were interrupted and then recovered (sort_any.py sets success only
  after a clean finish).
- Views: pre_close = crops/pre_close.png; wrist_hover = crops/wrist_hover.png (a full frame, not a brick crop);
  held_close = crops/held_close_r0.png (roll 0 only, because the sorter marks roll 180 NOT_USED); station_close =
  crops/station_close.png.
- Held-out split, fixed for every version: DEFECT is always held out (never trained on); GOOD is held out when
  md5(RUN/ep_NNNN) % 4 == 0, otherwise it trains. MERGED and other labels are excluded and counted.
- A view with fewer than 2 training GOOD crops is skipped. A view with fewer than 4 is written with low_n true (not a
  model to promote). A view with no held-out GOOD crop is skipped, because the threshold cannot be calibrated.

Files per version (models/VNNN/<view>/): model.xml and model.bin (FP16 IR), metadata.json, threshold.json (threshold,
rule, score meaning), metrics.json (AUROC, balanced accuracy, accuracy, counts, low_n, train and held-out groups,
fit and export seconds, device), train_list.txt (every training crop), eval_list.tsv (every held-out crop with score
and prediction), disagreements.tsv (prediction differs from the label; Opus checks these by eye).
models/VNNN/summary.json lists the views, the skip reasons and the excluded counts.

Versions so far:
- v000 = bootstrap from the legacy E2 close set (--source folder, pre_close only, 6 train GOOD, 12 held-out GOOD,
  7 DEFECT). Not a production model; do not promote it on its own. On the loop's held-out pre_close crops (2 GOOD,
  2 DEFECT) it has AUROC 0.500 (models/v000/eval_loop_vs_none.json).
- v001 = attempted on the loop data (any_1731, any_1740, 7 usable episodes). No view trained: under the fixed split
  each view has 0 or 1 training GOOD crops (models/v001/summary.json). Nothing to promote.

Caveats:
- The threshold is fitted on the held-out crops that the metrics use, so the metrics are optimistic.
- Held-out sets stay small until the loop has more episodes; eval.py says KEEP below 5 crops per class.
- anomalib exports print a non-fatal onnx version-converter assertion; the IR checks out (FP32 and FP16 agree on CPU).
- Before any retrain, read STATUS.md on the label conflicts in held_close and the wrist_hover full-frame crops.
