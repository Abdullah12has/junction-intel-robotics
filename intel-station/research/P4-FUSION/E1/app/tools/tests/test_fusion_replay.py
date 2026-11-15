"""P4-FUSION replay: the fused close decisions of the LOEO evaluation, replayed through the patched sort_loop.decide().

Fixtures in fixtures/fusion_v1 (written by the P4-FUSION E1 evaluation, stage fusion-v1): replay_fixture.tsv holds the 32
labelled close crops of the sensitivity set with the recorded tape verdict and tape score, each model's fold score and
threshold, and the binary LOEO decision of every fused rule; fusion_folds.json holds the logistic regression fitted
without the held-out episode, per model and held-out group. A replayed decision must equal the recorded one. No model
runs and no device is opened: the anomaly evidence comes from the fixture.
"""
import csv
import json
import os

import conftest as C

sl = C.sl
FUSED = os.path.join(C.FIX, "fusion_v1")
SET = "sensitivity_incl_v1_unverified"
RULES = ("AND", "OR", "LR")
N_CROPS = 32


def _rows():
    path = os.path.join(FUSED, "replay_fixture.tsv")
    if not os.path.exists(path):
        raise AssertionError("fixtures/fusion_v1/replay_fixture.tsv is missing")
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != N_CROPS:
        raise AssertionError("replay fixture has %d crops, expected %d" % (len(rows), N_CROPS))
    return rows


def _folds():
    path = os.path.join(FUSED, "fusion_folds.json")
    if not os.path.exists(path):
        raise AssertionError("fixtures/fusion_v1/fusion_folds.json is missing")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["sets"][SET]


def _replay(model, rule):
    col = "%s_%s" % (model, rule)
    folds = _folds()[model]
    checked = defects = 0
    for row in _rows():
        if row[col] == "":
            continue
        lr = None
        if rule == "LR":
            lr = folds.get(row["group"])
            if lr is None:
                raise AssertionError("no LR fold for %s" % row["group"])
        anomaly = {"model": model, "device": "replay", "score": float(row[model + "_score"]),
                   "threshold": float(row[model + "_thr"]), "rule": rule, "lr": lr}
        pre = {"verdict": row["tri_verdict"], "tape_score": float(row["tape_score"]), "reasons": ["replay"]}
        label = sl.decide(pre, ("2x2", "replay"), [], anomaly=anomaly)[1]
        want = "DEFECT" if row[col] == "1" else "GOOD"
        if label != want:
            raise AssertionError("%s %s %s: decide() gives %s, LOEO gives %s"
                                 % (model, rule, row["rec_id"], label, want))
        checked += 1
        defects += want == "DEFECT"
    if checked == 0:
        raise AssertionError("no decisions to replay for %s %s" % (model, rule))
    return checked, defects


def _replay_all(model):
    parts = []
    for rule in RULES:
        n, d = _replay(model, rule)
        parts.append("%s %d crops, %d DEFECT" % (rule, n, d))
    return "%s replayed: %s" % (model, "; ".join(parts))


def test_fusion_flag_off_and_no_evidence_unchanged():
    import fused
    assert fused.config()["enabled"] is False, "fused_config.json must ship with enabled false"
    assert fused.score(C.load_img("close/ep0006_pre_close_full.jpg")) is None, "score() must return None while off"
    defect = {"verdict": "CONFIDENT_DEFECT", "reasons": ["tape_frac_high"]}
    good = {"verdict": "CONFIDENT_GOOD", "reasons": []}
    assert sl.decide(defect, ("2x2", "replay"), [])[1] == "DEFECT"
    assert sl.decide(good, ("2x2", "replay"), [])[1] == "GOOD"
    assert sl.decide(defect, ("2x2", "replay"), [], anomaly=None)[1] == "DEFECT"
    return "flag off; decide() without evidence unchanged"


def test_fused_evidence_failure_falls_back_to_tape():
    import fused
    saved = dict(fused._cache)
    logs = []
    try:
        fused._cache["cfg"] = dict(fused.config(), enabled=True, ir_dir=os.path.join(C.FIX, "no_such_ir"))
        got = fused.score(C.load_img("close/ep0006_pre_close_full.jpg"), log=logs.append)
    finally:
        fused._cache.clear()
        fused._cache.update(saved)
    assert got is None, "unavailable evidence must give None, so the tape verdict stands"
    assert logs and "tape verdict stands" in logs[0], logs
    return "unavailable IR: score() gives None and logs; the tape verdict stands"


def test_fused_rule_sets_close_verdict():
    def lab(pre, anomaly):
        return sl.decide(pre, ("2x2", "replay"), [], anomaly=anomaly)

    over = {"model": "patchcore", "device": "replay", "score": 600.0, "threshold": 584.0, "rule": "OR", "lr": None}
    under = dict(over, score=400.0, rule="AND")
    tape_unc = {"verdict": "UNCERTAIN", "tape_score": 0.0, "reasons": []}
    tape_def = {"verdict": "CONFIDENT_DEFECT", "tape_score": 0.1, "reasons": ["tape_frac_high"]}
    before = dict(tape_def)
    got = lab(tape_unc, over)
    assert got[1] == "DEFECT" and "fused_OR" in " ".join(got[3]), got
    got = lab(tape_def, under)
    assert got[1] == "GOOD", "AND must clear a tape DEFECT that the anomaly does not confirm: %r" % (got,)
    assert lab(tape_def, None)[1] == "DEFECT"
    assert tape_def == before, "decide() must not change its input"
    return "OR adds an anomaly DEFECT (fused_OR); AND clears an unconfirmed tape DEFECT; input unchanged"


def test_replay_patchcore_fused_rules():
    return _replay_all("patchcore")


def test_replay_padim_fused_rules():
    return _replay_all("padim")
