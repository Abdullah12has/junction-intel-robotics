"""Shared data and metric helpers for the anom3 PatchCore models (R/app/anomaly/anom3).

Labels come only from episode.json (or the legacy folder layout). Nothing is relabelled here.

Sources
- episodes: R/logs/sort/RUN/ep_NNNN/episode.json plus its crops/ folder.
  Usable: label GOOD or DEFECT, success not false, plate blue for GOOD and green for DEFECT.
  Carry-over episodes that were interrupted and then recovered never get a success key; they are kept and
  counted with success "missing" (see the dry run).
  Views: pre_close -> crops/pre_close.png, wrist_hover -> crops/wrist_hover.png,
         held_close -> crops/held_close_r0.png (roll 0 only; the sorter marks roll 180 NOT_USED),
         station_close -> crops/station_close.png.
- folder: anomalib layout train/good, test/good, test/defect (legacy E2 bricks_close, smoke tests only).

Split rule (fixed, so every version and the baseline see the same held-out set)
- DEFECT crops are never trained on, so they are always held out.
- GOOD crops are held out when md5(group) % 4 == 0, otherwise they train. group = RUN/ep_NNNN, so both views
  of one episode fall on the same side.
- folder source: train/good trains, test/* is held out.
- Other labels (MERGED, empty, unknown) and the other skip reasons in load_episodes are excluded and counted.
"""
import glob
import hashlib
import json
import os

import numpy as np

R = "/home/intel/intel-junction-2026"
LOGS_SORT = os.path.join(R, "logs", "sort")
LEGACY_E2 = os.path.join(R, "work", "P2-B-DET2", "E2", "anomalib", "bricks_close")
VIEWS = ("pre_close", "wrist_hover", "held_close", "station_close")
HELDOUT_MOD = 4
PLATE_FOR_LABEL = {"GOOD": "blue", "DEFECT": "green"}


def _bucket(group):
    return int(hashlib.md5(group.encode("utf-8")).hexdigest(), 16) % HELDOUT_MOD


def _view_files(crops_dir, view):
    if view == "held_close":
        # roll 0 only: the sorter marks the roll-180 view NOT_USED (cable in view), so it never decides a label
        path = os.path.join(crops_dir, "held_close_r0.png")
    else:
        path = os.path.join(crops_dir, view + ".png")  # pre_close, wrist_hover, station_close
    return [path] if os.path.isfile(path) else []


def load_episodes(logs_root=LOGS_SORT, label_sources=None):
    items, excluded = [], {}

    def skip(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    for ep_json in sorted(glob.glob(os.path.join(logs_root, "*", "ep_*", "episode.json"))):
        ep_dir = os.path.dirname(ep_json)
        run = os.path.basename(os.path.dirname(ep_dir))
        group = run + "/" + os.path.basename(ep_dir)
        try:
            with open(ep_json, encoding="utf-8") as fh:
                ep = json.load(fh)
            label = ep.get("label")
            src = str(ep.get("label_source", "unknown"))
            plate = ep.get("plate")
            success = ep.get("success")
        except (OSError, ValueError, AttributeError):
            skip("unreadable")
            continue
        if label is None:
            skip("no label")
            continue
        if label not in PLATE_FOR_LABEL or (label_sources and src not in label_sources):
            skip("%s|%s" % (label, src))
            continue
        if success is False:
            skip("success false")
            continue
        if plate is not None and plate != PLATE_FOR_LABEL[label]:
            skip("plate %s on %s" % (plate, label))
            continue
        state = "true" if success is True else "missing"
        for view in VIEWS:
            for path in _view_files(os.path.join(ep_dir, "crops"), view):
                items.append({"path": path, "view": view, "label": label, "source": "episodes:" + src,
                              "group": group, "run": run, "episode": ep.get("episode"),
                              "plate": plate, "success": state})
    return items, excluded


def load_folder(root=LEGACY_E2):
    layout = (("train/good", "GOOD", "train"), ("test/good", "GOOD", "heldout"),
              ("test/defect", "DEFECT", "heldout"))
    items = []
    for sub, label, split in layout:
        for path in sorted(glob.glob(os.path.join(root, sub, "*.png"))):
            items.append({"path": path, "view": "pre_close", "label": label,
                          "source": "folder:" + os.path.basename(root),
                          "group": "folder/" + os.path.basename(path), "split": split})
    return items, {}


def assign_split(item):
    if "split" in item:
        return item["split"]
    if item["label"] == "DEFECT":
        return "heldout"
    return "heldout" if _bucket(item["group"]) == 0 else "train"


def load(source, logs_root=LOGS_SORT, folder_root=LEGACY_E2, label_sources=None):
    if source == "episodes":
        items, excluded = load_episodes(logs_root, label_sources)
    elif source == "folder":
        items, excluded = load_folder(folder_root)
    else:
        raise ValueError("source must be episodes or folder")
    for it in items:
        it["split"] = assign_split(it)
    return items, excluded


def select(items, view, split, label=None):
    return [it for it in items
            if it["view"] == view and it["split"] == split and (label is None or it["label"] == label)]


def pick_threshold(s_good, s_def):
    """Threshold t (DEFECT when score > t) with the highest balanced accuracy; ties go to the higher t."""
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


def binary_metrics(s_good, s_def, thr):
    from sklearn.metrics import roc_auc_score
    g = np.asarray(s_good, dtype=float)
    d = np.asarray(s_def, dtype=float)
    out = {"n_good": int(g.size), "n_defect": int(d.size), "threshold": thr}
    out["auroc"] = None
    if g.size and d.size:
        out["auroc"] = float(roc_auc_score(np.r_[np.zeros(g.size), np.ones(d.size)], np.r_[g, d]))
    out["tnr"] = float(np.mean(g <= thr)) if (g.size and thr is not None) else None
    out["tpr"] = float(np.mean(d > thr)) if (d.size and thr is not None) else None
    out["balanced_accuracy"] = None
    out["accuracy"] = None
    if out["tnr"] is not None and out["tpr"] is not None:
        out["balanced_accuracy"] = 0.5 * (out["tnr"] + out["tpr"])
        out["accuracy"] = (out["tnr"] * g.size + out["tpr"] * d.size) / float(g.size + d.size)
    return out
