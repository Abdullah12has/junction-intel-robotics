"""Inspection v2 in the loop.
1. Before the grasp, at the hover above the brick: survey it where it lies (survey.views_here: views from the relative
   library, faces hidden by neighbouring bricks are not used), judge each face (faces.score, fitted face model).
2. After the grasp: a 2x2 brick with faces still not judged (neighbours hid them, or no view) is put down alone at the
   planned inspection spot and surveyed there completely (survey_inspect), then re-picked.
3. decide_faces: blue only for a measured 2x2 brick whose every face is CLEAN; any DEFECT face, a not-2x2 shape or a
   confident pre-grasp defect -> green DEFECT; everything else -> green UNCERTAIN.
Faces are projected with the joints the arm actually reached (encoders), so moves need no extra settling.
Exact anchors; backups run/bak_v2_*."""
import os
import py_compile
import shutil
import sys
import time

T = "/home/intel/intel-junction-2026/app/tools"
B = "/home/intel/intel-junction-2026/run"


def patch(path, edits):
    s = open(path).read()
    for old, new in edits:
        n = s.count(old)
        if n != 1:
            sys.exit(f"ABORT {os.path.basename(path)}: anchor x{n}: {old[:100]!r}")
        s = s.replace(old, new)
    shutil.copy(path, f"{B}/bak_v2_{os.path.basename(path)[:-3]}_{time.strftime('%H%M', time.gmtime())}.py")
    open(path + ".tmp", "w").write(s)
    py_compile.compile(path + ".tmp", doraise=True)
    os.replace(path + ".tmp", path)
    print("patched", path)


DECIDE_FACES = '''

def decide_faces(pre, shape, faces, required):
    """OD-10 with full-surface evidence. DEFECT (green): the shape is not 2x2, a surveyed face shows a defect, or the
    pre-grasp view is a confident defect. GOOD (blue): a measured 2x2 brick whose required faces were all surveyed
    CLEAN. Everything else is UNCERTAIN (green). Returns (plate, label, confidence, reasons)."""
    if shape[0] == "not_2x2":
        return "green", "DEFECT", "confident", ["not_2x2:" + shape[1]]
    bad = [f for f in required if faces.get(f, {}).get("verdict") == "DEFECT"]
    if pre.get("verdict") == "CONFIDENT_DEFECT":
        bad.append("pre_close:" + ",".join(pre.get("reasons", [])))
    if bad:
        return "green", "DEFECT", "confident", ["defect: " + ", ".join(bad)]
    if shape[0] != "2x2":
        return "green", "UNCERTAIN", "shape_unknown", ["shape not measured: UNCERTAIN -> GREEN (OD-10)"]
    open_faces = [f for f in required if faces.get(f, {}).get("verdict") != "CLEAN"]
    if open_faces:
        why = {f: faces.get(f, {}).get("reason", faces.get(f, {}).get("verdict", "not surveyed")) for f in open_faces}
        return "green", "UNCERTAIN", "incomplete", [f"not cleared: {why} -> GREEN (OD-10)"]
    return "blue", "GOOD", "confident", ["2x2, every surveyed face clean"]
'''

s = open(f"{T}/sort_loop.py").read()
if "def decide_faces(" not in s:
    anchor = "\n\ndef place(plate):\n"
    if s.count(anchor) != 1:
        sys.exit("ABORT sort_loop place anchor")
    shutil.copy(f"{T}/sort_loop.py", f"{B}/bak_v2_sort_loop_{time.strftime('%H%M', time.gmtime())}.py")
    s = s.replace(anchor, DECIDE_FACES + anchor)
    open(f"{T}/sort_loop.py.tmp", "w").write(s)
    py_compile.compile(f"{T}/sort_loop.py.tmp", doraise=True)
    os.replace(f"{T}/sort_loop.py.tmp", f"{T}/sort_loop.py")
    print("patched sort_loop.py (decide_faces)")

SURVEY_FN = '''

FACE_MODEL_F = os.path.join(APP, "config", "face_model.json")
ALIGN_SEARCH_PX = int(round(6 * float(srv.wc.M["reproj_median_px"])))  # 6 x the camera model's median reprojection error
SCAN_RESID_MM = None  # median residual (mm) of the current scan map, set by fit_scan_corr: neighbour position uncertainty


def survey_plan():
    try:
        with open(srv.PLAN_F) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def judge_view(img, qa, faces3d, names, model):
    """Align the projected brick to this view and judge the named faces. qa: joints the arm reached."""
    proj = srv.project_faces(qa, faces3d, ga.JAW_MASK)
    al = srv.align_shift(img, proj, ALIGN_SEARCH_PX)
    out = {}
    for f in names:
        uv = proj[f]["uv"] + (np.array(al["shift"], float) if al.get("ok") else 0.0)
        r = fcs.score(img, uv, faces3d[f]["area_mm2"], model) if model else {"verdict": "UNCERTAIN", "reason": "no face model"}
        r.update({"cos": round(proj[f]["cos"], 3), "align": {k: al.get(k) for k in ("ok", "iou", "shift", "sep")},
                  "uv": np.round(uv, 1).tolist(), "q": [round(float(v), 2) for v in qa]})
        out[f] = r
    return out


def run_views(views, faces3d, model, sdir, tag):
    """Visit the views (least wrist-roll travel first), photograph, judge. Returns {face: result}."""
    os.makedirs(sdir, exist_ok=True)
    res = {}
    for i, v in enumerate(sorted(views, key=lambda v: v["q"][4])):
        ga.move(list(v["q"]) + [25.0], speed=CFG["transit"], comp=False)
        img = pa.frame()
        qa = ga.state()["present"][:5]
        if img is None:
            res.update({f: {"verdict": "UNCERTAIN", "reason": "no frame"} for f in v["faces"]})
            continue
        cv2.imwrite(f"{sdir}/{tag}{i}.jpg", img)
        for f, r in judge_view(img, qa, faces3d, v["faces"], model).items():
            r["view"] = f"{tag}{i}"
            res[f] = r
    return res


def survey_here(k, q_hover, bh, b, cand):
    """Survey the brick where it lies, before the grasp: views from the relative library that neighbouring bricks do
    not hide, each face judged; the arm returns to the hover. Faces without a clear view stay open for the spot."""
    sp = srv.spec()
    model = fcs.load_model(FACE_MODEL_F)
    t0 = time.time()
    img0 = pa.frame()
    q0 = ga.state()["present"][:5]
    z_top = kin.MAT_Z + sp["height_mm"]
    xy0 = [float(v) for v in srv.wc.pixel_to_mat(q0, bh["centre"], z_top)]
    yaw0 = kin.psi_of(q0)
    meas = srv.measure_top(img0, q0, xy0, yaw0, sp) if img0 is not None else {"ok": False, "why": "no frame"}
    xy, yaw = (meas["xy"], meas["yaw"]) if meas.get("ok") else (xy0, yaw0)
    fp = sp["footprint_mm"]
    inflate = 2.0 * SCAN_RESID_MM if SCAN_RESID_MM else fp
    others = [{"xy": c["xy"], "yaw": 0.0, "size_mm": (fp * max(1.0, float(c.get("aspect", 1.0))),) * 2}
              for c in cand if c is not b and math.hypot(c["xy"][0] - b["xy"][0], c["xy"][1] - b["xy"][1]) > 1.0]
    lib = srv.library()
    crit = (json.load(open(srv.LIB_F)).get("criteria", {}) if lib else {})
    plan = srv.views_here(list(q_hover), xy, yaw, sp, ga.JAW_MASK, lib, others, inflate,
                          crit.get("min_cos", 0.35), crit.get("max_jaw", 0.03), crit.get("max_occ", 0.02)) if lib else \\
        {"views": [], "missing": list(srv.FACES)}
    faces3d = srv.brick_faces(xy, yaw, (fp, fp), sp["height_mm"], kin.MAT_Z)
    sdir = f"{ga.OUT}/survey"
    os.makedirs(sdir, exist_ok=True)
    res = {}
    here = [v for v in plan["views"] if max(abs(a - c) for a, c in zip(v["q"], q0)) < 0.5]
    if here and img0 is not None:  # the hover itself is one of the views: no move
        cv2.imwrite(f"{sdir}/here_hover.jpg", img0)
        for f, r in judge_view(img0, q0, faces3d, here[0]["faces"], model).items():
            r["view"] = "here_hover"
            res[f] = r
    moved = [v for v in plan["views"] if not (here and v is here[0])]
    if moved:
        res.update(run_views(moved, faces3d, model, sdir, "here"))
        ga.move(list(q_hover) + [25.0], speed=CFG["transit"], comp=True)
    out = {"where": "in_place", "measure": meas, "faces": res, "missing": plan["missing"], "n_views": len(plan["views"]),
           "neighbours": len(others), "inflate_mm": round(inflate, 1), "survey_s": round(time.time() - t0, 1)}
    log(f"ep {k}: survey here {len(plan['views'])} views in {out['survey_s']} s: " +
        " ".join(f"{f}={res.get(f, {}).get('verdict', '-')}" for f in srv.FACES) + f"; missing {plan['missing']}")
    return out


def survey_inspect(k):
    """Put the held brick down alone at the planned inspection spot, judge its top and four side faces from the
    planned views, re-pick it. Returns (survey record, ok); ok is False only when the brick was put down and the
    re-pick failed (the next cycle picks it up from the spot)."""
    plan = survey_plan()
    if plan is None:
        return {"error": "no survey plan (run plan_survey.py)"}, True
    model = fcs.load_model(FACE_MODEL_F)
    sp = srv.spec()
    t0 = time.time()
    p, _ = kin.fk(ga.state()["present"][:5])
    if p[2] < kin.MAT_Z + 80:
        pa.move_rel(ga.state()["present"][:5], 0.0, [0, 0, kin.MAT_Z + 85 - p[2]], speed=CFG["desc"])
    hi = kin.ik_rel(plan["q_hover"], [0, 0, 35.0])
    if hi is None:
        return {"error": "no pose above the inspection spot"}, True
    ga.move(hi + [0.0], speed=CFG["transit"], comp=True)
    qd = pa.move_rel(hi, 0.0, [0, 0, -(35.0 + pa.HOVER_DZ - pa.TIP_FINAL - 1.0)], speed=CFG["desc"])
    if qd is None:
        return {"error": "put-down IK failed at the inspection spot"}, True  # still held
    ga.cmd({"op": "grip", "value": 25.0, "speed": 25})
    time.sleep(0.3)
    prior_xy = [float(v) for v in kin.fk(qd)[0][:2]]
    yaw0 = kin.psi_of(qd)
    pa.move_rel(qd, 25.0, [0, 0, 40.0], speed=CFG["desc"])
    top = plan["views"]["top"]
    ga.move(list(top["q"]) + [25.0], speed=CFG["transit"], comp=False)
    img = pa.frame()
    qa = ga.state()["present"][:5]
    meas = srv.measure_top(img, qa, prior_xy, yaw0, sp) if img is not None else {"ok": False, "why": "no frame"}
    xy, yaw = (meas["xy"], meas["yaw"]) if meas.get("ok") else (prior_xy, yaw0)
    fp = sp["footprint_mm"]
    faces3d = srv.brick_faces(xy, yaw, (fp, fp), sp["height_mm"], kin.MAT_Z)
    sdir = f"{ga.OUT}/survey"
    os.makedirs(sdir, exist_ok=True)
    res = {}
    if img is not None:
        cv2.imwrite(f"{sdir}/spot_top.jpg", img)
        res.update({f: dict(r, view="spot_top") for f, r in judge_view(img, qa, faces3d, ["top"], model).items()})
    res.update(run_views([{"q": plan["views"][f]["q"], "faces": [f]} for f in srv.FACES if f != "top"],
                         faces3d, model, sdir, "spot"))
    out = {"where": "spot", "plan": plan.get("created_utc"), "spot_xy": plan.get("spot_xy"), "measure": meas,
           "faces": res, "survey_s": round(time.time() - t0, 1)}
    log(f"ep {k}: survey at the spot in {out['survey_s']} s: " + " ".join(f"{f}={res.get(f, {}).get('verdict', '-')}"
                                                                         for f in srv.FACES))
    ga.move(list(plan["q_hover"]) + [25.0], speed=CFG["transit"], comp=True)
    want = EP.get("target_colour")
    bh = pa.find(pa.blobs(pa.frame(), min_area=pa.SERVO_MIN_AREA), pa.target(), 260, want)
    if bh is None:
        return out, False
    q, bh, okc = pa.servo(list(plan["q_hover"]), 25.0, bh["centre"], log=log, want=want)
    if bh is None or not okc:
        return out, False
    q, rot = pa.align_yaw(q, 25.0, bh, log=log)
    if rot:
        bh = pa.find(pa.blobs(pa.frame()), pa.target(), 260, want, short_ref=bh["short"])
        if bh is None:
            return out, False
        q, bh, okc = pa.servo(q, 25.0, bh["centre"], log=log, want=want)
        if bh is None or not okc:
            return out, False
    ok, _ = grasp(q, bh, tag="re_")
    return out, ok


def merged_faces():
    """Face verdicts of this episode: the spot survey (complete, alone) overrides the in-place one."""
    f = dict((EP.get("survey") or {}).get("faces", {}))
    f.update((EP.get("survey_spot") or {}).get("faces", {}))
    return f
'''

patch(f"{T}/sort_any.py", [
    ("import closeloc as cl", "import closeloc as cl\nimport faces as fcs  # noqa: E402\nimport survey as srv  # noqa: E402\nfrom paths import APP  # noqa: E402"),
    ("\n\ndef tmove(q, speed, comp=False, accel=120.0):\n", SURVEY_FN + "\n\ndef tmove(q, speed, comp=False, accel=120.0):\n"),
    ("    global SCAN_C, SCAN_H, SCAN_HULL\n", "    global SCAN_C, SCAN_H, SCAN_HULL, SCAN_RESID_MM\n"),
    ("                SCAN_HULL = cv2.convexHull(np.array(PX, np.float32))\n",
     "                SCAN_HULL = cv2.convexHull(np.array(PX, np.float32))\n                SCAN_RESID_MM = float(np.median(res))\n"),
    ("    if len(PX) < 4:\n        SCAN_C = np.eye(2)\n        return\n",
     "    if len(PX) < 4:\n        SCAN_C = np.eye(2)\n        SCAN_RESID_MM = None\n        return\n"),
    ("    SCAN_C = C if float(np.median(res)) <= 15 else np.eye(2)\n",
     "    SCAN_C = C if float(np.median(res)) <= 15 else np.eye(2)\n    if SCAN_H is None:\n        SCAN_RESID_MM = float(np.median(res))\n"),
    ('    EP["hover_q"], EP["hover_blob"] = q, bh\n    ok, info = grasp(q, bh)\n',
     '    EP["hover_q"], EP["hover_blob"] = q, bh\n    EP["survey"] = survey_here(k, q, bh, b, cand)\n    ok, info = grasp(q, bh)\n'),
    ('''    if shape[0] == "2x2" and pre.get("verdict") not in ("CONFIDENT_DEFECT", "CONFIDENT_GOOD"):
        sv, ok2 = station_inspect(k)
        views = [sv]
        ffv = EP.get("first_frame_view")
        if ffv and (pre.get("method") != "diff_first_frame" or pre.get("bbox") != ffv.get("bbox")):
            views.append(ffv)  # the picked brick as first seen, when the pre-grasp view did not already judge it
        EP["station"] = sv
        if not ok2:
            EP.update({"success": False, "failure": "re-pick at station failed"})
            log(f"ep {k}: station re-pick failed")
            go_scan(False)
            ga.cmd({"op": "rec_stop"})
            return
    T["inspect_s"] = round(time.time() - t1, 1)
    clean = [v for v in views if not any(str(r).startswith(("merged", "colour_split")) for r in v.get("reasons", []))]
    pre = EP.get("pre_close") or {"verdict": "UNCERTAIN"}
    colour = pre.get("colour") or next((v.get("colour") for v in clean if v.get("colour")), None) or "unknown"
    EP["views_used"] = len(clean)
    plate, label, conf, reasons = sl.decide(pre, shape, clean)''',
     '''    have = merged_faces()
    if shape[0] == "2x2" and pre.get("verdict") != "CONFIDENT_DEFECT" and \\
            not any(r.get("verdict") == "DEFECT" for r in have.values()) and \\
            any(have.get(f, {}).get("verdict") != "CLEAN" for f in srv.FACES):
        surv, ok2 = survey_inspect(k)
        EP["survey_spot"] = surv
        if not ok2:
            EP.update({"success": False, "failure": "re-pick at the inspection spot failed"})
            log(f"ep {k}: re-pick at the inspection spot failed")
            go_scan(False)
            ga.cmd({"op": "rec_stop"})
            return
    T["inspect_s"] = round(time.time() - t1, 1)
    pre = EP.get("pre_close") or {"verdict": "UNCERTAIN"}
    colour = pre.get("colour") or EP.get("target_colour") or "unknown"
    plate, label, conf, reasons = sl.decide_faces(pre, shape, merged_faces(), srv.FACES)'''),
    ('''        sv, ok2 = station_inspect(EP.get("episode", 0))
        views = [sv]
        if not ok2:
            log("held brick put down at the station; re-pick failed - the next cycle picks it up")
            return None
        pre = EP.get("pre_close") or {"verdict": "UNCERTAIN"}
        clean = clean_views(views)
        shape = (EP.get("shape") or EP.get("re_shape") or "unknown", EP.get("shape_why") or EP.get("re_shape_why") or "")
        plate, label, conf, reasons = sl.decide(pre, shape, clean)
        colour = pre.get("colour") or next((v.get("colour") for v in clean if v.get("colour")), None) or "unknown"
        EP.update({"views": views,''',
     '''        views = []
        have = merged_faces()
        if not any(r.get("verdict") == "DEFECT" for r in have.values()) and \\
                any(have.get(f, {}).get("verdict") != "CLEAN" for f in srv.FACES):
            surv, ok2 = survey_inspect(EP.get("episode", 0))
            EP["survey_spot"] = surv
            if not ok2:
                log("held brick put down at the inspection spot; re-pick failed - the next cycle picks it up")
                return None
        pre = EP.get("pre_close") or {"verdict": "UNCERTAIN"}
        shape = (EP.get("shape") or EP.get("re_shape") or "unknown", EP.get("shape_why") or EP.get("re_shape_why") or "")
        plate, label, conf, reasons = sl.decide_faces(pre, shape, merged_faces(), srv.FACES)
        colour = pre.get("colour") or EP.get("target_colour") or "unknown"
        EP.update({"views": views,'''),
])
