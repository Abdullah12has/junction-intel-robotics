#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 12: write the laptop-side deliverables (status, gate results, clock offsets, metrics).

No camera passes G3, so the camera files carry status fields and evidence pointers, not usable parameters. The wrist
entry carries the README values, labelled as not refit and not passing. Numbers are copied from the fit and diagnostic
logs in out/ (feat_*, feat2_*, feat3_*, diag_*.log) and from out/clock_offsets_stage2.json.
Run: envs/data-env/bin/python -I tools/s12_write_deliverables.py
"""
import datetime
import json
import os

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-CAMS/E1"
CAMS = f"{R}/sim/twin/cameras"
UTC = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
STAGE = "twin-cams-v1"

GATE = {"median_px_max": 3.0, "p90_px_max": 6.0, "protocol": "leave-one-episode-out (not reached)"}

TOP = {
    "camera": "top", "stage": STAGE, "utc": UTC, "status": "NOT_FITTED",
    "G3": {"gate": GATE, "result": "FAIL", "loeo_run": False,
           "reason": "no candidate reaches the in-sample median gate; best in-sample median 36.9 px"},
    "fitted_parameters": None,
    "best_rejected_candidate": {"method": "s9d: tip offsets, tip-frame rotation, joint scale multipliers",
                                "in_sample_median_px": 36.918, "p90_px": 102.555, "n_obs": 702,
                                "rejected_because": "scales 0.40, 0.41, 0.68, 0.79 on four joints are implausible; "
                                                    "median far above gate"},
    "depth_crosscheck": {"reference": "P3-TWIN-SCENE (depth)", "camera_to_mat_mm": 589.46, "tilt_deg": 20.32,
                         "result": "not matched: best diagnostic candidates place the camera 127 to 147 mm above "
                                   "the mat (kin convention, zero) and 69 to 121 mm (wrist joint offsets)"},
    "sources": {"detections": "out/detect_top.json (383 still windows, 319 two-jaw)",
                "joints": "logs/sort/any_*/ep_*/joints.csv",
                "fixed_tip_geometry": "out/jaw_geometry.json (vendor mesh bytes read from R/twin before its "
                                      "deletion at 19:48Z; md5s recorded in that file)"},
    "origin_note": "No file copied from R/twin; the side-by-side comparison with the other session was dropped (D-102).",
}

CLOSE = {
    "camera": "close", "stage": STAGE, "utc": UTC, "status": "NOT_FITTED",
    "G3": {"gate": GATE, "result": "FAIL", "loeo_run": False,
           "reason": "no candidate reaches the in-sample median gate; best in-sample median 52.7 px"},
    "fitted_parameters": None,
    "best_rejected_candidate": {"method": "s9d: tip offsets, tip-frame rotation, joint scale multipliers",
                                "in_sample_median_px": 52.659, "p90_px": 155.987, "n_obs": 233,
                                "rejected_because": "two joint scales at the bounds (1.6 and 0.4); median far above gate"},
    "depth_crosscheck": {"reference": "P3-TWIN-SCENE (depth)", "camera_to_mat_mm": 147.34, "tilt_deg": 50.78,
                         "result": "not matched: the best diagnostic candidates (12 to 17 of 233 observations) place "
                                   "the camera 260 to 530 mm above the mat"},
    "sources": {"detections": "out/detect_close.json (161 windows with jaws, 72 two-jaw)",
                "joints": "logs/sort/any_*/ep_*/joints.csv",
                "fixed_tip_geometry": "out/jaw_geometry.json (see top.json)"},
    "origin_note": "No file copied from R/twin.",
}

WRIST = {
    "camera": "wrist", "stage": STAGE, "utc": UTC, "status": "README_VALUES_NOT_PASSED",
    "G3": {"gate": GATE, "result": "FAIL (in-sample only; pixel LOEO not computed)",
           "in_sample": {"n": 126, "episodes": 8, "median_px": 4.05, "scan_frames_median_px": 3.8}},
    "values_from": "R/app/eyeinhand/wrist_cam_model.json (I-61, README run at 18:47Z); not refit by E1",
    "params": {"f": 392.8364, "cx": 336.6664, "cy": 232.9406, "k1": -0.15245,
               "rvec_tcp_cam": [-0.060581, -0.840449, 2.932187],
               "t_tcp_cam_mm": [0.113809, 61.841840, -24.125727],
               "joint_offsets_norm": [0.013158, -5.641743, -2.486231, -2.757761, 0.001345]},
    "runtime_loo": {"median_mm": 11.5, "max_mm": 24.3, "note": "millimetres, not pixels; not the G3 metric"},
    "clock_offset": None,
    "clock_note": "not computed (wrist segments need motion extraction from R/data/wrist/raw)",
}

G3_REPORT = {
    "task": "P3-TWIN-CAMS", "worker": "E1", "stage": STAGE, "utc": UTC, "gate": GATE,
    "result": {"top": "FAIL", "close": "FAIL", "wrist": "FAIL (in-sample only)"},
    "tests": [
        {"name": "s9_fit v2, first fold (LOEO)", "camera": "top", "train_median_px": 42, "test_median_px": 32,
         "verdict": "fail"},
        {"name": "s9_fit v2, first fold (LOEO)", "camera": "close", "train_median_px": 63, "test_median_px": 33,
         "verdict": "fail"},
        {"name": "s9b A: tip offsets +/-30 mm", "camera": "top", "n": 702, "median_px": 40.897, "p90_px": 100.592,
         "verdict": "fail"},
        {"name": "s9b B: + 5 joint offsets", "camera": "top", "n": 702, "median_px": 40.872, "p90_px": 102.737,
         "verdict": "fail"},
        {"name": "s9b A", "camera": "close", "n": 233, "median_px": 60.46, "p90_px": 149.553, "verdict": "fail"},
        {"name": "s9b B: + 5 joint offsets", "camera": "close", "n": 233, "median_px": 63.851, "p90_px": 149.202,
         "verdict": "fail"},
        {"name": "s9c A: + tip-frame rotation, offsets +/-0.10 m", "camera": "top", "n": 702, "median_px": 39.136,
         "p90_px": 98.77, "verdict": "fail"},
        {"name": "s9c A", "camera": "close", "n": 233, "median_px": 60.775, "p90_px": 150.797, "verdict": "fail"},
        {"name": "s9d A: + joint scale multipliers 0.4 to 1.6", "camera": "top", "n": 702, "median_px": 36.918,
         "p90_px": 102.555, "verdict": "fail"},
        {"name": "s9d A", "camera": "close", "n": 233, "median_px": 52.659, "p90_px": 155.987, "verdict": "fail"},
        {"name": "diag_kin: kin.CAL.rad_from_norm, zero DQ, global PnP-RANSAC on all blob candidates",
         "camera": "top", "candidates": 702, "best_inliers": 39, "best_f_px": 500, "best_inlier_median_px": 3.84,
         "verdict": "fail (camera 127 mm above mat)"},
        {"name": "diag_kin: kin convention, wrist joint offsets DQ", "camera": "top", "candidates": 702,
         "best_inliers": 40, "best_f_px": 400, "best_inlier_median_px": 3.53, "verdict": "fail (69 mm above mat)"},
        {"name": "diag_kin: kin convention, zero DQ", "camera": "close", "candidates": 233, "best_inliers": 17,
         "best_f_px": 400, "best_inlier_median_px": 2.27, "verdict": "fail"},
        {"name": "diag_kin: kin convention, wrist DQ", "camera": "close", "candidates": 233, "best_inliers": 17,
         "best_f_px": 400, "best_inlier_median_px": 2.17, "verdict": "fail"},
        {"name": "diag_global: mid and homing zero x 32 sign patterns (manual joint formula, not kin)",
         "camera": "top", "candidates": 702, "best_inliers": 65, "best_f_px": 500, "verdict":
         "fail (camera centre (-6.8, -4.9, 5.2) m; 365 of 384 combinations below 50 inliers)"},
        {"name": "wrist README, in-sample", "camera": "wrist", "n": 126, "median_px": 4.05, "verdict": "fail"},
    ],
    "invalid_evidence": ["diag_pnp, diag_pnp2, diag_sign, diag_zero, diag_episode, diag_dist: second blob paired with "
                         "the fixed-jaw tip (pairing bug); not used as evidence"],
    "crosscheck_depth": {"top": "not matched (fits near 0.07 to 0.15 m height, reference 0.589 m)",
                         "close": "not matched (fits 0.26 to 0.53 m, reference 0.147 m)"},
    "plate_cross_check": {"status": "not run as a G3 test", "plate_centres_px_first_pre_top": {
        "blue": [240.4, 147.8], "green": [576.3, 176.5]},
        "note": "image centres from ellipse fits; no camera model to project PLATES_XY into"},
    "brick_vanish_events": {"events": 6, "candidates_verified": 0, "note": "candidates on cables and gripper parts; "
                            "not usable as correspondences without verification"},
    "overlays_viewed": ["overlays/still_top_029.jpg", "overlays/still_close_185.jpg",
                        "overlays/sheet_top_two_0.jpg (20:13Z)", "overlays/sheet_top_two_1.jpg",
                        "overlays/sheet_close_two_0.jpg", "overlays/sheet_close_one_0.jpg",
                        "overlays/s11_events_overlay.jpg (20:18Z)"],
    "overlays_not_possible": "projected model overlays: no model passed G3",
}

CLOCK = {
    "stage": STAGE, "utc": UTC,
    "method": "cross-correlation of joint activity (log1p of max joint speed) with motion energy from "
              "out/motion_<cam>.npz; lags -0.5..+0.5 s; offset = video - joint (video_time = joint_time + offset). "
              "NOT the onset method in the spec: deviation.",
    "grid_s": round(1 / 30, 4),
    "top": {"per_episode_median_s": 0.0333, "per_episode_n": 14, "per_episode_spread": "MAD 0.0 (below one frame)",
            "per_event_corr_ge_0p6": {"n": 31, "of": 50, "median_s": 0.0333, "iqr_s": 0.0333,
                                      "p10_s": 0.0, "p90_s": 0.0333}},
    "close": {"per_episode_median_s": 0.0167, "per_episode_n": 8, "per_episode_spread": "MAD 0.0247 s",
              "per_event_corr_ge_0p6": {"n": 10, "of": 50, "median_s": 0.0333, "mad_s": 0.0494,
                                        "p10_s": -0.0033, "p90_s": 0.1533},
              "note": "episode lags 0 to 12 frames; low confidence"},
    "wrist": None,
    "wrist_note": "not computed",
    "quantisation": "one frame at 30 fps (33 ms); spreads below this are not resolved",
    "stage1_note": "out/clock_offsets_stage1.json has the sign inverted relative to the stated convention and is not used",
}

METRICS = {
    "stage": STAGE, "worker": "E1", "utc": UTC,
    "still_windows": {"segments": 383, "median_duration_s": 0.91,
                      "rule": "all six joints max-min < 0.5 units over >= 0.5 s"},
    "detections": {"top": {"windows_with_jaws": 383, "two_jaw_windows": 319, "observations": 702},
                   "close": {"windows_with_jaws": 161, "two_jaw_windows": 72, "observations": 233}},
    "G3": {"top": "FAIL", "close": "FAIL", "wrist": "FAIL (in-sample 4.05 px)"},
    "best_in_sample_median_px": {"top": 36.918, "close": 52.659, "wrist": 4.05},
    "clock_offsets_s": {"top_per_event_median": 0.0333, "close_per_event_median": 0.0333, "wrist": None},
    "plate_centres_px": {"blue": [240.4, 147.8], "green": [576.3, 176.5]},
    "brick_events": {"events": 6, "verified": 0},
    "kin_vs_scan_brick_mm": {"successful_picks": 4, "range_mm": [33.9, 49.3],
                             "note": "kin TCP at lift vs episode.json brick_xy_est; which one is right is not established"},
    "overlays_viewed": 7,
    "deliverables": {"cameras": ["sim/twin/cameras/top.json", "sim/twin/cameras/close.json",
                                 "sim/twin/cameras/wrist.json"],
                     "work": ["g3_report.json", "clock_offsets.json", "metrics.json"]},
}


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)
    print("wrote", path, flush=True)


def main():
    write(f"{CAMS}/top.json", TOP)
    write(f"{CAMS}/close.json", CLOSE)
    write(f"{CAMS}/wrist.json", WRIST)
    write(f"{E1}/g3_report.json", G3_REPORT)
    write(f"{E1}/clock_offsets.json", CLOCK)
    write(f"{E1}/metrics.json", METRICS)


if __name__ == "__main__":
    main()
