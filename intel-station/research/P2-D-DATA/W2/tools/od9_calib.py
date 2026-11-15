import json
import numpy as np

W = "/home/intel/intel-junction-2026/work/P2-D-DATA/W2/results3"
calib_p = f"{W}/calib.json"
review_p = f"{W}/review.json"

# Release (open event) joint poses of the put-down episodes, from segments.json
release = {
    "3": [-5.39, -60.67, 61.89, 66.51, -45.57, 8.49],
    "4": [-13.66, -60.58, 79.22, 43.2, -51.27, 8.75],
    "5": [-12.62, 20.99, -19.33, 69.73, -40.44, 9.67],
    "6": [-13.14, 12.61, 1.27, 56.25, -49.59, 8.43],
}
# Placed-brick centroid in the top view (243622061421), from spot_plates.py
placed = {
    "3": {"cx": 354.2, "cy": 123.1, "sx": 56.5, "sy": 44.0},
    "4": {"cx": 361.3, "cy": 129.1, "sx": 55.7, "sy": 46.5},
    "5": {"cx": 387.4, "cy": 182.3, "sx": 54.0, "sy": 97.3},
    "6": {"cx": 377.0, "cy": 170.6, "sx": 61.8, "sy": 71.3},
}
rel_arr = np.array(list(release.values()))
pl_arr = np.array([[v["cx"], v["cy"]] for v in placed.values()])

od9 = {
    "decision": "OD-9 (13:22Z): S1 = pick, put down on the inspection spot, return home",
    "inspection_spot": {
        "joint_pose_at_put_down": {
            "per_episode": release,
            "median": np.median(rel_arr, axis=0).round(2).tolist(),
            "spread_std": rel_arr.std(axis=0).round(2).tolist(),
            "note": "eps 3-4 release at shoulder_lift about -60; eps 5-6 at +12..+21; two groups, the median mixes them. Use per-group poses and verify live.",
        },
        "placed_brick_top_view_px": {
            "per_episode": placed,
            "median_cx_cy": np.median(pl_arr, axis=0).round(1).tolist(),
            "method": "changed pixels |end - start| > 60 (sum RGB), mat box rows 60-380 cols 120-520, centroid and std. Rough: includes arm motion; the spread is large.",
        },
        "placed_brick_low_view_px": "not measured in this pass",
    },
    "plate_drop_poses": {
        "blue_plate": {
            "centroid_top_px": {"cx": 242.0, "cy": 150.5, "method": "blue colour threshold, left half, end frames"},
            "drop_poses": {"ep7": [32.1, 5.47, 25.59, 61.85, -36.78, 8.62], "ep8": [-54.67, 24.51, 1.91, 54.22, -55.03, 8.95]},
            "evidence": "ep7 release frame: gripper over blue plate (verified image). ep8 rest frame: brick on blue plate (verified image); ep8 release frame not viewed.",
        },
        "green_plate": {
            "centroid_top_px": {"cx": 578.0, "cy": 182.0, "method": "green colour threshold, right half, end frames"},
            "drop_poses": {"ep10": [-51.83, 47.71, -49.82, 80.08, -52.31, 8.75]},
            "evidence": "ep10 release frame: gripper over green plate (verified image). Single episode.",
        },
    },
    "home_pose": {"median": None, "note": "see home_pose in this file (median of 11 episodes)"},
    "hover_8cm_above_spot": {
        "derivable": False,
        "reason": "no forward kinematics or camera-to-robot calibration in the recorded data; a hover pose needs the arm model and must be verified live at slow speed",
    },
    "s1_part_od9": {
        "definition": "start to home return: 0 .. rest_frame for episodes with a put-down and return",
        "episodes": {"3": {"end_frame": 644, "end_s": 21.47}, "5": {"end_frame": 618, "end_s": 20.6}, "6": {"end_frame": 479, "end_s": 15.97}},
        "episode_4": "put-down but no return to home inside the episode; usable up to the put-down (frame 634, 21.13 s)",
    },
}

calib = json.load(open(calib_p))
calib["od9"] = od9
json.dump(calib, open(calib_p, "w"), indent=1)

rev = json.load(open(review_p))
ds = "6d6bdce3-b382-478d-9697-fb9e573b97af"
r = rev[ds]
for ep in ("3", "4", "5", "6"):
    r[ep]["plate"] = "none"
    r[ep]["release_clear"] = "yes"
    r[ep]["notes"] = "OD-9 put-down on the mat (gripper not over either plate at release); spot candidate; return home present except ep4"
r["7"]["plate"] = "blue"
r["8"]["plate"] = "blue"
r["8"]["release_clear"] = "yes"
r["8"]["notes"] = "starts with brick held; rest frame shows brick on blue plate (release frame not viewed)"
r["10"]["plate"] = "green"
r["10"]["release_clear"] = "yes"
r["10"]["notes"] = "starts with brick held; gripper over green plate at release; orange brick already on blue plate at start"
rev[ds] = r
rev["_scene_notes"] = [
    "Top camera 243622061421: blue plate upper-left (centre about 242,150 px), green plate upper-right (about 578,182 px); mat bricks clustered in the centre.",
    "An orange brick already sits on the blue plate in ep10 at start; ep8 rest frame shows a brick on the blue plate.",
]
json.dump(rev, open(review_p, "w"), indent=1)
print("OK", sorted(calib.keys())[-3:], sorted(rev.keys()))
