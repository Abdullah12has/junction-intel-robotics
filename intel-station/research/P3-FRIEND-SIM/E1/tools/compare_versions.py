#!/usr/bin/env python3
"""Compare two inventory JSONs (data only). Usage: python3 -I compare_versions.py NEW.json OLD.json OUT.json"""
import collections, json, os, sys
import numpy as np

NEW = json.load(open(sys.argv[1]))
OLD = json.load(open(sys.argv[2]))
FILES = ("metadata.json", "trajectory.npz", "trace.json", "overhead.mp4", "wrist.mp4")
na = {a["name"]: a for a in NEW["attempts_detail"]}
oa = {a["name"]: a for a in OLD["attempts_detail"]}
common = sorted(set(na) & set(oa))
out = {"new_root": NEW["root"], "old_root": OLD["root"],
       "attempts": {"new": len(na), "old": len(oa), "common": len(common),
                    "only_new": sorted(set(na) - set(oa)), "only_old": sorted(set(oa) - set(na))},
       "tar_sha_equal": NEW["tar_sha256"] == OLD["tar_sha256"], "tar_sha256": NEW["tar_sha256"],
       "manifest_new": NEW["manifest_present"], "manifest_old": OLD["manifest_present"],
       "raw_top_entries_new": NEW.get("raw_top_entries"), "raw_top_entries_old": OLD.get("raw_top_entries")}
out["file_hash"] = {}
for f in FILES:
    changed = [a for a in common if na[a]["sha"].get(f) != oa[a]["sha"].get(f)]
    out["file_hash"][f] = {"identical": len(common) - len(changed), "changed": len(changed), "changed_attempts": changed}
md_diff, md_ex = collections.Counter(), {}
for a in common:
    for k in sorted(set(na[a]["md"]) | set(oa[a]["md"])):
        if na[a]["md"].get(k) != oa[a]["md"].get(k):
            md_diff[k] += 1
            md_ex.setdefault(k, [a, str(oa[a]["md"].get(k))[:160], str(na[a]["md"].get(k))[:160]])
out["metadata_fields_changed"] = dict(md_diff)
out["metadata_field_examples"] = md_ex
keys = ["task", "target_color", "destination", "brick_type", "source", "action_units", "accepted", "success",
        "physical_grasp_verified", "hardware_mapping_verified", "error", "mujoco_version", "reference_layout",
        "friction_provenance", "fps", "frames"]
out["md_values_new"] = {k: NEW["md_scalar_values"].get(k) for k in keys}
out["md_values_old"] = {k: OLD["md_scalar_values"].get(k) for k in keys}
rn, ro = NEW.get("recording_context") or {}, OLD.get("recording_context") or {}
out["recording_context_keys_new"] = sorted(rn) if isinstance(rn, dict) else None
out["recording_context_changed_keys"] = sorted(k for k in set(rn) | set(ro) if rn.get(k) != ro.get(k)) if isinstance(rn, dict) and isinstance(ro, dict) else None
sn, so = NEW.get("source_files") or {}, OLD.get("source_files") or {}
out["source_files_new"] = sn
out["source_files_changed"] = sorted(k for k in set(sn) | set(so) if sn.get(k) != so.get(k))
fn = {tuple(x) for x in NEW["files_outside_raw"]}
fo = {tuple(x) for x in OLD["files_outside_raw"]}
out["outside_raw_only_new"] = sorted(fn - fo)
out["outside_raw_only_old_count"] = len(fo - fn)
out["outside_raw_only_old_examples"] = sorted(fo - fn)[:6]


def od10_bad(a):
    b, d = a["md"].get("brick_type"), a["md"].get("destination")
    return (b == "2x2" and d == "green") or (b != "2x2" and d == "blue")


def pairs(att):
    return dict(collections.Counter("%s->%s" % (a["md"].get("brick_type"), a["md"].get("destination")) for a in att.values()))


out["brick_destination_pairs_new"] = pairs(na)
out["brick_destination_pairs_old"] = pairs(oa)
out["od10_contradictions_new"] = sum(od10_bad(a) for a in na.values())
out["od10_contradictions_old"] = sum(od10_bad(a) for a in oa.values())
# trajectory payloads: are the changed npz arrays different in value?
cmp_npz = {"checked": 0, "same_shape": 0, "max_abs_diff": 0.0, "attempts_differing": []}
for a in out["file_hash"]["trajectory.npz"]["changed_attempts"]:
    zn = np.load(os.path.join(NEW["raw_dir"], a, "trajectory.npz"))
    zo = np.load(os.path.join(OLD["raw_dir"], a, "trajectory.npz"))
    cmp_npz["checked"] += 1
    if zn["state"].shape == zo["state"].shape:
        cmp_npz["same_shape"] += 1
        d = float(np.max(np.abs(zn["state"].astype(float) - zo["state"].astype(float))))
        cmp_npz["max_abs_diff"] = max(cmp_npz["max_abs_diff"], d)
        if d > 0:
            cmp_npz["attempts_differing"].append(a)
out["trajectory_value_compare"] = cmp_npz
with open(sys.argv[3], "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(json.dumps({"attempts": out["attempts"], "tar_sha_equal": out["tar_sha_equal"], "file_hash": {k: (v["identical"], v["changed"]) for k, v in out["file_hash"].items()},
                  "metadata_fields_changed": out["metadata_fields_changed"], "recording_context_changed_keys": out["recording_context_changed_keys"],
                  "source_files_changed": out["source_files_changed"], "outside_raw_only_new": out["outside_raw_only_new"],
                  "outside_raw_only_old_count": out["outside_raw_only_old_count"], "raw_top_new": out["raw_top_entries_new"],
                  "raw_top_old": out["raw_top_entries_old"], "pairs_new": out["brick_destination_pairs_new"],
                  "pairs_old": out["brick_destination_pairs_old"], "od10_new": out["od10_contradictions_new"],
                  "od10_old": out["od10_contradictions_old"], "npz": {k: v for k, v in cmp_npz.items() if k != "attempts_differing"},
                  "npz_differing_n": len(cmp_npz["attempts_differing"])}, indent=1, default=str)[:5000])
