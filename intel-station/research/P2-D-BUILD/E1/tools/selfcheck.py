import json
import zipfile

L = "/home/intel/intel-junction-2026"
W = L + "/work/P2-D-BUILD/E1"
cam = "observation.images.realsense 243722064316"
src = json.load(open(L + "/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af/meta/info.json", encoding="utf-8"))
si = src["features"][cam]["info"]
print("SRC total_episodes", src["total_episodes"], "total_frames", src["total_frames"], "fps", src["fps"], "robot_type", src["robot_type"])
print("SRC cam codec/g/crf/preset/pix_fmt", si["video.codec"], si["video.g"], si["video.crf"], si["video.preset"], si["video.pix_fmt"])
out = json.load(open(L + "/data/curated/S1_v0/meta/info.json", encoding="utf-8"))
oi = out["features"][cam].get("info", {})
print("OUT total_episodes", out["total_episodes"], "total_frames", out["total_frames"], "fps", out["fps"], "robot_type", out["robot_type"])
print("OUT cam codec/g/crf/preset/pix_fmt", oi.get("video.codec"), oi.get("video.g"), oi.get("video.crf"), oi.get("video.preset"), oi.get("video.pix_fmt"))
print("OUT feature keys", sorted(out["features"].keys()))
print("SRC feature keys", sorted(src["features"].keys()))
z = zipfile.ZipFile(L + "/data/curated/S1_v0.zip")
print("ZIP entries", len(z.namelist()), "testzip", z.testzip(), sorted(z.namelist()))
v = json.load(open(W + "/results/validate_v0.json", encoding="utf-8"))
s1 = v["classes"]["S1_v0"]
print("VAL all_pass", v["all_pass"], "S1 pass", s1["pass"], "episodes", s1["episodes"], "frames", s1["frames"], "problems", s1["problems"])
j = json.load(open(W + "/logs/import_s1_job_final.json", encoding="utf-8"))
print("JOB", j["id"], j["status"], j["payload"]["step"], j["payload"]["result_dataset_id"], j["payload"]["dataset_name"], j["payload"]["finalize_input"])
b = json.load(open(W + "/results/build_v0.json", encoding="utf-8"))
print("BUILD per-episode (source_ep, frames)", [(e["source_episode"], e["frames"]) for e in b["classes"]["S1_v0"]["per_episode"]])
print("VAL per-episode exact (action, state)", [(e["source_episode"], e["exact"]["action"], e["exact"]["observation.state"]) for e in s1["per_episode"]])
