import json
import sys

s = json.load(open(sys.argv[1]))
print("verdict", s["verdict"], "| reasons", s["reasons"], "| load_s", s["load_s"], "| device", s["device"])
print("latency query update", s["latency_update_at_query_ticks_all"], "| chunk infer", s["latency_chunk_infer_all"])
for r in s["episodes"]:
    p = r["pan"]
    print(r["set"], "ep", r["episode_index"], r["role"], "L", r["length"], "q", r["n_queries"],
          "exec", r["exec_mae_mean"], "hold", r["hold_mae_mean"], "beats", r["beats_hold"],
          "dt_close_s", r["dt_close_s"], "dt_release_s", r["dt_release_s"],
          "rec_ev", r["rec_gripper_events_frames"], "pred_ev", r["pred_gripper_events_frames"],
          "rec_disp", p.get("rec_disp"), "pred_disp", p.get("pred_disp"), "side_ok", p.get("side_ok"),
          "near_blue", p.get("pred_release_nearer_blue_ref"), "lat_q", r["latency_update_at_query_ticks"])
for role, a in s["aggregate_by_role"].items():
    print("AGG", role, "steps", a["steps"], "exec", a["exec_mae_mean"], "hold", a["hold_mae_mean"], "exec_j", a["exec_mae_per_joint"], "hold_j", a["hold_mae_per_joint"])
