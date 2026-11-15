import json
import sys

W = "/home/intel/intel-junction-2026/work/P2-B-OVRUN2/E1"
S = json.load(open(f"{W}/results/f06a253e_GPU_summary.json"))
F = json.load(open(f"{W}/results/f06a253e_GPU_full.json"))
E = {(r["set"], r["episode_index"]): r for r in S["episodes"]}
n_ok = [0, 0]


def chk(name, got, exp, tol=0.006):
    if isinstance(exp, list):
        ok = len(got) == len(exp) and all(abs(a - b) <= tol for a, b in zip(got, exp))
    elif isinstance(exp, str):
        ok = got == exp
    else:
        ok = got is not None and abs(got - exp) <= tol
    n_ok[0 if ok else 1] += 1
    print("PASS" if ok else "FAIL", name, got, exp)


chk("verdict", S["verdict"], "NOT SANE")
chk("tr0 exec", E[("trainBlueV1", 0)]["exec_mae_mean"], 2.177)
chk("tr0 hold", E[("trainBlueV1", 0)]["hold_mae_mean"], 1.284)
chk("tr1 exec", E[("trainBlueV1", 1)]["exec_mae_mean"], 3.267)
chk("tr1 hold", E[("trainBlueV1", 1)]["hold_mae_mean"], 1.39)
chk("agg train exec", S["aggregate_by_role"]["train"]["exec_mae_mean"], 2.735)
chk("agg train hold", S["aggregate_by_role"]["train"]["hold_mae_mean"], 1.338)
chk("agg train exec per joint", S["aggregate_by_role"]["train"]["exec_mae_per_joint"], [2.146, 5.174, 4.919, 1.896, 0.854, 1.421])
chk("agg train hold per joint", S["aggregate_by_role"]["train"]["hold_mae_per_joint"], [0.752, 2.097, 2.634, 1.25, 0.31, 0.988])
chk("held blue exec", E[("heldBlueV2", 0)]["exec_mae_mean"], 12.218)
chk("held blue hold", E[("heldBlueV2", 0)]["hold_mae_mean"], 1.27)
chk("green0 exec", E[("heldGreenV2", 0)]["exec_mae_mean"], 17.019)
chk("green1 exec", E[("heldGreenV2", 1)]["exec_mae_mean"], 14.254)
chk("green0 hold", E[("heldGreenV2", 0)]["hold_mae_mean"], 1.331)
chk("green1 hold", E[("heldGreenV2", 1)]["hold_mae_mean"], 1.328)
chk("tr0 pan rec", E[("trainBlueV1", 0)]["pan"]["rec_disp"], 63.86)
chk("tr0 pan pred", E[("trainBlueV1", 0)]["pan"]["pred_disp"], 69.62)
chk("tr1 pan rec", E[("trainBlueV1", 1)]["pan"]["rec_disp"], 49.96)
chk("tr1 pan pred", E[("trainBlueV1", 1)]["pan"]["pred_disp"], 54.9)
chk("held blue pan rec", E[("heldBlueV2", 0)]["pan"]["rec_disp"], 46.02)
chk("held blue pan pred", E[("heldBlueV2", 0)]["pan"]["pred_disp"], 45.81)
chk("green0 pan rec", E[("heldGreenV2", 0)]["pan"]["rec_disp"], -26.1)
chk("green0 pan pred", E[("heldGreenV2", 0)]["pan"]["pred_disp"], 13.55)
chk("green1 pan rec", E[("heldGreenV2", 1)]["pan"]["rec_disp"], -18.22)
chk("green1 pan pred", E[("heldGreenV2", 1)]["pan"]["pred_disp"], 14.49)
chk("tr0 events rec", E[("trainBlueV1", 0)]["rec_gripper_events_frames"], [0, 236, 541], 0)
chk("tr0 events pred", E[("trainBlueV1", 0)]["pred_gripper_events_frames"], [0, 246, 531], 0)
chk("tr1 events rec", E[("trainBlueV1", 1)]["rec_gripper_events_frames"], [0, 423, 679], 0)
chk("tr1 events pred", E[("trainBlueV1", 1)]["pred_gripper_events_frames"], [0, 423, 691], 0)
chk("held blue events rec", E[("heldBlueV2", 0)]["rec_gripper_events_frames"], [0, 320, 1177], 0)
chk("held blue events pred", E[("heldBlueV2", 0)]["pred_gripper_events_frames"], [0, 463, 503], 0)
chk("green0 events pred", E[("heldGreenV2", 0)]["pred_gripper_events_frames"], [0, 260, 288], 0)
chk("green1 events rec", E[("heldGreenV2", 1)]["rec_gripper_events_frames"], [0, 357, 488], 0)
chk("latency update p50", S["latency_update_at_query_ticks_all"]["p50_ms"], 5.52)
chk("latency update p95", S["latency_update_at_query_ticks_all"]["p95_ms"], 10.55)
chk("latency update max", S["latency_update_at_query_ticks_all"]["max_ms"], 14.21)
chk("latency update n", S["latency_update_at_query_ticks_all"]["n"], 102, 0)
chk("latency chunk p50", S["latency_chunk_infer_all"]["p50_ms"], 5.33)
chk("latency chunk p95", S["latency_chunk_infer_all"]["p95_ms"], 10.4)
chk("latency chunk max", S["latency_chunk_infer_all"]["max_ms"], 14.03)
chk("blue ref", S["plate_pan_refs"]["blue_median_recorded_release_pan"], 33.822)
chk("green ref", S["plate_pan_refs"]["green_median_recorded_release_pan"], -52.432)
dc = {(c["query"], c["train"]): c["mean_abs_state_diff"] for c in S["duplicate_check_v2_vs_train"]}
chk("dup ep1 vs tr0", dc[("heldBlueV2 ep1", "trainBlueV1 ep0")], 0.0)
chk("dup ep2 vs tr1", dc[("heldBlueV2 ep2", "trainBlueV1 ep1")], 0.0)
chk("held0 min diff", min(dc[("heldBlueV2 ep0", "trainBlueV1 ep0")], dc[("heldBlueV2 ep0", "trainBlueV1 ep1")]), 18.3082, 0.0001)
chk("duplicate flags", S["duplicate_flags_heldBlueV2"], {"0": False, "1": True, "2": True})
print("SUMMARY pass", n_ok[0], "fail", n_ok[1])
