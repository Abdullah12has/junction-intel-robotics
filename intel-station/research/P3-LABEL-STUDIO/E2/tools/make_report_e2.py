import hashlib, json, sys
R = "/home/intel/intel-junction-2026"
E2 = R + "/work/P3-LABEL-STUDIO/E2"
E1 = R + "/work/P3-LABEL-STUDIO/E1"
RUN_UTC = sys.argv[1]          # actual UTC stamp of this run, e.g. 2026-11-14T20:58Z

lab = [json.loads(l) for l in open(E2 + "/labels_episodes.jsonl", encoding="utf-8")]
e1_raw = open(E1 + "/labels_episodes.jsonl", "rb").read()
e2_raw = open(E2 + "/labels_episodes.jsonl", "rb").read()
assert e2_raw.startswith(e1_raw), "E1 rows changed"
assert len(lab) == 52 and len({r["entry"] for r in lab}) == 52

cen = json.load(open(E2 + "/out/census_e2.json"))
info_ok = []
for k, d in cen["datasets"].items():
    info_ok.append((k[:8], d["total_episodes"] == d["meta_rows"] == d["episodes_in_data"], d["total_frames"] == d["data_rows"]))
print("info totals vs rows (prefix, episodes ok, frames ok):", info_ok)

def tr(r):
    return r["trust"]
cnt = {}
for r in lab:
    cnt[tr(r)] = cnt.get(tr(r), 0) + 1
phys = {}
for r in lab:
    phys.setdefault(r["physical_episode"], set()).add(tr(r))
print("entry trust", cnt)
print("physical count", len(phys))

def sha(b):
    return hashlib.sha256(b).hexdigest()

# ---- generated tables ----
new = json.load(open(E2 + "/out/new_rows_e2.json"))
rep_rows = {r["entry"]: r for r in new if r["entry"].startswith("8a8bb05e")}
T = []
T.append("## 4. New physical episodes in 8a8bb05e (P18 to P23), per brick")
T.append("Each episode starts with 7 bricks on the table and ends with the bricks on the BLUE plate (task text: pick up all the blocks and put them on the blue plate). OD-10: clean 2x2 goes to BLUE, tape or non-2x2 goes to GREEN. Tape from top-view zoom grids (out/zoom_8a8bb05e_start.png); placement from end-state zoom (out/zoom_8a8bb05e_end.png).")
for key in ["8a8bb05e:%d" % i for i in range(6)]:
    r = rep_rows[key]
    T.append("")
    T.append("### %s (%s), %d frames, %.1f s, outcome: %s" % (key, r["physical_episode"], r["n_frames"], r["dur_s"], r["outcome"]))
    T.append("| brick | colour | 2x2 | tape | placement | OD-10 expected | decision correct |")
    T.append("|---|---|---|---|---|---|---|")
    for b in r["bricks"]:
        T.append("| %s | %s | %s | %s | %s | %s | %s |" % (b["id"], b["colour"], "yes" if b["shape"] == "2x2" else "no (%s)" % b["shape"], b["tape"], b["placement"], b["od10_expected"], b["decision_correct"]))
    T.append("")
    T.append("Trust: evaluation. Why: %s. Joint data: lag %d frames (MAE %.3f), gripper %s, limit hits: %s." % (
        r["trust_reason"], r["joint_sanity"]["lag_best_frames"], r["joint_sanity"]["mae_lag_best"],
        r["joint_sanity"]["gripper_range"], "; ".join(r["joint_sanity"]["saturated_at_normalized_limit"]) or "none"))
TABLES_NEW = "\n".join(T)

P = []
P.append("## 5. Per-entry trust (52 entries)")
P.append("| entry | P | trust | pixel check vs representative | outcome | decision correct | task text |")
P.append("|---|---|---|---|---|---|---|")
for r in lab:
    fc = r.get("frame_check_vs_rep") or {}
    pc = fc.get("frame_verdict", "own frames")
    P.append("| %s | %s | %s | %s | %s | %s | %s |" % (r["entry"], r["physical_episode"], r["trust"], pc, str(r["outcome"])[:40], str(r["decision_correct"])[:30], str(r["task_text_ok"])[:30]))
TABLES_ENTRY = "\n".join(P)

open(E2 + "/out/_tables_new.md", "w", encoding="utf-8").write(TABLES_NEW + "\n")
open(E2 + "/out/_tables_entries.md", "w", encoding="utf-8").write(TABLES_ENTRY + "\n")
print("tables written", len(TABLES_NEW), len(TABLES_ENTRY))
print("RUN_UTC", RUN_UTC)
