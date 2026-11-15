PARTIAL 2026-11-14T20:30Z align-v1: deliverables written on both copies; mentor answers not in allowed inputs; no closed-loop run; Challenge Day date and submission format not in the materials

## Done
- requirements.json: 77 entries (73 rated rows; MD-05, OQ-01, OQ-02 and OQ-03 are background or open). Identical on both copies, sha256 c6c921358955e857eaa8b174fe59ea9e68f86d22e7bcd34bbd6d42d43e1a386f.
- metrics.json: stage align-v1, utc 2026-11-14T20:17:23Z, written by check_align.py on the laptop. Identical on both copies, sha256 41b8257b00c19a69dc5d18440ee68b58909dbe42c60e81ca152ed4331e853686.
- alignment.md: 266 lines, 41929 bytes, 77 ID rows. Identical on both copies, sha256 d5ff77a2d86e836ffcbca3e5085783769c63e59e1e7eef9c52af53a98de3331a.
- Checker run on the laptop after the final patch: bad []; quote_text_mismatch 0; quotes_over_15_words 0.
- Sources read: guide JUNCTION2026.md (all 140 lines); brief by text layer (slides 1-3, 8-15) and rendered images (slides 4-7); architecture.md; policy_rules.yaml; anomalib metrics.json; FACTS-LIVE.md; T09B-MODAL-BROWSER.md (grep); models.tsv rows 2-6; RULES.md; LAPTOP.md.

## Corrections in this run
- GD-13 (live GPU use for ACT) downgraded from met to partial. RULES plates say device auto (lines 32, 41); architecture.md lines 77 and 108 say auto resolves to CPU under D-080.
- GD-21 added (partial): connect detection to action (guide Step 6). No live connection run recorded.
- GD-22 added (missing): test a trained model on the arm from the Studio Models page with Run model (guide line 114). No record. A motion run, so Opus only.
- GD-05 downgraded to partial: no PaDiM decision in architecture.md. The earlier PaDiM benchmark claim was wrong and is removed.
- OQ-03 evidence updated: fusion evidence rests on BR-08, BR-14, BR-25, JC-01 to JC-03.
- Earlier in this run: BR-07, BR-11, GD-09 and GD-16 added; BG-01 unused.

## Numbers
- Rated rows 73: met 17, partial 49, missing 3 (BR-22, BR-27, GD-22), unknown 3 (MD-02, MD-03, MD-04), not applicable 1 (MD-01).
- Background or open 4: MD-05 met; OQ-01 open (conflict); OQ-02 and OQ-03 open.
- Quotes: 48 text checked on the laptop, 5 sandbox-text checked in the sandbox, 0 mismatches. 24 image quotes transcribed from rendered pages, not machine-checked. None over 15 words.
- IDs: BR-01 to BR-34, JC-01 to JC-06, RB-01 to RB-05, GD-01 to GD-22, MD-01 to MD-05, BG-02, BG-03, OQ-01 to OQ-03 (77).

## Self-check (R12)
- 20 facts re-verified against source, each with its method: alignment.md, Self-check section (line 226). Methods: grep -n, awk page split, sed -n, checker run (laptop-log entries 71, 77), sandbox greps.
- Corrections from the self-check: GD-13 downgraded; GD-21 and GD-22 added; PaDiM claim removed; BR-28 and GD-16 citations confirmed. The earlier 19 met included MD-05, so the rated count is 17 met after the GD-13 change.

## Coverage checklist
- alignment.md, Coverage checklist (line 167). The requirement table has 77 ID rows (grep count 77).

## Next
- Final sha256 check on both folders, then SubagentHandback. No further E1 work unless Opus answers the requests below.

## Blockers
- OQ-03: mentor answers not read. MENTOR-ANSWERS.md is outside the allowed inputs and was not opened.
- OQ-02: Challenge Day date and main submission format are not in the materials.
- BR-22: no closed-loop run recorded. The 16:09Z spot test did not grasp the brick (FACTS-LIVE line 36). Any arm motion needs Opus under the robot motion rule; E1 has not commanded the arm.

## Open issues
- Out-of-scope call: E1 called mcp__claude-code-remote__list_triggers once in error (a scheduling-listing tool). Its output was not used; nothing was created, changed or enabled.
- SmolVLA lineage: FACTS-LIVE lines 11 and 20 name 6540166d; models.tsv row 5 is d128dc1a, and no FACTS-LIVE line names it (BG-02).
- Dataset episode counts: FACTS-LIVE 15:03Z (PP-BLUE-v2 3, PP-GREEN-v2 2, MT-ALL-v2 9) against models.tsv labels (RB-03, GD-20).
- Task strings: FACTS-LIVE 14:00Z against policy_rules.yaml lines 32 and 41 (GD-20).
- Routing: FACTS-LIVE line 1 and OD-10 (good to blue, defective to green) against the team note in FACTS-LIVE line 20 (good to green bowl, defective to red bowl).
- Serial ports: FACTS-LIVE line 1 and line 19 disagree on the ttyACM numbers. Use the by-id paths (LAPTOP.md rule 8).
- Rubric conflict (OQ-01): brief slide 7 has six criteria (25/20/20/20/10/5); the guide has five (25/25/20/20/10).
- Model fusion: the term is absent from the materials (whole-word grep count 0 in brief and guide). Expectations come from must-deliver item 2, brief slides 4, 6 and 7, and guide Step 6 (JC-02, JC-03).
- Pro QHD webcam (brief slide 8) is not in the kit recorded in FACTS-LIVE (BR-32).
- Robotics AI Suite named on brief slides 11 and 14 and scored at guide line 136, but not used (RB-02).
- No PaDiM decision recorded (GD-05). PatchCore batch 1 not benchmarked (architecture.md line 76). Power needs root (architecture.md line 109; JC-04).
- No runnable loop entry point (architecture.md line 104). The brief lists GR00T as runnable (slide 11); guide line 107 does not.
- Parked arm found mid-air at 16:00Z (FACTS-LIVE line 34) and rescued; the safe-stop test is still open (GD-10).
- MuJoCo twin: 150 scripted episodes, 73 contradict OD-10; not used for training (FACTS-LIVE line 25).
- Team Modal target blocked on a payment method (FACTS-LIVE line 20).
- Modal track submissions close at the end of Monday 16 November (T09B lines 50, 422). Event 13-15 November (T09B line 40).
- architecture.md status (15:11Z) may be out of date.

## Requests to Opus (same 14 items as alignment.md, line 251)
1. Decide OQ-01 with Hassan: which rubric governs, the brief's six criteria or the guide's five.
2. Get the Challenge Day date and the main submission format from the kickoff sheet (OQ-02).
3. Export the mentor Q&A into the allowed inputs so OQ-03 can close.
4. Replace the PLACEHOLDER model ids (RULES lines 31, 40) before any routing test (BR-02, BR-08).
5. Set the ACT device explicitly (RULES lines 32, 41) and confirm it on a live load (GD-13).
6. Settle the routing rule with Hassan (FACTS line 1 and OD-10 against FACTS line 20).
7. Reconcile the dataset episode counts and the task strings (RB-03, GD-20, BR-16).
8. Confirm whether SmolVLA 6540166d and models.tsv row 5 (d128dc1a) are the same training run (BG-02, BR-16).
9. Decide on the Robotics AI Suite (RB-02).
10. Schedule any closed-loop, grasp or Run model test under the robot motion rule (BR-22, BR-17, JC-05, GD-22, GD-10). E1 has not commanded the arm.
11. Decide the Modal track entry before the end of Monday 16 November (MD-02).
12. Confirm whether the Pro QHD webcam is in the kit (BR-32).
13. Decide the PaDiM row (GD-05) and how power is measured without root (JC-04, RB-04).
14. Serial ports: use the by-id paths (LAPTOP.md rule 8). No action for E1.
