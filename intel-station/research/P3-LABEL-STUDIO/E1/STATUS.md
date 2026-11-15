DONE 2026-11-14T20:26Z E1 P3-LABEL-STUDIO, stage studio-labels-v1 (run 2): 41 Studio entries from 17 physical episodes, all verified in frames; verified 41, mismatched 12; usable training 23, evaluation 12, calibration 5, nothing 1 (entries).

Deliverables (the same files are in R/work/P3-LABEL-STUDIO/E1/ on the laptop and in this folder): datasets.json, labels_episodes.jsonl, mismatches.tsv, trust_report.md, metrics.json, laptop-log.md, STATUS.md. Evidence (frames, montages, out/ JSON, tools/) stays on the laptop.

Coverage checklist: trust_report.md section 10 (9 Studio datasets, 41 entries, 24 of 24 curated episodes, V1 used as a hint only). Two items are left unused on purpose: zip archives inside the curated folders and the extra V1 files.

Self-check: trust_report.md section 11, 14 items, re-verified in this run.

Copy check (sha256 equal on the laptop and in this folder):
- labels_episodes.jsonl 0dcb15be26ffa4ff5846d38e4e3f7685ee440510c41332b536aa6ce2b82bc1d9
- datasets.json 5e1bcfe2d1a4c29d75b9e16fcc06b3b52880fd1b1f270bfff241f13ec98e5ce2
- mismatches.tsv 745fc1c3e1fc0a4c83a4173bd2a7843c1934af4ce80f96d130d48ede238cdc78
- metrics.json 4641c1f311bf12806f2d449b2db4c8fd9122f80e0984ee0dc1f24f6a76a9193e
- trust_report.md 4b47f82ad4f1884e4756556ca00f52ff5afee8a68d4f60287e308cf1206f049e
- laptop-log.md 32ee75c64efe5d30a40a97bb234c49574f3084bfe488938a5a0ecc861a410f35

Open issues (trust_report.md section 8): orientation is not recorded for most bricks; info.json gives no joint units (the observed ranges fit the normalized SO-101 units, not checked on hardware); the visual labels of the 17 representatives come from earlier frame viewing and were not all re-checked in this run; re-encoded copies a4ba6ede:0 to :3 are not byte-identical; P1 has a text conflict between 6d6bdce3:7 and its copies.

Blocker: none. The remote-devices MCP connection failed at session start; this task does not use it.
No further refresh is needed after DONE.
