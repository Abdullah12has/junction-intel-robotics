#!/usr/bin/env python3
# V3: correct the snapshot label (sync 13:23Z-13:31Z) and blank "None" cells in episodes.csv.
# usage: python3 -I v3_fix.py REPORT_DIR
import sys
import json
from pathlib import Path

R = Path(sys.argv[1])
p = R / "INVENTORY.md"
t = p.read_text()
t = t.replace("Snapshot: sync log last entry 2026-11-14T13:23:07Z; 11 episodes, 6693 frames on disk.",
              "Snapshot: sync log 2026-11-14T13:23Z to 13:31Z (265M, unchanged since 13:05Z); 11 episodes, 6693 frames on disk.")
t = t.replace("this is the 13:23Z snapshot", "this is the 13:23-13:31Z snapshot")
p.write_text(t)
g = json.loads((R / "gaps.json").read_text())
g["snapshot"] = "sync log 2026-11-14T13:23Z to 13:31Z (265M, unchanged since 13:05Z); info.json total_episodes 11, total_frames 6693"
(R / "gaps.json").write_text(json.dumps(g, indent=1))
c = (R / "episodes.csv").read_text().replace(",None,", ",,")
(R / "episodes.csv").write_text(c)
print("V3_FIX_OK", t.count("13:31Z"), "None cells left:", c.count("None"))
