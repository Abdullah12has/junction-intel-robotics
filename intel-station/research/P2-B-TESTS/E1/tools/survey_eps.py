"""Read-only survey of logs/sort/any_1740 episode.json files (summary lines only). Scratch tool of builder E1."""
import glob
import json
import os

ROOT = "/home/intel/intel-junction-2026/logs/sort/any_1740"
for f in sorted(glob.glob(os.path.join(ROOT, "ep_*", "episode.json"))):
    d = json.load(open(f))
    hb = d.get("hover_blob")
    pc = d.get("pre_close")
    ep_dir = os.path.dirname(f)
    cdir = os.path.join(ep_dir, "crops")
    crops = sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []
    hb_s = (hb.get("centre"), hb.get("short"), hb.get("colour")) if isinstance(hb, dict) else None
    pc_s = None
    if isinstance(pc, dict):
        pc_s = (pc.get("verdict"), pc.get("colour"), pc.get("hue"), pc.get("reasons"))
    elif pc is not None:
        pc_s = pc
    print(os.path.basename(ep_dir), "ep", d.get("episode"), "success", d.get("success"),
          "target_colour", d.get("target_colour"), "failure", d.get("failure"),
          "| hover_blob", hb_s, "| pre_close", pc_s, "| crops", crops)
