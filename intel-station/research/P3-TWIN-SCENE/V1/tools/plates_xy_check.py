#!/usr/bin/env python3
"""V1 check of the three plate separations: PLATES_XY is read from sort_any.py by ast (no import of the robot code);
the ridge and outline centres are read from the builder's JSON files (read-only). Writes outputs/plates_xy_check.json."""
import ast
import json
import math

SORT = "/home/intel/intel-junction-2026/app/tools/sort_any.py"
V4 = "/home/intel/intel-junction-2026/work/P3-TWIN-SCENE/E1/scene_measurements_raw_v4.json"
RIDGE = "/home/intel/intel-junction-2026/work/P3-TWIN-SCENE/E1/plates_ridge_v1.json"
OUT = "/home/intel/intel-junction-2026/work/P3-TWIN-SCENE/V1/outputs/plates_xy_check.json"

src = open(SORT, encoding="utf-8").read()
tree = ast.parse(src)
plates, line = None, None
for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PLATES_XY" for t in node.targets):
        plates = ast.literal_eval(node.value)
        line = node.lineno
b, g = plates["blue"], plates["green"]
sep_cfg = math.hypot(g[0] - b[0], g[1] - b[1])
v4 = json.load(open(V4, encoding="utf-8"))
pl = v4["depth"]["top"]["plates"]
bl, gr = pl[0], pl[1]
sep_outline = math.hypot(gr["cx_mm"] - bl["cx_mm"], gr["cy_mm"] - bl["cy_mm"])
rj = json.load(open(RIDGE, encoding="utf-8"))
rb = rj["plates"]["ring_left"]["centre_mat_mm"]
rg = rj["plates"]["ring_right"]["centre_mat_mm"]
sep_ridge = math.hypot(rg[0] - rb[0], rg[1] - rb[1])
res = {"sort_any_py_line": line, "PLATES_XY": plates, "sep_PLATES_XY_mm": round(sep_cfg, 2),
       "sep_ridge_v1_mm": round(sep_ridge, 2), "sep_outline_v4_mm": round(sep_outline, 2),
       "outline_blue_seed": bl["seed"], "outline_blue_centre_mm": [bl["cx_mm"], bl["cy_mm"]],
       "outline_blue_R_outline_fit_mm": bl["R_outline_fit_mm"],
       "outline_pair_uses_rejected_blue_centre": True,
       "PLATES_XY_radial_diff_mm": round(g[1] - b[1], 2)}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
