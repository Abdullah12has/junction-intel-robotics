"""plan_survey.py: also write the relative view library (config/survey_library.json)."""
P = "/home/intel/intel-junction-2026/app/tools/plan_survey.py"
s = open(P).read()
R = [
    ('''    spots = []
    for r in spec["spot_search"]["radius_mm"]:''', '''    spots, keys = [], set()
    for r in spec["spot_search"]["radius_mm"]:'''),
    ('''            worst, best, psi = evaluate((x, y), q_h, spec, jaw, COARSE)
            spots.append((worst, x, y, q_h))''', '''            worst, best, psi = evaluate((x, y), q_h, spec, jaw, COARSE)
            spots.append((worst, x, y, q_h))
            keys |= {tuple(v["key"]) for v in best.values()}'''),
    ('''    worst_f, best_f, psi = evaluate((x, y), q_h, spec, jaw, FINE)''', '''    worst_f, best_f, psi = evaluate((x, y), q_h, spec, jaw, FINE)
    keys |= {tuple(v["key"]) for v in best_f.values()}
    lib = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "relative survey views (TCP offset mm from the hover above the brick, wrist roll): the best view of "
                   "each face at every candidate spot of the workspace search",
           "keys": sorted([list(k) for k in keys]), "from_spots": len(spots),
           "criteria": {"min_cos": FINE["min_cos"], "max_jaw": FINE["max_jaw"], "max_occ": 0.02}}'''),
    ('''    if not a.dry:
        os.makedirs(os.path.dirname(sv.PLAN_F), exist_ok=True)''', '''    print("library keys", len(lib["keys"]), "distinct offsets", len({tuple(k[:3]) for k in lib["keys"]}))
    if not a.dry:
        os.makedirs(os.path.dirname(sv.PLAN_F), exist_ok=True)
        lib["inputs"] = plan["inputs"]
        with open(sv.LIB_F + ".tmp", "w") as f:
            json.dump(lib, f, indent=1)
        os.replace(sv.LIB_F + ".tmp", sv.LIB_F)'''),
]
for a, b in R:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(P, "w").write(s)
print("plan_survey.py patched")
