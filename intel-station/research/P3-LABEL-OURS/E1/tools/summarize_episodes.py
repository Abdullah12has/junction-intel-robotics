import json, glob, os, sys

R = sys.argv[1]
paths = sorted(glob.glob(os.path.join(R, 'logs/sort/*/ep_*/episode.json')))
keys = {}
rows = []
for p in paths:
    with open(p) as f:
        d = json.load(f)
    for k in d:
        keys[k] = keys.get(k, 0) + 1
    tsb = d.get('target_scan_blob') or {}
    t0 = d.get('t_start')
    t1 = d.get('t_end')
    dur = round(t1 - t0, 1) if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) else None
    rows.append({
        'ep': os.path.relpath(os.path.dirname(p), R),
        'run': d.get('run'),
        'n': d.get('episode'),
        'mode': d.get('mode'),
        'shape': d.get('shape'),
        'shape_why': d.get('shape_why'),
        'success': d.get('success'),
        'failure': d.get('failure'),
        'grip_closed': d.get('grip_closed'),
        'n_candidates': d.get('n_candidates'),
        'hue': tsb.get('hue'),
        'sat': tsb.get('sat'),
        'val': tsb.get('val'),
        'area': tsb.get('area'),
        'aspect': tsb.get('aspect'),
        'dur_s': dur,
        'tip_mm': d.get('hover_tip_size_mm'),
        'plate': d.get('plate'),
        'tape': d.get('tape'),
        'verdict': d.get('verdict'),
    })
print('KEYS ' + json.dumps(keys, sort_keys=True))
for r in rows:
    print(json.dumps(r, sort_keys=True))
