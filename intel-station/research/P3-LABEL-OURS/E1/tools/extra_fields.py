import json, glob, os, sys

R = sys.argv[1]
want = ['colour', 'target_colour', 'label', 'label_source', 'label_det2', 'plate',
        'decision_conf', 'decision_reasons', 'opus_note', 'task', 'views_used', 'views',
        'held_size_ratio', 'open_short', 'grip_lifted', 'grasp_use_mm', 'timings', 'pre_close']
paths = sorted(glob.glob(os.path.join(R, 'logs/sort/*/ep_*/episode.json')))
for p in paths:
    with open(p) as f:
        d = json.load(f)
    ep = os.path.relpath(os.path.dirname(p), R)
    out = {}
    for k in want:
        if k in d:
            s = json.dumps(d[k], sort_keys=True)
            out[k] = s if len(s) <= 260 else s[:260] + '...'
    print('== ' + ep)
    for k, v in out.items():
        print('  ' + k + ': ' + v)
