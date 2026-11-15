import json, os, sys
from collections import Counter

R = sys.argv[1]
OUT = sys.argv[2]
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def rd(name):
    with open(os.path.join(D, name)) as f:
        return [l.rstrip('\n') for l in f if l.strip() and not l.startswith('#')]


meta = {}
for l in rd('episode_meta.tsv'):
    eid, tcol, tdesc, conf = l.split('\t')
    meta[eid] = (tcol, tdesc, conf)

PHASE_SHOWS = {
    'pre': 'before grasp: plates, robot and mat bricks; target {d} among other bricks',
    'stage1': 'approach: gripper above target {d}',
    'grasp_open': 'jaws open around target {d}',
    'closed': 'jaws closed on target {d}',
    'lifted': 'target {d} lifted in the jaws',
    'released': 'target {d} released over the plate',
    'station': 'target {d} put down at the station on the mat',
}
CAM_WHY = {
    'top': 'full table view, not a training crop',
    'close': 'full close frame; other bricks, jaws or cable in view',
    'wrist': 'wrist frame; jaws and other bricks in view, not an isolated brick',
}
COMPOSITE = {
    'review.jpg': 'loop review composite of several phases with status text',
    'chk.jpg': 'pre-check composite of three camera views',
    'held_sheet.jpg': 'held-brick composite of three frames',
}
HAND_NOTE = {
    ('any_1731/ep_0001', 'closed_top.jpg'): 'human hand at left edge of top view',
    ('any_1731/ep_0002', 'pre_top.jpg'): 'human hand at left edge of top view',
    ('any_1731/ep_0002', 'grasp_open_top.jpg'): 'human hand at left edge of top view',
    ('any_1731/ep_0002', 'closed_top.jpg'): 'human hand at left edge of top view',
    ('any_1740/ep_0001', 'grasp_open_top.jpg'): 'human hand at left edge of top view',
    ('any_1740/ep_0001', 'lifted_top.jpg'): 'human hand at left edge of top view',
    ('any_1740/ep_0004', 'closed_top.jpg'): 'human hand at left edge pointing at blue plate',
    ('any_1740/ep_0011', 'pre_top.jpg'): 'human hand at top-left edge',
}

out = []


def add(path, eid, kind, brick, usable, why, shows, conf, notes=''):
    out.append({'image_path': path, 'episode_id': eid, 'kind': kind, 'shows': shows,
                'brick': brick, 'usable_for_training': usable, 'reason': why,
                'label_source': 'visual', 'confidence': conf, 'notes': notes})


# 1 sort episodes: phase snapshots and composites (kind = phase name)
for l in rd('index_sort.tsv'):
    eid, files = l.split('\t')
    tcol, tdesc, conf = meta[eid]
    for fn in files.split():
        path = os.path.join(R, 'logs/sort', eid, fn)
        if fn in COMPOSITE:
            add(path, eid, 'composite', tcol, 'no', 'composite of several frames, not a training sample',
                COMPOSITE[fn] + '; target ' + tdesc, conf, HAND_NOTE.get((eid, fn), ''))
            continue
        phase, cam = fn[:-4].rsplit('_', 1)
        shows = PHASE_SHOWS[phase].format(d=tdesc) + '; ' + cam + ' view'
        add(path, eid, phase, tcol, 'no', CAM_WHY[cam], shows, conf, HAND_NOTE.get((eid, fn), ''))

# 2 crops of sort episodes (usable decided per crop in crops.tsv)
for l in rd('crops.tsv'):
    eid, cf, usable, why, shows = l.split('\t')
    tcol, tdesc, conf = meta[eid]
    add(os.path.join(R, 'logs/sort', eid, 'crops', cf), eid, 'crop', tcol, usable, why, shows, conf)

# 3 e-stop images (one row per file in estop_verified.tsv, verified at full resolution)
for l in rd('estop_verified.tsv'):
    f, view, label, holding, verified, conf, note = l.split('\t')
    path = os.path.join(R, 'logs/monitor', f)
    shows = view + ' e-stop frame: ' + label + ('; brick held between jaws' if holding == '1' else '; no brick held')
    if holding == '1' and view == 'wrist':
        usable, why = 'no', 'wrist view gated while holding; jaws and brick in frame'
    elif label == 'hand_partial':
        usable, why = 'no', 'partial hand at frame edge; ambiguous for training'
    elif verified != 'yes':
        usable, why = 'no', 'label not verified at full resolution'
    else:
        usable, why = 'yes', 'label verified at full resolution'
    add(path, 'none', 'estop', 'none', usable, why, shows, conf, note)


# 4 grasp_auto, depth, jprobe and servo images (one row per file in scope_index.tsv)
def scope_rule(fn):
    if fn.endswith('_height_viz.png') or fn.endswith('_ir.png'):
        return 'depth', 'depth or IR visualisation, not a training crop'
    if fn in ('quad.jpg', 'rel.jpg') or fn.startswith(('now', 'park', 'peek', 'hover', 'masks', 'jawcmp')):
        return 'composite', 'composite of several camera frames, not a training sample'
    if fn.startswith('live_'):
        return 'other', 'live camera frame from a servo or probe session, not a brick crop'
    if fn.endswith('_color.jpg'):
        return 'other', 'colour close frame with jaws and other bricks in view, not a brick crop'
    if fn.startswith('c0_'):
        return 'other', 'calibration frame, not a brick crop'
    return 'other', 'servo debug or single snapshot frame, not a brick crop'


for l in rd('scope_index.tsv'):
    group, rel, files, shows = l.split('\t')
    for fn in files.split():
        kind, why = scope_rule(fn)
        add(os.path.join(R, rel, fn), 'none', kind, 'none', 'no', why, shows + ' [' + fn + ']', 'med',
            'scope group ' + group)

with open(OUT, 'w') as f:
    for r in out:
        f.write(json.dumps(r, sort_keys=True) + '\n')
print('WROTE', len(out), 'records to', OUT)
print('KINDS', dict(Counter(r['kind'] for r in out)))
print('USABLE', dict(Counter(r['usable_for_training'] for r in out)))
