#!/usr/bin/env python3
"""P2-D-DATA W3: read-only inventory, per-episode QA, sub-episode segmentation (S1 / S2 / S3),
loop calibration, gap list and contact sheets for Studio LeRobot v3.0 datasets.
Run: python -I inventory.py --root DIR --out DIR [--sync-log F] [--manual F] [--studio JSON]
[--watch --every SEC --max-cycles N]. Writes only under --out; never writes into --root."""
import argparse, glob, hashlib, json, os, subprocess, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

CLOSED_MAX = 6.0        # gripper units: a closure settling below this has no brick-width plateau
CLOSE_V = -8.0          # gripper units/s: closure onset speed, sustained for RUN_MIN_S
OPEN_V = 8.0            # gripper units/s: opening speed
RUN_MIN_S = 0.15        # sustained duration for a closure or opening onset
OPEN_RISE = 5.0         # gripper units: minimum rise for a release
STAT_THR = 2.0          # arm speed units/s: max |velocity| over the 5 arm joints (5-frame centred mean)
DWELL_S = 1.0           # stationary dwell after lift
CUT_AFTER_LIFT_S = 1.0  # S1 part ends this long after the lift begins
END_HOLD_MAX = 25.0     # whole-episode S1 check: end within this distance of own hold spot
RETURN_DIST = 20.0      # arm within this distance (5 arm joints) of median home = back at rest
RECENT_S = 150          # files modified less than this many seconds ago may still be syncing
ARM = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos']
GRIP = 'gripper.pos'
CAMS = ['realsense 243722064316', 'realsense 243622061421']
NEEDS = {'S1': (40, 30), 'S2': (30, 25), 'S3': (30, 25)}
THUMB = (160, 120)
KEYS = ['start', 'grasp', 'hold', 'release', 'end']
PAGE = 6


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def r3(x):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 3)


def runs(mask, min_len):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def text_class(task):
    t = task.lower()
    if 'blue' in t:
        return 'S2'
    if 'green' in t:
        return 'S3'
    if any(k in t for k in ('inspection', 'hold', 'pick', 'lift')):
        return 'S1'
    return 'OTHER'


def grab(video, t_abs, png):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-ss', '%.4f' % t_abs, '-i', video, '-frames:v', '1',
                        '-vf', 'scale=%d:%d' % THUMB, '-y', png], capture_output=True, text=True)
    ok = p.returncode == 0 and os.path.exists(png) and os.path.getsize(png) > 0
    return ok, (p.stderr or '').strip()[:160]


def cached(video, t_abs, png):
    if os.path.exists(png) and os.path.getsize(png) > 0:
        return True, ''
    return grab(video, t_abs, png)


def packets(video):
    p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                        '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', video],
                       capture_output=True, text=True)
    try:
        return int(p.stdout.strip().splitlines()[0])
    except Exception:
        return None


def seg_status(sg, r, mv, struct_bad):
    bad, warn = list(struct_bad), []
    if sg['skill'] == 'S1':
        bg = str(mv.get('brick_at_grasp', 'unverified')).lower()
        if bg == 'no':
            bad.append('no brick between the jaws at grasp (visual)')
        elif bg != 'yes':
            warn.append('brick at grasp not verified visually')
    else:
        if sg.get('plate_label') == 'NONE':
            bad.append('release not over a plate (visual): not an S2/S3 part')
        elif sg.get('plate_label') not in ('BLUE', 'GREEN'):
            warn.append('release plate not labelled (visual check needed)')
        if sg.get('return_idx') is None:
            warn.append('no return to rest before episode end')
        if r['grasp_idx'] is None:
            bs = str(mv.get('brick_at_start', 'unverified')).lower()
            if bs == 'no':
                bad.append('closed-start part: no brick held at start (visual)')
            elif bs != 'yes':
                warn.append('closed-start part: brick held at start not verified visually')
    st = 'BAD' if bad else ('WARN' if warn else 'OK')
    return st, bad + warn


def analyse(ddir, tmp, studio, manual):
    did = os.path.basename(os.path.normpath(ddir))
    info = json.load(open(os.path.join(ddir, 'meta', 'info.json')))
    fps = float(info['fps'])
    feats = info['features']
    names = feats['observation.state']['names']
    ai = [names.index(j) for j in ARM]
    gi = names.index(GRIP)
    tr = pd.read_parquet(os.path.join(ddir, 'meta', 'tasks.parquet')).reset_index()
    tcol = [c for c in tr.columns if c != 'task_index'][0]
    task_of = {int(t): str(x) for x, t in zip(tr[tcol], tr['task_index'])}
    ep_meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ddir, 'meta', 'episodes', 'chunk-*', 'file-*.parquet')))],
                        ignore_index=True).sort_values('episode_index').reset_index(drop=True)
    dat = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ddir, 'data', 'chunk-*', 'file-*.parquet')))],
                    ignore_index=True).sort_values(['episode_index', 'frame_index']).reset_index(drop=True)
    S = np.stack(dat['observation.state'].values).astype(float)
    A = np.stack(dat['action'].values).astype(float)
    EPI = dat['episode_index'].values
    TI = dat['task_index'].values
    TS = dat['timestamp'].values.astype(float)
    FI = dat['frame_index'].values
    cams_ok = all(('observation.images.' + c) in feats for c in CAMS)
    compat = {'robot_type_SO101_Follower': info.get('robot_type') == 'SO101_Follower',
              'camera_keys_present': cams_ok, 'fps_30': abs(fps - 30.0) < 1e-6,
              'codebase_v3': str(info.get('codebase_version', '')).startswith('v3'),
              'studio_environment_name_verified': False}
    compat['compatible'] = all(compat[k] for k in ['robot_type_SO101_Follower', 'camera_keys_present', 'fps_30', 'codebase_v3'])
    newest = 0.0
    for root_dir, _, files in os.walk(ddir):
        for f in files:
            newest = max(newest, os.path.getmtime(os.path.join(root_dir, f)))
    recent = (datetime.now(timezone.utc).timestamp() - newest) < RECENT_S
    D = {'id': did, 'studio': studio, 'compat': compat, 'recently_modified_may_be_mid_sync': bool(recent),
         'info': {k: info.get(k) for k in ['codebase_version', 'fps', 'robot_type', 'total_episodes', 'total_frames',
                                            'total_tasks', 'chunks_size', 'splits', 'data_path', 'video_path']},
         'state_names': names, 'action_names_equal_state': feats['action']['names'] == names,
         'video_feature_keys': [k for k in feats if k.startswith('observation.images.')],
         'tasks': [{'task_index': k, 'task': v} for k, v in sorted(task_of.items())],
         'data_files': [os.path.relpath(f, ddir) for f in sorted(glob.glob(os.path.join(ddir, 'data', 'chunk-*', 'file-*.parquet')))],
         'episode_meta_rows': int(len(ep_meta)), 'episodes_in_data': int(dat['episode_index'].nunique()),
         'data_rows': int(len(dat))}
    eps, series = [], []
    for _, m in ep_meta.iterrows():
        e = int(m['episode_index'])
        idx = np.where(EPI == e)[0]
        n = len(idx)
        s, a = S[idx], A[idx]
        r = {'dataset': did, 'episode': e, 'global_from': int(m['dataset_from_index']), 'meta_length': int(m['length']),
             'data_rows': n, 'duration_s': r3(n / fps)}
        r['task'] = task_of.get(int(TI[idx[0]]), '?')
        r['cls_text'] = text_class(r['task'])
        r['frame_index_contiguous'] = bool(np.array_equal(FI[idx], np.arange(n)))
        r['dt_max_over_nominal'] = r3(np.diff(TS[idx]).max() * fps) if n > 1 else None
        P = pd.DataFrame(s[:, ai]).rolling(5, center=True, min_periods=1).mean().values
        speed = np.abs(np.gradient(P, axis=0) * fps).max(axis=1)
        mov = speed > STAT_THR
        g = pd.Series(s[:, gi]).rolling(3, center=True, min_periods=1).mean().values
        vg = np.gradient(g) * fps
        mn = int(round(RUN_MIN_S * fps))
        half, one = int(0.5 * fps), int(fps)
        cl = runs(vg < CLOSE_V, mn)
        op = runs(vg > OPEN_V, mn)

        def settled(end, g=g, n=n, half=half):
            seg = g[end:min(n, end + half)]
            return float(np.median(seg)) if len(seg) else float('nan')

        full = [c for c in cl if settled(c[1]) < CLOSED_MAX]
        r['closures_total'] = len(cl)
        r['closures_full'] = len(full)
        r['closures_partial'] = len(cl) - len(full)
        r['opening_runs'] = len(op)
        r['closed_start'] = bool(g[0] <= CLOSED_MAX)
        mo = np.where(mov)[0]
        r['move_onset_s'] = r3(mo[0] / fps) if len(mo) else None
        r['idle_start_s'] = r3(mo[0] / fps) if len(mo) else r3(n / fps)
        r['idle_end_s'] = r3((n - 1 - mo[-1]) / fps) if len(mo) else None
        r['start_arm'] = [r3(x) for x in s[0, ai]]
        r['start_grip'] = r3(s[0, gi])
        r['end_arm'] = [r3(x) for x in s[-1, ai]]
        r['end_grip'] = r3(s[-1, gi])
        for k in ['grasp_idx', 'grasp_end_idx', 'lift_idx', 'hold_idx', 'hold_end_idx', 'release_idx', 'return_idx']:
            r[k] = None
        for k in ['grasp_t_s', 'grasp_open_level', 'closed_level', 'closed_gap_state_minus_action', 'lift_t_s', 'hold_t_s',
                  'hold_end_s', 'hold_arm', 's1_segment_s', 'release_t_s', 'release_level', 'release_arm',
                  'end_dist_own_hold', 'usable_clip_s']:
            r[k] = None
        if full:
            gs, ge = full[-1]
            r['grasp_idx'] = int(gs)
            r['grasp_end_idx'] = int(ge)
            r['grasp_t_s'] = r3(gs / fps)
            r['grasp_open_level'] = r3(np.median(g[max(0, gs - half):gs])) if gs > 0 else None
            if ge < n:
                r['closed_level'] = r3(np.median(g[ge:min(n, ge + one)]))
                r['closed_gap_state_minus_action'] = r3(np.median(s[ge:min(n, ge + one), gi] - a[ge:min(n, ge + one), gi]))
            after = np.where(mov[ge:])[0]
            if len(after):
                lift = ge + int(after[0])
                r['lift_idx'] = int(lift)
                r['lift_t_s'] = r3(lift / fps)
                hr = runs(~mov[lift:], int(round(DWELL_S * fps)))
                if hr:
                    h0, h1 = lift + hr[0][0], lift + hr[0][1]
                    r['hold_idx'] = int(h0)
                    r['hold_end_idx'] = int(h1)
                    r['hold_t_s'] = r3(h0 / fps)
                    r['hold_end_s'] = r3(h1 / fps)
                    r['hold_arm'] = [r3(x) for x in np.median(s[h0:h1, ai], axis=0)]
                    if len(mo):
                        r['s1_segment_s'] = r3((h0 - mo[0]) / fps)
        ref = r['hold_idx'] if r['hold_idx'] is not None else (r['grasp_end_idx'] if r['grasp_end_idx'] is not None else 0)
        rel = [c for c in op if c[0] > ref and g[min(c[1], n - 1)] - g[c[0]] >= OPEN_RISE]
        if rel:
            r['release_idx'] = int(rel[0][0])
            r['release_t_s'] = r3(rel[0][0] / fps)
            r['release_level'] = r3(g[min(rel[0][1], n - 1)])
            r['release_arm'] = [r3(x) for x in s[rel[0][0], ai]]
        r['video'] = {}
        for c in CAMS:
            vk = 'videos/observation.images.' + c
            r['video'][c] = {'chunk': int(m[vk + '/chunk_index']), 'file': int(m[vk + '/file_index']),
                             'from_ts': float(m[vk + '/from_timestamp']), 'to_ts': float(m[vk + '/to_timestamp'])}
        eps.append(r)
        series.append(s[:, ai])
    home_med = np.median(np.array([r['start_arm'] for r in eps], dtype=float), axis=0)
    holds = np.array([r['hold_arm'] for r in eps if r['hold_arm'] is not None], dtype=float)
    hold_med = np.median(holds, axis=0) if len(holds) else None
    for k, r in enumerate(eps):
        r['start_dist_home'] = r3(np.linalg.norm(np.array(r['start_arm']) - home_med))
        r['end_dist_hold'] = r3(np.linalg.norm(np.array(r['end_arm']) - hold_med)) if hold_med is not None else None
        r['hold_dist_median'] = (r3(np.linalg.norm(np.array(r['hold_arm']) - hold_med))
                                 if (r['hold_arm'] is not None and hold_med is not None) else None)
        r['end_dist_own_hold'] = (r3(np.linalg.norm(np.array(r['end_arm']) - np.array(r['hold_arm'])))
                                  if r['hold_arm'] is not None else None)
        if r['release_idx'] is not None:
            dist = np.linalg.norm(series[k] - home_med, axis=1)
            cand = np.where(dist[r['release_idx']:] < RETURN_DIST)[0]
            r['return_idx'] = int(r['release_idx'] + cand[0]) if len(cand) else None
        r['whole_episode_S1'] = bool(r['hold_idx'] is not None and r['release_idx'] is None
                                     and r['end_dist_own_hold'] is not None and r['end_dist_own_hold'] <= END_HOLD_MAX)
        n = r['data_rows']
        segs, start2 = [], 0
        if r['grasp_idx'] is not None:
            base = r['lift_idx'] if r['lift_idx'] is not None else r['grasp_end_idx']
            cut = min(n - 1, base + int(round(CUT_AFTER_LIFT_S * fps)))
            segs.append({'skill': 'S1', 'start_idx': 0, 'end_idx': int(cut), 'rule': 'start to 1.0 s after lift start'})
            start2 = int(cut)
        if r['release_idx'] is not None:
            end2 = r['return_idx'] if r['return_idx'] is not None else n - 1
            if end2 > start2:
                segs.append({'skill': 'S2/S3', 'start_idx': start2, 'end_idx': int(end2), 'release_idx': r['release_idx'],
                             'return_idx': r['return_idx'], 'rule': 'S1 cut to release and return to rest'})
        for sg in segs:
            sg['start_t_s'] = r3(sg['start_idx'] / fps)
            sg['end_t_s'] = r3(sg['end_idx'] / fps)
            sg['global_start'] = r['global_from'] + sg['start_idx']
            sg['global_end'] = r['global_from'] + sg['end_idx']
            sg['duration_s'] = r3((sg['end_idx'] - sg['start_idx']) / fps)
        r['segments'] = segs
        r['decode'] = {}
        tg = r['grasp_idx'] / fps if r['grasp_idx'] is not None else None
        th = r['hold_idx'] / fps if r['hold_idx'] is not None else (tg + 2.0 if tg is not None else None)
        tr_ = r['release_idx'] / fps if r['release_idx'] is not None else None
        times = {'start': 0.0, 'grasp': tg, 'hold': th, 'release': tr_, 'end': (n - 1) / fps}
        for c in CAMS:
            v = r['video'][c]
            path = os.path.join(ddir, 'videos', 'observation.images.' + c, 'chunk-%03d' % v['chunk'], 'file-%03d.mp4' % v['file'])
            vsig = '%d_%d' % (os.path.getsize(path), int(os.path.getmtime(path))) if os.path.exists(path) else 'missing'
            r['decode'][c] = {}
            for kk in KEYS:
                t = times[kk]
                if t is None:
                    r['decode'][c][kk] = {'ok': None, 'png': None, 't_rel_s': None, 'err': 'no event'}
                    continue
                png = os.path.join(tmp, '%s_e%d_%s_%s_%s.png' % (did[:8], r['episode'], c.replace(' ', '_'), kk, vsig))
                ok, err = cached(path, v['from_ts'] + t, png)
                r['decode'][c][kk] = {'ok': ok, 'png': png if ok else None, 't_rel_s': r3(t), 'err': err}
    fc = {}
    for c in CAMS:
        for f in sorted({r['video'][c]['file'] for r in eps}):
            mem = [r for r in eps if r['video'][c]['file'] == f]
            chunk = mem[0]['video'][c]['chunk']
            path = os.path.join(ddir, 'videos', 'observation.images.' + c, 'chunk-%03d' % chunk, 'file-%03d.mp4' % f)
            exp = sum(r['meta_length'] for r in mem)
            act = packets(path)
            fc['%s|file-%03d' % (c, f)] = {'expected_frames': exp, 'packets': act, 'episodes': [r['episode'] for r in mem], 'ok': act == exp}
    D['files'] = fc
    for r in eps:
        mv = (manual or {}).get(str(r['episode'])) or {}
        bad_s, warn = [], []
        if r['meta_length'] != r['data_rows']:
            bad_s.append('meta length %d != data rows %d' % (r['meta_length'], r['data_rows']))
        if not r['frame_index_contiguous']:
            bad_s.append('frame_index not contiguous')
        for c in CAMS:
            for kk in KEYS:
                if r['decode'][c][kk]['ok'] is False:
                    bad_s.append('decode failed %s %s' % (c, kk))
            fk = '%s|file-%03d' % (c, r['video'][c]['file'])
            if not fc[fk]['ok']:
                bad_s.append('frame count %s: packets %s != expected %s' % (fk, fc[fk]['packets'], fc[fk]['expected_frames']))
        if r['dt_max_over_nominal'] is not None and r['dt_max_over_nominal'] > 1.5:
            warn.append('timestamp gap x%s' % r['dt_max_over_nominal'])
        if not r['segments']:
            bad_s.append('no usable part: no gripper closure with lift and no release (fully closed or idle episode)')
        if r['grasp_idx'] is None and r['release_idx'] is not None:
            warn.append('starts with gripper closed: no S1 part; carry-release-return part only')
        if r['grasp_idx'] is not None and r['hold_idx'] is None:
            warn.append('no stationary hold after lift')
        if r['idle_start_s'] is not None and r['idle_start_s'] < DWELL_S:
            warn.append('no idle start (%s s)' % r['idle_start_s'])
        if r['closures_full'] > 1 or r['closures_partial'] > 0:
            warn.append('%d full + %d partial closures (retries)' % (r['closures_full'], r['closures_partial']))
        if r['closed_level'] is not None and r['closed_level'] < CLOSED_MAX:
            warn.append('gripper settles at %s (<%s): brick presence must be checked on the contact sheet' % (r['closed_level'], CLOSED_MAX))
        if r['start_dist_home'] is not None and r['start_dist_home'] > 15:
            warn.append('start pose %s from median home' % r['start_dist_home'])
        plate = str(mv.get('release_plate', '') or '').upper()
        for sg in r['segments']:
            if sg['skill'] == 'S2/S3':
                sg['skill'] = {'BLUE': 'S2', 'GREEN': 'S3'}.get(plate, 'S2/S3')
                sg['plate_label'] = plate or None
                sg['how_decided'] = mv.get('how_decided')
            sg['status'], sg['why'] = seg_status(sg, r, mv, bad_s)
        r['cls_final'] = '+'.join(sg['skill'] for sg in r['segments']) or 'NONE'
        r['class_note'] = 'S1 and S2/S3 parts are per segment; whole-episode S1 = %s' % r['whole_episode_S1']
        if any(sg['status'] == 'BAD' for sg in r['segments']):
            bad_s.append('segment BAD')
        if bad_s:
            v = 'BAD'
        elif warn or any(sg['status'] == 'WARN' for sg in r['segments']):
            v = 'WARN'
        else:
            v = 'OK'
        reasons = bad_s + warn
        if mv.get('verdict'):
            v = mv['verdict']
        if mv.get('note'):
            reasons = reasons + ['visual: ' + mv['note']]
        r['visual'] = mv or None
        r['verdict'] = v
        r['reasons'] = reasons
    return D, eps


def calibrate(allc, gpool, s1p, names):
    if not allc:
        return {'generated_utc': utc(), 'note': 'no compatible episodes', 'joint_names_as_stored': names}

    def st(vals):
        v = [float(x) for x in vals if x is not None]
        if not v:
            return None
        return {'median': r3(np.median(v)), 'min': r3(min(v)), 'max': r3(max(v)), 'n': len(v)}

    def arr_stats(rows):
        if not rows:
            return None
        A_ = np.array(rows, dtype=float)
        return {'median': [r3(x) for x in np.median(A_, axis=0)], 'std': [r3(x) for x in np.std(A_, axis=0)],
                'range_max_minus_min': [r3(x) for x in (A_.max(axis=0) - A_.min(axis=0))], 'n': len(rows)}

    seg = [r['s1_segment_s'] for r in s1p if r['s1_segment_s'] is not None]
    dur = [r['duration_s'] for r in allc]
    hs = arr_stats([r['hold_arm'] for r in s1p if r['hold_arm'] is not None])
    cal = {'generated_utc': utc(), 'fps': 30, 'joint_names_as_stored': names, 'arm_joints': ARM, 'gripper_joint': GRIP,
           'pools': {'gripper_stats': 'all %d episodes with a full closure' % len(gpool),
                     'hold_and_S1_duration': 'episodes with a grasp and a hold (%d); brick presence not yet verified for all' % len(s1p),
                     'home_pose': 'all %d compatible episodes' % len(allc)},
           'gripper': {'units': 'gripper.pos as stored',
                       'closing_direction': 'decreasing (closing = negative velocity)',
                       'open_level': st([r['grasp_open_level'] for r in gpool]),
                       'closed_level': st([r['closed_level'] for r in gpool]),
                       'closed_minus_action': st([r['closed_gap_state_minus_action'] for r in gpool]),
                       'closure_onset_rule': {'speed_threshold_units_per_s': CLOSE_V, 'min_duration_s': RUN_MIN_S,
                                              'settled_below': CLOSED_MAX, 'settle_window_s': 0.5,
                                              'onset': 'first sample of the last sustained closing run whose settled level is below settled_below'},
                       'validation': {'episodes_with_full_closure': len(gpool),
                                      'closures_full_partial_per_episode': {'%s:%d' % (r['dataset'][:8], r['episode']): [r['closures_full'], r['closures_partial']] for r in allc},
                                      'grasp_onset_s_per_episode': {'%s:%d' % (r['dataset'][:8], r['episode']): r['grasp_t_s'] for r in allc}}},
           'stationary': {'joint_speed_threshold_units_per_s': STAT_THR,
                          'speed_definition': 'max abs velocity over the 5 arm joints; 5-frame centred mean; np.gradient x fps',
                          'dwell_s': DWELL_S,
                          'validation': {'episodes': len(allc),
                                         'idle_start_ge_dwell': sum(1 for r in allc if r['idle_start_s'] is not None and r['idle_start_s'] >= DWELL_S),
                                         'hold_found': sum(1 for r in allc if r['hold_arm'] is not None)}},
           'home_pose': {'joints': ARM, 'stats': arr_stats([r['start_arm'] for r in allc]),
                         'gripper_start': st([r['start_grip'] for r in allc])},
           'hold_pose': {'joints': ARM, 'S1_hold_spots': hs,
                         'consistent_within_15_units': (max(hs['range_max_minus_min']) <= 15) if hs else None,
                         'whole_episode_S1_count': sum(1 for r in allc if r['whole_episode_S1'])},
           'skills': {'S1': {'n': len(seg),
                             'segment_definition': 'movement onset to start of first stationary dwell (>= dwell_s) after lift that follows the final full closure',
                             'duration_s_p50': r3(np.percentile(seg, 50)) if seg else None,
                             'duration_s_p95': r3(np.percentile(seg, 95)) if seg else None,
                             'max_s_suggested': r3(1.5 * np.percentile(seg, 95)) if seg else None},
                      'S2': None, 'S3': None,
                      'note': 'p95 interpolated from %d episodes; S2/S3 durations not yet measured (no verified S2/S3 parts)' % len(seg)}}
    return cal


def requests(skill, n_ok, target, minimum, cal, n_unlab):
    need_min = max(0, minimum - n_ok)
    need_tgt = max(0, target - n_ok)
    if skill == 'S1':
        g = cal.get('gripper') or {}
        hs = ((cal.get('hold_pose') or {}).get('S1_hold_spots')) or {}
        rng = hs.get('range_max_minus_min')
        out = ['Need %d more usable S1 parts (%d for the target of %d). An S1 part runs from rest to 1.0 s after the lift begins, with the brick held between the jaws.' % (need_min, need_tgt, target),
               'Record S1 as its own task text (pick the brick and lift it to the inspection spot). Do not mix S1 takes with carry-and-release takes under the same task text.']
        if rng:
            out.append('Hold spot: S1 parts reach hold poses with a max joint range of %s units (shoulder_lift range %s, elbow_flex range %s; limit 15). Use one fixed inspection spot, marked on the table.' % (max(rng), rng[1], rng[2]))
        if g.get('closed_level'):
            out.append('Grasp check: across %d grasps the gripper state settles at a median of %s (range %s to %s) while the leader action is about 1. Confirm in the Studio preview that the brick is between the red jaws at closure before each take; re-record misses.' % (
                g['closed_level']['n'], g['closed_level']['median'], g['closed_level']['min'], g['closed_level']['max']))
        out.append('If several bricks lie clustered on the mat, pick only the target brick, keep the others out of the gripper path, and log the brick type (GOOD clean 2x2, or DEFECTIVE: tape on it, or not 2x2) per take.')
        out.append('Vary brick start positions across the pick zone; start each take at rest for at least 1.0 s.')
        return out
    colour = 'BLUE' if skill == 'S2' else 'GREEN'
    out = ['Need %d more usable %s parts (%d for the target of %d). An %s part runs from the hold spot with the brick held, through the carry, to release on the %s plate and return to rest.' % (
        need_min, skill, need_tgt, target, skill, colour),
        'Record %s as its own task text (carry the brick from the hold spot to the %s plate, release, return to rest). Keep the %s plate position fixed.' % (skill, colour, colour)]
    if n_unlab:
        out.append('%d carry-release parts are unlabelled (plate not confirmed on the contact sheet); label them before counting.' % n_unlab)
    return out


def gaps_build(all_eps, datasets, cal):
    G = {'generated_utc': utc(),
         'rule': 'counts are per segment (sub-episodes): a full pick-and-place episode counts once for S1 and once for S2 or S3; usable = segment status OK (brick verified visually, plate labelled, decode ok)',
         'needs': {}, 'datasets': []}
    for skill, (target, minimum) in NEEDS.items():
        mine = [(r, sg) for r in all_eps for sg in r['segments'] if sg['skill'] == skill]
        ok = [x for x in mine if x[1]['status'] == 'OK']
        wa = [x for x in mine if x[1]['status'] == 'WARN']
        bd = [x for x in mine if x[1]['status'] == 'BAD']

        def row(x):
            r, sg = x
            return {'dataset': r['dataset'], 'episode': r['episode'], 'start_idx': sg['start_idx'], 'end_idx': sg['end_idx'],
                    'start_t_s': sg['start_t_s'], 'end_t_s': sg['end_t_s'], 'global_start': sg['global_start'],
                    'global_end': sg['global_end'], 'why': '; '.join(sg['why'])}
        unlab = sum(1 for r in all_eps for sg in r['segments'] if sg['skill'] == 'S2/S3') if skill in ('S2', 'S3') else 0
        G['needs'][skill] = {'target': target, 'minimum': minimum, 'usable_OK_segments': len(ok),
                             'WARN_segments': len(wa), 'BAD_segments': len(bd),
                             'shortfall_to_target': max(0, target - len(ok)), 'shortfall_to_minimum': max(0, minimum - len(ok)),
                             'OK_segments': [row(x) for x in ok], 'WARN_segments': [row(x) for x in wa],
                             'BAD_segments': [row(x) for x in bd],
                             'requests': requests(skill, len(ok), target, minimum, cal, unlab)}
    G['unlabelled_S2_S3_segments'] = sum(1 for r in all_eps for sg in r['segments'] if sg['skill'] == 'S2/S3')
    G['episode_level'] = {'episodes': len(all_eps), 'OK': sum(1 for r in all_eps if r['verdict'] == 'OK'),
                          'WARN': sum(1 for r in all_eps if r['verdict'] == 'WARN'), 'BAD': sum(1 for r in all_eps if r['verdict'] == 'BAD')}
    for d in datasets:
        mine = [r for r in all_eps if r['dataset'] == d['id']]
        G['datasets'].append({'id': d['id'], 'studio_name': d['studio'].get('name'), 'compatible': d['compat']['compatible'],
                              'episodes': len(mine), 'recently_modified_may_be_mid_sync': d['recently_modified_may_be_mid_sync'],
                              'task_texts': sorted({r['task'] for r in mine})})
    return G


def sheet(eps, out_png):
    cols = [(c, k) for c in CAMS for k in KEYS]
    W = 170 + len(cols) * THUMB[0]
    H = 20 + len(eps) * (THUMB[1] + 4)
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    for j, (c, k) in enumerate(cols):
        d.text((176 + j * THUMB[0], 4), c.split()[-1][-6:] + ' ' + k, fill='black')
    for i, r in enumerate(eps):
        y = 20 + i * (THUMB[1] + 4)
        d.text((4, y + 4), 'ep%d\n%s %s\n%d frames' % (r['episode'], r['cls_final'], r['verdict'], r['data_rows']), fill='black')
        for j, (c, k) in enumerate(cols):
            x = 170 + j * THUMB[0]
            cell = r['decode'][c][k]
            if cell['png'] and os.path.exists(cell['png']):
                im.paste(Image.open(cell['png']).convert('RGB'), (x, y))
                d.text((x + 3, y + THUMB[1] - 12), 't=%ss' % cell['t_rel_s'], fill='yellow')
            else:
                d.rectangle([x, y, x + THUMB[0] - 1, y + THUMB[1] - 1], outline='red')
                d.text((x + 4, y + 4), 'n/a', fill='red')
    im.save(out_png)


def md_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join('' if v is None else str(v) for v in row) + ' |')
    return '\n'.join(lines)


def inventory_md(datasets, eps, cal, gaps, sync_last, incomplete, sheets, when):
    L = ['# INVENTORY (W3, P2-D-DATA)', '',
         'Generated %s by inventory.py from /home/intel/intel-junction-2026/data/studio-datasets (read-only). Last sync line (logs/sync_datasets.log, UTC): `%s`.' % (when, sync_last), '',
         '## Datasets', '']
    rows = []
    for d in datasets:
        s = d['studio']
        rows.append([d['id'], s.get('name'), s.get('default_task'), s.get('environment_id'), d['info']['robot_type'],
                     d['info']['fps'], d['episode_meta_rows'], d['episodes_in_data'], d['info']['total_frames'],
                     d['data_rows'], 'yes' if d['compat']['compatible'] else 'NO',
                     'yes' if d['recently_modified_may_be_mid_sync'] else 'no'])
    L += [md_table(['folder id', 'Studio name', 'default_task', 'environment_id', 'robot_type', 'fps', 'episodes (meta)',
                    'episodes (data)', 'total_frames (info)', 'rows (data)', 'compatible', 'recently modified (<150 s)'], rows), '']
    for d in datasets:
        L += ['## Dataset %s' % d['id'], '']
        L.append('- info.json: codebase %s, fps %s, robot_type %s, total_episodes %s, total_frames %s, total_tasks %s, splits %s.' % (
            d['info']['codebase_version'], d['info']['fps'], d['info']['robot_type'], d['info']['total_episodes'],
            d['info']['total_frames'], d['info']['total_tasks'], d['info']['splits']))
        L.append('- tasks: ' + '; '.join('%s = "%s"' % (t['task_index'], t['task']) for t in d['tasks']) + '.')
        L.append('- joint names (observation.state; action names equal: %s): %s.' % (d['action_names_equal_state'], ', '.join(d['state_names'])))
        L.append('- video keys: ' + ', '.join(d['video_feature_keys']) + '.')
        L.append('- data files: ' + ', '.join(d['data_files']) + '.')
        L.append('- Studio (GET /api/projects and /api/dataset/{id}): name "%s", default_task "%s", environment_id %s. The API returns the environment id only, so the environment name "SO101 Recording - two cameras" is NOT verified.' % (
            d['studio'].get('name'), d['studio'].get('default_task'), d['studio'].get('environment_id')))
        L.append('- compatibility: ' + ', '.join('%s=%s' % (k, v) for k, v in d['compat'].items()) + '.')
        L.append('')
        frows = [[k, v['expected_frames'], v['packets'], 'ok' if v['ok'] else 'MISMATCH', ', '.join(str(x) for x in v['episodes'])] for k, v in d['files'].items()]
        L += [md_table(['camera|video file', 'expected frames (meta)', 'packets (ffprobe)', 'check', 'episodes'], frows), '']
    L += ['## Per-episode table', '',
          'class = segments found (S1 = start to 1.0 s after lift start; S2/S3 = S1 cut to release and return to rest). closures = full/partial. open and closed lvl = gripper state before and after the final closure. state-action = state minus leader action at closed level. end-hold = distance (5 arm joints) from end pose to own hold pose; whole-ep S1 = holds to the end, no release. decode = successful frame extractions per camera (start, grasp, hold, release, end).', '']
    rows = []
    for r in eps:
        dec = []
        for c in CAMS:
            oks = [r['decode'][c][k]['ok'] for k in KEYS]
            dec.append('%d/%d' % (sum(1 for x in oks if x), sum(1 for x in oks if x is not None)))
        rows.append([r['dataset'][:8], r['episode'], r['verdict'], r['cls_final'], r['data_rows'], r['duration_s'], r['idle_start_s'],
                     '%d/%d' % (r['closures_full'], r['closures_partial']), r['grasp_t_s'], r['grasp_open_level'],
                     r['closed_level'], r['closed_gap_state_minus_action'], r['lift_t_s'], r['hold_t_s'], r['end_dist_own_hold'],
                     r['whole_episode_S1'], r['release_t_s'], r['return_idx'], ' / '.join(dec), '; '.join(r['reasons'])])
    L += [md_table(['dataset', 'ep', 'verdict', 'segments', 'frames', 'dur s', 'idle start s', 'closures full/partial', 'grasp t s',
                    'open lvl', 'closed lvl', 'state-action', 'lift t s', 'hold t s', 'end-hold', 'whole-ep S1', 'release t s',
                    'return idx', 'decode cam1 / cam2', 'reasons'], rows), '']
    L += ['## Segments (sub-episodes; frame indices are episode-relative, global = dataset row index)', '']
    rows = []
    for r in eps:
        for sg in r['segments']:
            rows.append([r['dataset'][:8], r['episode'], sg['skill'], sg['status'], sg['start_idx'], sg['end_idx'], sg['start_t_s'],
                         sg['end_t_s'], sg['duration_s'], sg['global_start'], sg['global_end'], sg.get('plate_label'),
                         '; '.join(sg['why'])])
    L += [md_table(['dataset', 'ep', 'skill', 'status', 'start idx', 'end idx', 'start t s', 'end t s', 'dur s', 'global start',
                    'global end', 'plate', 'why'], rows), '']
    L += ['## Calibration summary (calib.json)', '']
    gr = cal.get('gripper')
    if gr:
        op, cdv, ga = gr['open_level'] or {}, gr['closed_level'] or {}, gr['closed_minus_action'] or {}
        L.append('- gripper, %s grasps: open level median %s (range %s to %s); closed level median %s (range %s to %s); state minus action at closed, median %s. Closing = decreasing. Onset rule: velocity below %s units/s sustained >= %s s; settled below %s.' % (
            op.get('n'), op.get('median'), op.get('min'), op.get('max'), cdv.get('median'), cdv.get('min'), cdv.get('max'),
            ga.get('median'), CLOSE_V, RUN_MIN_S, CLOSED_MAX))
        hs = cal['hold_pose']['S1_hold_spots'] or {}
        L.append('- S1 hold spots median %s, max joint range %s (within 15 units: %s); whole-episode S1 count %s.' % (
            hs.get('median'), hs.get('range_max_minus_min'), cal['hold_pose']['consistent_within_15_units'],
            cal['hold_pose']['whole_episode_S1_count']))
        hm = cal['home_pose']['stats'] or {}
        L.append('- home pose median %s (n=%s); stationary: arm speed below %s units/s for >= %s s.' % (hm.get('median'), hm.get('n'), STAT_THR, DWELL_S))
        s1 = cal['skills']['S1']
        L.append('- S1 duration, movement onset to hold: p50 %s s, p95 %s s, suggested max_s %s s (n=%s). S2 and S3: not yet measured.' % (
            s1['duration_s_p50'], s1['duration_s_p95'], s1['max_s_suggested'], s1['n']))
    L += ['', '## Gaps (per skill, sub-episodes counted)', '']
    rows = [[k, v['target'], v['minimum'], v['usable_OK_segments'], v['WARN_segments'], v['BAD_segments'],
             v['shortfall_to_target'], v['shortfall_to_minimum']] for k, v in gaps['needs'].items()]
    L += [md_table(['skill', 'target', 'minimum', 'usable OK segments', 'WARN segments', 'BAD segments', 'shortfall to target', 'shortfall to minimum'], rows), '']
    L += ['## Contact sheets', '']
    L += ['- laptop: %s' % p for p in sheets] if sheets else ['- none']
    L += ['', '## Incomplete or unreadable folders', '']
    L += ['- ' + x for x in incomplete] if incomplete else ['- none']
    return '\n'.join(L) + '\n'


def fingerprint(root):
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(root, '**', '*'), recursive=True)):
        if os.path.isfile(p):
            st = os.stat(p)
            h.update(('%s|%d|%d' % (p, st.st_size, st.st_mtime_ns)).encode())
    return h.hexdigest()


def write(path, text):
    with open(path + '.tmp', 'w') as f:
        f.write(text)
    os.replace(path + '.tmp', path)


def build(a):
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, 'frames')
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(os.path.join(a.out, 'contact'), exist_ok=True)
    manual = json.load(open(a.manual)) if a.manual and os.path.exists(a.manual) else {}
    studio = json.loads(a.studio) if a.studio else {}
    datasets, all_eps, incomplete, allc, names, sheets = [], [], [], [], None, []
    for d in sorted(glob.glob(os.path.join(a.root, '*'))):
        if not os.path.isdir(d):
            continue
        did = os.path.basename(d)
        if not os.path.isfile(os.path.join(d, 'meta', 'info.json')):
            incomplete.append('%s: no meta/info.json yet (still syncing or empty)' % did)
            continue
        try:
            D, eps = analyse(d, tmp, studio.get(did, {}), manual.get(did, {}))
        except Exception as ex:
            incomplete.append('%s: unreadable, possibly mid-sync (%s)' % (did, str(ex)[:200]))
            continue
        datasets.append(D)
        all_eps += eps
        if names is None:
            names = D['state_names']
        if D['compat']['compatible']:
            allc += eps
        for p in range(0, len(eps), PAGE):
            out_png = os.path.join(a.out, 'contact', '%s_p%d.png' % (did, p // PAGE + 1))
            sheet(eps[p:p + PAGE], out_png)
            sheets.append(out_png)
    gpool = [r for r in allc if r['grasp_t_s'] is not None]
    s1p = [r for r in allc if r['grasp_t_s'] is not None and r['hold_t_s'] is not None]
    cal = calibrate(allc, gpool, s1p, names or [])
    gaps = gaps_build(all_eps, datasets, cal)
    sync_last = ''
    if a.sync_log and os.path.exists(a.sync_log):
        lines = open(a.sync_log).read().strip().splitlines()
        sync_last = lines[-1] if lines else ''
    write(os.path.join(a.out, 'calib.json'), json.dumps(cal, indent=1))
    write(os.path.join(a.out, 'gaps.json'), json.dumps(gaps, indent=1))
    write(os.path.join(a.out, 'episodes.json'), json.dumps(all_eps, indent=1, default=str))
    write(os.path.join(a.out, 'INVENTORY.md'), inventory_md(datasets, all_eps, cal, gaps, sync_last, incomplete, sheets, utc()))
    write(os.path.join(a.out, 'contact_sheets.txt'), '\n'.join(sheets) + '\n')
    print('BUILD_OK', utc(), 'datasets', len(datasets), 'episodes', len(all_eps), 'grasps', len(gpool), 'S1_parts_with_hold', len(s1p), 'sheets', len(sheets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sync-log', default='')
    ap.add_argument('--manual', default='')
    ap.add_argument('--studio', default='')
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--every', type=float, default=120.0)
    ap.add_argument('--max-cycles', type=int, default=0)
    a = ap.parse_args()
    if not a.watch:
        build(a)
        return
    last, cycles = None, 0
    while True:
        fp = fingerprint(a.root)
        if fp != last:
            build(a)
            last = fp
            cycles += 1
        if a.max_cycles and cycles >= a.max_cycles:
            break
        time.sleep(a.every)


if __name__ == '__main__':
    main()
