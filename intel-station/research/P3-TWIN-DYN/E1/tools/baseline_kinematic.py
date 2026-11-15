#!/usr/bin/env python3
"""P3-TWIN-DYN E1: candidate B = no-dynamics (kinematic) twin, scored with the same leave-one-episode-out folds and
the same G2 metric as the identified model.

Prediction: present_hat(t) = goal(t - delta_j), zero-order hold of the logged goal (the twin assumes the joint
reaches the commanded goal after a pure delay and has no sag, friction or speed limit). delta_j is fitted on the 16
training episodes of each fold (grid 0..0.30 s) and scored on the held-out episode. Lag error = best extra shift of the
prediction on the held-out episode, pooled over held-out rows, in frames (1/30 s).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import twin_lib as T  # noqa: E402

WORK = f"{T.R}/work/P3-TWIN-DYN/E1"
DELAYS = np.arange(0.0, 0.3001, 0.005)
SHIFTS = np.arange(-0.15, 0.1501, 0.005)
FRAME = 1.0 / 30.0


def goal_at(ep, query):
    """Zero-order hold of the logged goal at query times; query is (n,) or (n, 6) with one time column per joint."""
    query = np.asarray(query, dtype=float)
    if query.ndim == 1:
        idx = np.searchsorted(ep["t"], query, side="right") - 1
        return ep["goal"][np.clip(idx, 0, None)]
    out = np.zeros((query.shape[0], 6))
    for j in range(6):
        idx = np.searchsorted(ep["t"], query[:, j], side="right") - 1
        out[:, j] = ep["goal"][np.clip(idx, 0, None), j]
    return out


def sq_cnt(ep, shift):
    """Per joint: sum of squared normalized errors (goal shifted by shift, minus logged present), masked rows."""
    err = goal_at(ep, ep["t"] - shift) - ep["pos"]
    sq = np.zeros(6)
    cnt = np.zeros(6)
    for j in range(6):
        m = T.mask_for(ep, j)
        sq[j] = np.sum(err[m, j] ** 2)
        cnt[j] = m.sum()
    return sq, cnt


def main():
    eps = T.load_episodes()
    tot_sq = np.zeros(6)
    tot_cnt = np.zeros(6)
    lag_tot = np.zeros((len(SHIFTS), 6))
    per_ep = []
    for e in range(len(eps)):
        train = [ep for i, ep in enumerate(eps) if i != e]
        held = eps[e]
        S = np.zeros((len(DELAYS), 6))
        for di, d in enumerate(DELAYS):
            for ep in train:
                S[di] += sq_cnt(ep, d)[0]
        delta = np.array([DELAYS[int(np.argmin(S[:, j]))] for j in range(6)])
        sq = np.zeros(6)
        cnt = np.zeros(6)
        err_sq_shift = np.zeros((len(SHIFTS), 6))
        for si, s in enumerate(SHIFTS):
            sq_s = np.zeros(6)
            # shift applied on top of each joint's fitted delay
            err = goal_at(held, held["t"][:, None] - delta[None, :] - s) - held["pos"]
            for j in range(6):
                m = T.mask_for(held, j)
                sq_s[j] = np.sum(err[m, j] ** 2)
            err_sq_shift[si] = sq_s
        base_sq, base_cnt = sq_cnt(held, 0.0)
        # held-out error with each joint's own fitted delay (shift 0 on top of delta)
        err0 = goal_at(held, held["t"][:, None] - delta[None, :]) - held["pos"]
        for j in range(6):
            m = T.mask_for(held, j)
            sq[j] = np.sum(err0[m, j] ** 2)
            cnt[j] = m.sum()
        tot_sq += sq
        tot_cnt += cnt
        lag_tot += err_sq_shift
        rmse_e = np.sqrt(sq / np.maximum(cnt, 1))
        per_ep.append({"fold": e, "episode": held["name"], "delta_s": [round(float(x), 3) for x in delta],
                       "rmse_norm": [round(float(x), 4) for x in rmse_e],
                       "zero_delay_rmse_norm": [round(float(x), 4) for x in np.sqrt(base_sq / np.maximum(base_cnt, 1))]})
        print(e, held["name"], np.round(delta, 3).tolist(), np.round(rmse_e, 3).tolist(), flush=True)
    loeo = np.sqrt(tot_sq / np.maximum(tot_cnt, 1))
    best_shift = np.array([SHIFTS[int(np.argmin(lag_tot[:, j]))] for j in range(6)])
    lag_frames = np.abs(best_shift) / FRAME
    g2_rmse = bool(np.all(loeo <= 0.5))
    g2_lag = bool(np.all(lag_frames <= 1.0))
    out = {"candidate": "B kinematic (no dynamics): present = goal delayed by fitted delay",
           "loeo_rmse_norm": loeo.tolist(), "loeo_lag_frames": lag_frames.tolist(),
           "loeo_lag_shift_s": best_shift.tolist(), "loeo_rows_scored": tot_cnt.tolist(),
           "g2_rmse_pass": g2_rmse, "g2_lag_pass": g2_lag, "g2_pass": g2_rmse and g2_lag,
           "per_episode": per_ep}
    with open(f"{WORK}/baseline_kinematic.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("baseline LOEO rmse normalized", np.round(loeo, 3).tolist())
    print("baseline lag frames", np.round(lag_frames, 2).tolist(), "g2_rmse", g2_rmse, "g2_lag", g2_lag)


if __name__ == "__main__":
    main()
