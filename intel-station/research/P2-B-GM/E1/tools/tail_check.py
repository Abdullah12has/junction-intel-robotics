"""Diagnostic (read-only): arrival at home near episode end, stationary run length, and recent speed profile."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_releases as ar  # noqa: E402

EPISODES = [int(x) for x in sys.argv[1].split(",")]


def main() -> None:
    data = ar.read_all("data/chunk-000/*.parquet")
    data = data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    states = np.stack(data["observation.state"].to_numpy()).astype(np.float64)
    ep_ids = data["episode_index"].to_numpy()
    for e in EPISODES:
        s = states[ep_ids == e]
        g = s[:, 5]
        q = s[:, :5]
        n = len(s)
        v = np.zeros(n)
        v[:-1] = np.linalg.norm(np.diff(q, axis=0), axis=1) * ar.FPS
        rels = ar.release_frames(g)
        print(f"== ep {e} n={n} releases={rels}")
        for t in rels:
            hit = [k for k in range(t, n) if abs(q[k, 1] - ar.HOME_LIFT) <= ar.HOME_TOL]
            if not hit:
                print(f"   release {t}: lift never within tol after release")
                continue
            k0 = hit[0]
            run = 0
            for k in range(k0, n - 1):
                if v[k] < ar.STILL_SPEED:
                    run += 1
                else:
                    break
            print(f"   release {t}: first lift-in-tol frame {k0} (t={k0 / ar.FPS:.3f}s), still run from there = {run} frames (need 15)")
        start = max(0, n - 60)
        print("   last 60 frames (t_s, lift, pan, v_units_s):")
        for k in range(start, n, 4):
            print(f"     {k / ar.FPS:6.3f} {q[k, 1]:9.3f} {q[k, 0]:9.3f} {v[k]:8.3f}")


if __name__ == "__main__":
    main()
