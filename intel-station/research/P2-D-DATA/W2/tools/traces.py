import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
out_png = Path(sys.argv[2])
out_png.parent.mkdir(parents=True, exist_ok=True)
fps = 30.0
files = sorted((root / "data").rglob("*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
S = np.stack(df["observation.state"].to_numpy()).astype(float)
ep = df["episode_index"].to_numpy()
eps = np.unique(ep)
names = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]

fig, axes = plt.subplots(len(eps), 1, figsize=(13, 2.3 * len(eps)))
allv = []
for ax, e in zip(axes, eps):
    s = S[ep == e]
    t = np.arange(len(s)) / fps
    for j in range(5):
        ax.plot(t, s[:, j], lw=0.8, label=names[j])
    v = np.linalg.norm(np.diff(s[:, :5], axis=0), axis=1) * fps
    vs = pd.Series(v).rolling(5, center=True, min_periods=1).median().to_numpy()
    allv.append(vs)
    ax2 = ax.twinx()
    ax2.plot(t, s[:, 5], "k", lw=1.6, label="grip")
    ax2.plot(t[1:], vs, "r", lw=0.6, alpha=0.7, label="speed")
    ax.set_title(f"episode {e} frames={len(s)}", fontsize=9)
    ax.set_xlim(0, t[-1])
axes[0].legend(loc="upper left", fontsize=7, ncol=6)
fig.tight_layout()
fig.savefig(out_png, dpi=70)
print("PNG", out_png)
sv = np.concatenate(allv)
print("SPEED_PCT deg/s p10,p25,p50,p75,p90,p95,p99:", np.round(np.percentile(sv, [10, 25, 50, 75, 90, 95, 99]), 2).tolist())
for e in eps:
    s = S[ep == e]
    g = s[:, 5]
    L = len(s)
    v = np.linalg.norm(np.diff(s[:, :5], axis=0), axis=1) * fps
    mv = np.where(v > 5.0)[0]
    first = mv[0] / fps if len(mv) else -1
    last = mv[-1] / fps if len(mv) else -1
    print(f"EP {e} L={L} dur={L/fps:.2f}s g0={g[0]:.1f} gend={g[-1]:.1f} gmin={g.min():.1f}@{g.argmin()/fps:.2f}s gmax={g.max():.1f}@{g.argmax()/fps:.2f}s move_first={first:.2f}s move_last={last:.2f}s")
    print("  GRIP every 0.5s:", np.round(g[::15], 0).astype(int).tolist())
    print("  PAN every 0.5s:", np.round(s[::15, 0], 0).astype(int).tolist())
