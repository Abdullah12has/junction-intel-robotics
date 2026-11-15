"""Plot joint vs time for each of the four sequences from the fake-bus setpoint CSVs (matplotlib, Agg backend)."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

E1 = Path("/home/intel/intel-junction-2026/work/P2-B-GM/E1")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SEQS = ["blue_plate_to_home", "blue_home_to_plate", "green_plate_to_home", "green_home_to_plate"]


def main() -> None:
    (E1 / "plots").mkdir(parents=True, exist_ok=True)
    for seq in SEQS:
        with (E1 / "fake_bus" / f"{seq}_setpoints.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        t0 = float(rows[0]["t_s"])
        t = [float(r["t_s"]) - t0 for r in rows]
        fig, axes = plt.subplots(len(JOINTS), 1, sharex=True, figsize=(9, 10))
        for ax, joint in zip(axes, JOINTS, strict=True):
            ax.plot(t, [float(r[joint]) for r in rows], lw=1.0)
            ax.set_ylabel(joint, fontsize=8)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("virtual time, s (fake bus; includes 0.5 s settle per waypoint)")
        fig.suptitle(f"{seq}: executed setpoints (speed cap 25 units/s, gripper fixed 25)", fontsize=10)
        fig.tight_layout()
        fig.savefig(E1 / "plots" / f"{seq}.png", dpi=80)
        plt.close(fig)
        print("wrote", E1 / "plots" / f"{seq}.png", "points", len(t))


if __name__ == "__main__":
    main()
