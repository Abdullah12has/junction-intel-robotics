"""P3-TWIN-DYN E2 servo model: run with coefficients in servo_arx_v2.json (per_joint_fit_on_all_17)."""
def run_servo_arx(cfg, goal, grav, y0):
    """Open-loop servo model. cfg: per-joint dicts with a1, a2, b0, b1, c, e, d (frames).
    goal: (n, 6) normalized goal per frame. grav(q_norm) -> (6,) gravity torque, N m.
    y0: (2, 6) first two normalized present frames. Returns (n, 6) predicted normalized present."""
    import numpy as np
    n = goal.shape[0]
    y = np.zeros((n, 6))
    y[:2] = y0[:2]
    for k in range(2, n):
        g = grav(y[k - 1])
        for j, p in enumerate(cfg):
            d = p["d"]
            u0 = goal[max(k - d, 0), j]
            u1 = goal[max(k - d - 1, 0), j]
            y[k, j] = p["a1"] * y[k - 1, j] + p["a2"] * y[k - 2, j] + p["b0"] * u0 + p["b1"] * u1 + p["c"] * g[j] + p["e"]
    return y


def make_grav(urdf_path, k):
    """k: rad per normalized unit (6,). Returns grav(q_norm) -> gravity torque (6,) N m via MuJoCo mj_rne."""
    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(urdf_path)
    d = mujoco.MjData(m)
    out = np.zeros(6)

    def grav(qn):
        d.qpos[:] = np.asarray(qn) * np.asarray(k)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        mujoco.mj_rne(m, d, 0, out)
        return out.copy()
    return grav
