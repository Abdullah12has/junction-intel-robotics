import numpy as np
import pytest

from so101_env import BrickInspectionEnv
from prepare_training import select_episodes
import mujoco


def test_seeded_reset_and_control_clock():
    env = BrickInspectionEnv()
    try:
        obs = env.reset(seed=15)
        original = env.data.qpos.copy()
        for _ in range(30):
            env.step(obs["observation.state"])
        assert env.data.time == pytest.approx(1, abs=1e-10)
        assert np.isfinite(env.data.qpos).all()
        env.reset(seed=15)
        np.testing.assert_array_equal(original, env.data.qpos)
        env.reset(seed=16)
        assert not np.array_equal(original, env.data.qpos)
    finally:
        env.close()


@pytest.mark.parametrize("action", [[float("nan")] * 6, [0] * 5, [100] * 6])
def test_invalid_action_rejected_before_simulation_advances(action):
    env = BrickInspectionEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.data.time == 0


def test_horizon_and_reset():
    env = BrickInspectionEnv(horizon=1)
    obs = env.reset()
    _, _, terminated, truncated, _ = env.step(obs["observation.state"])
    assert truncated and not terminated
    with pytest.raises(RuntimeError):
        env.step(obs["observation.state"])
    env.reset()
    env.step(obs["observation.state"])


def test_train_validation_disjoint_and_review_required():
    rows = [{"episode_index": str(i), "accepted": "yes", "split": "train" if i < 40 else "validation"} for i in range(50)]
    train, val = select_episodes(rows, set(range(50)))
    assert not set(train) & set(val)
    rows[-1]["episode_index"] = "0"
    with pytest.raises(ValueError, match="unique"):
        select_episodes(rows, set(range(50)))
    rows[-1]["accepted"] = ""
    with pytest.raises(ValueError, match="50"):
        select_episodes(rows, set(range(50)))


@pytest.mark.parametrize("lift_first", [False, True])
@pytest.mark.parametrize("brick_type", ["2x2", "2x4"])
def test_placement_requires_lift_and_settling(lift_first, brick_type):
    env = BrickInspectionEnv(brick_type=brick_type)
    obs = env.reset()
    a = env.brick_adr
    target = env.model.site("inspection_target").pos[:2]
    if lift_first:
        env.data.qpos[a:a + 3] = [*target, 0.08]
        mujoco.mj_forward(env.model, env.data)
        env.step(obs["observation.state"])
        assert env.lifted
    env.data.qpos[a:a + 3] = [*target, env.half_height]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(60):
        _, _, terminated, _, info = env.step(obs["observation.state"])
        if terminated:
            break
    assert info["success"] == lift_first
