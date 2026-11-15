import mujoco
import numpy as np
import pytest

from sorting_env import BrickSortingEnv, COLORS, PLATES, PLATE_TOP


def test_randomized_scene_is_reproducible_separated_and_varied():
    env = BrickSortingEnv()
    counts, destinations = set(), set()
    for seed in range(30):
        env.reset(seed)
        counts.add(len(env.active_indices))
        destinations.add(env.destination)
        assert env.target_index in env.active_indices
        points = [env.data.body(env.blocks[i]['body']).xpos.copy() for i in env.active_indices]
        for i, p in enumerate(points):
            for q in points[i + 1:]:
                assert np.linalg.norm(p[:2] - q[:2]) > 0.04
        initial, task = env.data.qpos.copy(), env.instruction
        env.reset(seed)
        np.testing.assert_array_equal(initial, env.data.qpos)
        assert task == env.instruction
    assert counts == {4, 5, 6}
    assert destinations == {'blue', 'green'}
    assert PLATES['blue'][0] < PLATES['green'][0]


@pytest.mark.parametrize('destination', ['blue', 'green'])
@pytest.mark.parametrize('correct_plate', [True, False])
def test_only_requested_plate_counts(destination, correct_plate):
    env = BrickSortingEnv()
    obs = env.reset(42, target_color='purple', destination=destination)
    a = env.brick_adr
    plate = destination if correct_plate else ('green' if destination == 'blue' else 'blue')
    xy = PLATES[plate]
    env.data.qpos[a:a + 3] = [*xy, 0.08]
    mujoco.mj_forward(env.model, env.data)
    env.step(obs['observation.state'])
    assert env.lifted
    env.data.qpos[a:a + 3] = [*xy, PLATE_TOP + env.half_height]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(60):
        _, _, done, _, info = env.step(obs['observation.state'])
        if done:
            break
    assert info['success'] == correct_plate


def test_distractor_does_not_count_and_both_destinations_use_same_layout():
    env = BrickSortingEnv()
    obs = env.reset(42, target_color='purple', destination='blue', count=6)
    before = env.data.qpos.copy()
    env.reset(42, target_color='purple', destination='green', count=6)
    np.testing.assert_array_equal(before, env.data.qpos)
    distractor = next(i for i in env.active_indices if i != env.target_index)
    a = env.model.joint(env.blocks[distractor]['joint']).qposadr[0]
    env.data.qpos[a:a + 3] = [*PLATES['green'], PLATE_TOP + env.half_height]
    mujoco.mj_forward(env.model, env.data)
    for _ in range(60):
        _, _, _, _, info = env.step(obs['observation.state'])
    assert not info['success']
    assert obs['task'] == 'move the purple brick to the blue plate'


def test_reference_layout_and_station_scale():
    env = BrickSortingEnv()
    env.reset(42, reference=True)
    assert [env.block_colors[i] for i in env.active_indices] == ['red', 'yellow', 'orange', 'white', 'cyan']
    np.testing.assert_allclose(env.model.geom('brick_geom').size, [.0159, .0159, .0096])
    assert env.model.camera('wrist').bodyid == 0
    assert env.model.geom('black_mat').rgba[0] < .15
    assert env.model.geom('plate_blue').pos[0] < env.model.geom('plate_green').pos[0]
    # Visual ring mesh is excluded from contacts; invisible cylinder approximates stud contact.
    for block in env.blocks:
        if block['body'] == 'brick':
            assert np.any(env.model.geom_rgba[block['geoms'], 3] == 0)
            assert np.any(env.model.geom_contype[block['geoms']] == 0)
