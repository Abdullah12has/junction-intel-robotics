import numpy as np
import pytest

from expert import PickPlaceExpert
from sorting_env import BrickSortingEnv, PLATES


@pytest.mark.parametrize('destination', ['blue','green'])
def test_actual_contact_lift_release_and_placement(destination):
    env=BrickSortingEnv(horizon=1800)
    env.reset(50,destination=destination)
    start=env.data.body(env.brick_body_name).xpos.copy()
    expert=PickPlaceExpert(env)
    assert expert.run()
    assert any(len(t['gripper_contact_bodies'])==2 and t['object_xyz'][2]>.06 for t in expert.trace)
    end=env.data.body(env.brick_body_name).xpos
    assert np.linalg.norm(end[:2]-np.array(PLATES[destination]))<.02
    assert np.linalg.norm(end[:2]-start[:2])>.05
    assert expert.trace[-1]['gripper_contact_bodies']==[]
    assert expert.trace[-1]['success']


def test_reset_friction_does_not_depend_on_previous_target():
    env=BrickSortingEnv();fresh=BrickSortingEnv()
    try:
        env.reset(1003,target_color='orange')
        env.reset(1142,target_color='red',destination='blue')
        fresh.reset(1142,target_color='red',destination='blue')
        np.testing.assert_array_equal(env.model.geom_friction,fresh.model.geom_friction)
        np.testing.assert_array_equal(env.data.qpos,fresh.data.qpos)
    finally:
        env.close();fresh.close()
