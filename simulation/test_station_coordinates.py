import numpy as np
import pytest
from station_coordinates import EncoderCoordinates, StationCoordinates, NOMINAL, JOINTS
from sorting_env import BrickSortingEnv
from normalized_env import NormalizedSortingEnv

K = 2*np.pi/4096


def test_driver_conversion_uses_truncation_and_preserves_out_of_range_observations():
    c=EncoderCoordinates();n=np.array([.1,.1,.1,.1,.1,50.1])
    expected=np.array([int((n[i]+100)/200*(c.tick_max[i]-c.tick_min[i])+c.tick_min[i]) if i<5
                       else int(n[i]/100*(c.tick_max[i]-c.tick_min[i])+c.tick_min[i]) for i in range(6)])
    np.testing.assert_array_equal(c.to_encoder_ticks(n),expected)
    assert np.any(expected != np.rint(c.continuous_ticks(n)))
    np.testing.assert_array_equal(c.from_encoder_ticks(c.tick_max+1),c.high)
    assert np.all(c.from_encoder_ticks(c.tick_max+1,clip=False)>c.high)


def test_nominal_mapping_has_encoder_scale_and_intersected_limits():
    env=BrickSortingEnv()
    try:
        c=StationCoordinates(env.limits)
        n=np.array([0.,0.,0.,0.,0.,50.]);q=c.to_model(n)
        np.testing.assert_allclose(c.from_model(q),n,atol=1e-10)
        # One normalized unit is the encoder's angle, never the model range's.
        np.testing.assert_allclose(c.to_model(n+1)-q,(c.tick_max-c.tick_min)/(c.high-c.low)*K,atol=1e-12)
        np.testing.assert_allclose(c.from_model(np.zeros(6)),c.from_encoder_ticks(np.full(6,2048.),clip=False),atol=1e-10)
        # Common range never exceeds either the model's or the encoder's.
        assert np.all(c.to_model(c.command_low)>=env.limits[:,0]-1e-9) and np.all(c.to_model(c.command_high)<=env.limits[:,1]+1e-9)
        assert np.all(c.command_low>=c.low) and np.all(c.command_high<=c.high)
        with pytest.raises(ValueError):c.to_model(c.command_high+1)
    finally:env.close()


def test_gripper_zero_is_closed_and_opens_with_increasing_command():
    env=BrickSortingEnv()
    try:
        c=StationCoordinates(env.limits)
        q=c.to_model(np.array([[0,0,0,0,0,0.],[0,0,0,0,0,60.]]))[:,5]
        # The expert closes near -0.08 rad and opens at 0.7 rad.
        assert abs(q[0])<.01 and q[1]>.7
    finally:env.close()


def test_sign_and_zero_overrides_and_rejects_bad_mapping():
    env=BrickSortingEnv()
    try:
        m={**NOMINAL,'joints':{j:dict(v) for j,v in NOMINAL['joints'].items()}}
        m['joints']['elbow_flex']={'sign':-1,'zero_tick':2100}
        c=StationCoordinates(env.limits,mapping=m);i=JOINTS.index('elbow_flex')
        ticks=np.full(6,2048.);ticks[i]=2200
        assert np.isclose(c.to_model(c.from_encoder_ticks(ticks))[i],-100*K)
        assert c.mapping_sha256!=StationCoordinates(env.limits).mapping_sha256
        m['joints']['elbow_flex']['sign']=2
        with pytest.raises(ValueError):StationCoordinates(env.limits,mapping=m)
        with pytest.raises(ValueError):StationCoordinates(env.limits,mapping={'joints':{}})
    finally:env.close()


def test_wrapper_preserves_physics_and_reports_unverified():
    normalized=NormalizedSortingEnv();native=BrickSortingEnv()
    try:
        normalized.reset(42);native.reset(42)
        n=np.array([0.,-15.,20.,25.,0.,40.])
        for _ in range(10):
            obs,*_,info=normalized.step(n)
            native.step(normalized.coordinates.to_model(n))
            np.testing.assert_allclose(normalized.data.qpos,native.data.qpos,atol=1e-12)
            expected=normalized.coordinates.from_model(native.observe()['observation.state'])
            np.testing.assert_allclose(obs['observation.state'],expected,atol=1e-5)
        assert not info['hardware_mapping_verified']
    finally:normalized.close();native.close()


@pytest.mark.parametrize('action', [[101,0,0,0,0,50], [0,0,0,0,0,-1], [0,0,0,0,0,float('nan')], [0]*5])
def test_bad_actions_leave_sim_unchanged(action):
    env = NormalizedSortingEnv()
    try:
        env.reset(1)
        before = env.data.qpos.copy()
        with pytest.raises(ValueError): env.step(action)
        np.testing.assert_array_equal(before, env.data.qpos)
    finally: env.close()
