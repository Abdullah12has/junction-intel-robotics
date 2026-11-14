"""Station-normalized policy boundary over the same MuJoCo physics."""
import numpy as np

from sorting_env import BrickSortingEnv
from station_coordinates import StationCoordinates, UNITS


class NormalizedSortingEnv:
    """reset/observe/step in station units; MuJoCo internals and the IK expert stay in radians."""
    action_units = UNITS

    def __init__(self, render=False, horizon=900, mapping=None):
        self.native = BrickSortingEnv(render=render, horizon=horizon)
        try:
            self.coordinates = StationCoordinates(self.native.limits, mapping=mapping)
        except Exception:
            self.native.close()
            raise
        self.limits = np.column_stack((self.coordinates.command_low, self.coordinates.command_high))

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _observation(self, obs):
        # Observations are not clipped: limit overshoots stay visible.
        return {**obs, 'observation.state': self.coordinates.from_model(obs['observation.state']).astype(np.float32)}

    def reset(self, *args, **kwargs):
        return self._observation(self.native.reset(*args, **kwargs))

    def observe(self):
        return self._observation(self.native.observe())

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (6,):
            raise ValueError('Action must have shape (6,)')
        obs, reward, done, truncated, info = self.native.step(self.coordinates.to_model(action))
        verified = self.coordinates.mapping.get('status') == 'verified'
        return self._observation(obs), reward, done, truncated, {**info, 'action_units': UNITS, 'hardware_mapping_verified': verified}

    def close(self):
        self.native.close()
