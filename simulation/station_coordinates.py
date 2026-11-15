"""Station-normalized <-> MuJoCo joint coordinates. Offline only; no device access."""
import hashlib
import json
from pathlib import Path
import numpy as np

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
UNITS = 'so101_station_normalized_v1'
ROOT = Path(__file__).parent
# STS3215 is direct drive with a 4096-tick encoder, so the scale is hardware fact.
# Only direction and zero are per-joint unknowns. Nominal: LeRobot homes the
# mid-pose to tick 2048 and the Menagerie model's zero is that same mid-pose.
# Sim-side convention only: this is NOT hardware calibration and changes none.
# Signs were checked on the station 2026-11-14; zeros are good to about 5 degrees.
# Known gap: holding a 32 mm brick reads 2.2 on the station (flexible fingers) and
# 5.8 in the model (rigid jaw). Shifting the gripper zero to close that gap leaves
# the model too little squeeze (expert success fell from 30/30 to 17/30), so it stays.
# Evidence: NORMALIZED-INTERFACE.md.
# Shoulder lift, elbow and wrist flex zeros come from the so101-mujoco touch calibration of this
# same arm (six hand-guided fingertip touches on the table, fit rms 1.1 mm), converted from its
# "range midpoint + offset" convention. They sit 0.3, 1.1 and 1.6 degrees from the nominal 2048.
NOMINAL = {'status': 'station_checked', 'ticks_per_revolution': 4096,
           'joints': {**{j: {'sign': 1, 'zero_tick': 2048} for j in
                         ('shoulder_pan', 'wrist_roll', 'gripper')},
                      'shoulder_lift': {'sign': 1, 'zero_tick': 2044.5},
                      'elbow_flex': {'sign': 1, 'zero_tick': 2035.3},
                      'wrist_flex': {'sign': 1, 'zero_tick': 2029.5}}}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class EncoderCoordinates:
    """Exactly the station's range-normalized driver boundary (already-homed ticks)."""
    def __init__(self, config=ROOT / 'arm_config.json'):
        self.config = json.loads(Path(config).read_text())
        if self.config['joint_order'] != list(JOINTS):
            raise ValueError('Joint order mismatch')
        self.calibration = self.config['calibration']
        self.calibration_sha256 = fingerprint(self.calibration)
        if self.config['calibration_sha256'] != self.calibration_sha256:
            raise ValueError('Calibration fingerprint mismatch')
        self.tick_min = np.array([self.calibration[j]['range_min'] for j in JOINTS], float)
        self.tick_max = np.array([self.calibration[j]['range_max'] for j in JOINTS], float)
        if np.any(self.tick_max <= self.tick_min):
            raise ValueError('Invalid calibration ranges')
        # This station has no inverted drive modes. Never silently infer the driver's setting.
        if any(self.calibration[j]['drive_mode'] != 0 for j in JOINTS):
            raise ValueError('Changed drive mode requires verified driver configuration')
        self.low = np.array([-100.] * 5 + [0.])
        self.high = np.full(6, 100.)

    @staticmethod
    def array(value):
        value = np.asarray(value, dtype=float)
        if value.ndim == 0 or value.shape[-1] != 6 or not np.isfinite(value).all():
            raise ValueError('Expected finite values with final dimension six')
        return value

    def continuous_ticks(self, normalized):
        value = self.array(normalized)
        return self.tick_min + (value-self.low)/(self.high-self.low)*(self.tick_max-self.tick_min)

    def to_encoder_ticks(self, normalized):
        value = self.array(normalized)
        if np.any(value < self.low) or np.any(value > self.high):
            raise ValueError('Normalized command outside station bounds')
        # LeRobot uses int(), NOT round(). Avoid adding homing_offset twice.
        return self.continuous_ticks(value).astype(np.int64)

    def from_encoder_ticks(self, ticks, *, clip=True):
        value = self.array(ticks)
        if clip:
            value = np.clip(value, self.tick_min, self.tick_max)
        return self.low + (value-self.tick_min)/(self.tick_max-self.tick_min)*(self.high-self.low)


class StationCoordinates(EncoderCoordinates):
    def __init__(self, model_limits, config=ROOT/'arm_config.json', mapping=None):
        super().__init__(config)
        if mapping is None:
            mapping = NOMINAL
        self.mapping = m = mapping if isinstance(mapping, dict) else json.loads(Path(mapping).read_text())
        self.model_limits = np.asarray(model_limits, float)
        if self.model_limits.shape != (6, 2) or not np.isfinite(self.model_limits).all():
            raise ValueError('Invalid model limits')
        try:
            sign = np.array([m['joints'][j]['sign'] for j in JOINTS], float)
            zero = np.array([m['joints'][j]['zero_tick'] for j in JOINTS], float)
            ticks_per_revolution = float(m['ticks_per_revolution'])
        except (KeyError, TypeError) as error:
            raise ValueError('Incomplete joint mapping') from error
        if not np.all(np.abs(sign) == 1) or not np.isfinite(zero).all() or ticks_per_revolution <= 0:
            raise ValueError('Invalid joint mapping')
        self.slope = sign*2*np.pi/ticks_per_revolution
        self.offset = -zero*self.slope
        self.mapping_sha256 = fingerprint(m)
        # Intersection, not stretching: keep original model and hardware limits.
        model_endpoints = np.stack((self.from_model(self.model_limits[:, 0]), self.from_model(self.model_limits[:, 1])))
        self.command_low = np.maximum(self.low, model_endpoints.min(axis=0))
        self.command_high = np.minimum(self.high, model_endpoints.max(axis=0))
        if np.any(self.command_low >= self.command_high):
            raise ValueError('Model and hardware have no common range')

    def to_model(self, normalized):
        value = self.array(normalized)
        if np.any(value < self.command_low-1e-4) or np.any(value > self.command_high+1e-4):
            raise ValueError('Command outside intersected model/hardware limits')
        return self.continuous_ticks(value)*self.slope+self.offset

    def from_model(self, radians):
        ticks = (self.array(radians)-self.offset)/self.slope
        return self.from_encoder_ticks(ticks, clip=False)

    def provenance(self):
        return dict(action_units=UNITS, state_units=UNITS, coordinate_mapping=self.mapping,
                    mapping_sha256=self.mapping_sha256, calibration_sha256=self.calibration_sha256,
                    joint_mapping_verified=self.mapping.get('status') == 'verified', hardware_compatible=False)


def constrain_simulation(env, coordinates):
    """Restrict expert targets to the common range without expanding any limit."""
    endpoints=np.stack((coordinates.to_model(coordinates.command_low), coordinates.to_model(coordinates.command_high)))
    env.limits=np.column_stack((endpoints.min(axis=0),endpoints.max(axis=0)))
