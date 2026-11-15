"""Validate physical-unit policy outputs before speed limiting; no hardware access.

Small extrapolations are projected inside calibrated normalized limits. Larger
ones and sustained requests into a boundary stop the trial. Calibration is never
modified. This does not certify collision-free motion or task success.
"""
import numpy as np

HARD_LOW = np.array([-100.0] * 5 + [0.0])
HARD_HIGH = np.array([100.0] * 6)
COMMAND_LOW = np.array([-99.0] * 5 + [1.0])
COMMAND_HIGH = np.array([99.0] * 6)


class PolicyActionBounds:
    def __init__(self):
        self.since = [None] * 6

    def prepare(self, actions):
        raw = np.asarray(actions, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != 6 or not len(raw) or not np.isfinite(raw).all():
            raise ValueError('Invalid policy action chunk')
        violation = np.maximum(np.maximum(HARD_LOW - raw, raw - HARD_HIGH), 0)
        if np.max(violation) > 5.0:
            raise ValueError('Policy exceeds calibrated limits by more than 5 normalized units')
        bounded = np.clip(raw, COMMAND_LOW, COMMAND_HIGH)
        return bounded, {
            'raw_min': raw.min(0).tolist(), 'raw_max': raw.max(0).tolist(),
            'hard_limit_overshoot': violation.max(0).tolist(),
            'projected_values': int(np.count_nonzero(raw != bounded)),
            'max_projection': float(np.abs(raw - bounded).max()),
        }

    def observe(self, measured, requested, now):
        measured, requested = np.asarray(measured), np.asarray(requested)
        if measured.shape != (6,) or requested.shape != (6,) or not np.isfinite(measured).all() or not np.isfinite(requested).all() or not np.isfinite(now):
            raise ValueError('Invalid boundary monitor input')
        if np.any(measured < HARD_LOW) or np.any(measured > HARD_HIGH):
            raise ValueError('Measured joints outside calibrated range')
        # Interior targets remain permitted to recover from a near-limit pose.
        seeking = ((measured <= COMMAND_LOW + .25) & (requested <= COMMAND_LOW)) | ((measured >= COMMAND_HIGH - .25) & (requested >= COMMAND_HIGH))
        for i, yes in enumerate(seeking):
            if not yes:
                self.since[i] = None
            elif self.since[i] is None:
                self.since[i] = now
            elif now - self.since[i] >= 2.0:
                raise ValueError(f'Persistent boundary request on joint {i}; holding')
