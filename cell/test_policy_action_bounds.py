import unittest
import numpy as np
from policy_action_bounds import PolicyActionBounds, COMMAND_LOW, COMMAND_HIGH


class BoundsTests(unittest.TestCase):
    def test_recorded_act_overshoot_projected_inward(self):
        raw = [[-7.5, -101.65, 94, 55, -51, 15]]
        bounded, report = PolicyActionBounds().prepare(raw)
        self.assertEqual(bounded[0, 1], -99)
        np.testing.assert_array_equal(bounded[0, [0, 2, 3, 4, 5]], np.array(raw)[0, [0, 2, 3, 4, 5]])
        self.assertAlmostEqual(report['hard_limit_overshoot'][1], 1.65)

    def test_nonfinite_and_large_errors_rejected(self):
        for v in [float('nan'), float('inf'), -106]:
            with self.assertRaises(ValueError):
                PolicyActionBounds().prepare([[0, v, 50, 20, 0, 10]])

    def test_boundary_dwell_stops(self):
        b = PolicyActionBounds(); state = [0, -99.4, 50, 20, 0, 10]
        b.observe(state, [0, -103, 50, 20, 0, 10], 0)
        with self.assertRaises(ValueError):
            b.observe(state, [0, -103, 50, 20, 0, 10], 2.01)

    def test_inward_recovery_resets_dwell(self):
        b = PolicyActionBounds(); state = [0, -99.4, 50, 20, 0, 10]
        b.observe(state, [0, -103, 50, 20, 0, 10], 0)
        b.observe(state, [0, -98, 50, 20, 0, 10], 2.1)
        b.observe(state, [0, -103, 50, 20, 0, 10], 2.2)

    def test_hardware_range_not_relaxed(self):
        with self.assertRaises(ValueError):
            PolicyActionBounds().observe([0, 0, 100.18, 0, 0, 10], [0]*6, 0)
        a,_=PolicyActionBounds().prepare([[-100]*5+[0], [100]*6])
        self.assertTrue(np.all(a >= COMMAND_LOW) and np.all(a <= COMMAND_HIGH))

if __name__ == '__main__': unittest.main()
