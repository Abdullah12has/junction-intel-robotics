import csv
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_mserver as sm  # noqa: E402

REPLY_KEYS = {"last_id", "status", "msg", "t", "target", "goal", "present", "err", "offset"}
POSE = [0.0, 30.0, -30.0, 40.0, 0.0, 25.0]


def make(**kw):
    os.makedirs(sm.UNIT_ROOT, exist_ok=True)
    root = tempfile.mkdtemp(prefix="t_", dir=sm.UNIT_ROOT)
    return sm.new_server(sm.load_params(), root, **kw)


def send(srv, obj):
    with open(srv.cmdf, "a") as f:
        f.write(json.dumps(obj) + "\n")
    srv.poll()
    with open(os.path.join(srv.D, "state.json")) as f:
        return json.load(f)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.srv = make()
        self.srv.state(0, "started")

    def test_started_reply_format(self):
        with open(os.path.join(self.srv.D, "state.json")) as f:
            st = json.load(f)
        self.assertEqual(set(st), REPLY_KEYS)
        self.assertEqual((st["last_id"], st["msg"], st["status"]), (0, "started", "ready"))
        for key in ("target", "goal", "present", "err", "offset"):
            self.assertEqual(len(st[key]), 6)

    def test_state_op(self):
        r = send(self.srv, {"id": 1, "op": "state"})
        self.assertEqual((r["last_id"], r["msg"], r["status"]), (1, "state", "ready"))

    def test_move_comp_converges(self):
        r = send(self.srv, {"id": 2, "op": "move", "q": POSE, "speed": 15, "comp": True})
        self.assertEqual((r["status"], r["last_id"]), ("ok", 2))
        self.assertTrue(r["msg"].startswith("comp_iters="))
        self.assertGreater(int(r["msg"].split("=")[1]), 0)
        self.assertLessEqual(np.max(np.abs(np.array(r["err"])[:5])), 0.5)

    def test_move_without_comp(self):
        r = send(self.srv, {"id": 3, "op": "move", "q": POSE, "speed": 15, "comp": False})
        self.assertEqual((r["msg"], r["status"]), ("comp_iters=0", "ok"))

    def test_grip_op_keeps_body(self):
        send(self.srv, {"id": 4, "op": "move", "q": POSE, "speed": 15, "comp": True})
        body = list(self.srv.target[:5])
        r = send(self.srv, {"id": 5, "op": "grip", "value": 0.0})
        self.assertEqual(r["msg"], "comp_iters=0")
        self.assertAlmostEqual(r["present"][5], 0.52, delta=0.03)
        self.assertEqual(list(np.round(self.srv.target[:5], 6)), list(np.round(body, 6)))

    def test_rel_op(self):
        send(self.srv, {"id": 6, "op": "move", "q": POSE, "speed": 15, "comp": False})
        before = self.srv.target.copy()
        r = send(self.srv, {"id": 7, "op": "rel", "dq": [5, 0, 0, 0, 0, 0], "speed": 15, "comp": False})
        self.assertEqual(r["msg"], "comp_iters=0")
        self.assertAlmostEqual(self.srv.target[0], before[0] + 5.0)

    def test_estop_refuses_move(self):
        open(self.srv.estop_path, "w").close()
        r = send(self.srv, {"id": 8, "op": "move", "q": POSE, "speed": 15, "comp": True})
        self.assertEqual((r["status"], r["msg"]), ("estop", "refused: estop present"))
        os.remove(self.srv.estop_path)
        r = send(self.srv, {"id": 9, "op": "move", "q": POSE, "speed": 15, "comp": False})
        self.assertEqual(r["status"], "ok")

    def test_unknown_op_keeps_status(self):
        send(self.srv, {"id": 10, "op": "move", "q": POSE, "speed": 15, "comp": False})
        r = send(self.srv, {"id": 11, "op": "frobnicate"})
        self.assertEqual((r["msg"], r["status"]), ("unknown op frobnicate", "ok"))

    def test_bad_target_is_error(self):
        r = send(self.srv, {"id": 12, "op": "move", "q": [0, 0, 0, 0, 0], "speed": 15})
        self.assertEqual(r["status"], "error")
        self.assertTrue(r["msg"].startswith("error: MotionError("))
        self.assertIn("expected 6 values", r["msg"])

    def test_rec_start_stop_csv(self):
        path = os.path.join(self.srv.D, "ep", "joints.csv")
        r = send(self.srv, {"id": 13, "op": "rec_start", "path": path})
        self.assertEqual(r["msg"], "rec_start " + path)
        send(self.srv, {"id": 14, "op": "move", "q": POSE, "speed": 15, "comp": False})
        r = send(self.srv, {"id": 15, "op": "rec_stop"})
        self.assertEqual(r["msg"], "rec_stop")
        with open(path) as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows[0]), 13)
        self.assertGreater(len(rows), 10)

    def test_exit_stops_server(self):
        r = send(self.srv, {"id": 16, "op": "exit"})
        self.assertEqual(r["msg"], "exit")
        self.assertTrue(self.srv.stop)


class PhysicsTests(unittest.TestCase):
    def test_empty_close_reads_closed_stop(self):
        srv = make()
        srv.move(POSE, 15.0, True)
        srv.move(POSE[:5] + [0.0], 20.0, False, accel=120.0, keep_offset=True)
        self.assertAlmostEqual(srv.world.reading()[5], 0.52, delta=0.02)

    def test_open_reads_25(self):
        srv = make()
        srv.move(POSE[:5] + [0.0], 20.0, False, accel=120.0, keep_offset=True)
        srv.move(POSE, 20.0, False, accel=120.0, keep_offset=True)
        self.assertAlmostEqual(srv.world.reading()[5], 25.0, delta=0.5)

    def test_weld_holds_brick_then_releases(self):
        srv = make(gravity=False, seed=3)
        srv.move(POSE, 15.0, True)
        sm.place_brick(srv.world, "brick_blue", srv.world.data.site_xpos[srv.world.site].copy())
        srv.move(POSE[:5] + [0.0], 20.0, False, accel=120.0, keep_offset=True)
        self.assertIsNotNone(srv.world.weld)
        self.assertEqual(srv.world.weld["brick"], "brick_blue")
        lo, hi = sm.HELD_RANGE
        self.assertTrue(lo <= srv.world.reading()[5] <= hi)
        srv.move(POSE, 20.0, False, accel=120.0, keep_offset=True)
        self.assertIsNone(srv.world.weld)
        self.assertAlmostEqual(srv.world.reading()[5], 25.0, delta=0.5)

    def test_no_brick_no_weld(self):
        srv = make(gravity=False)
        srv.move(POSE, 15.0, True)
        srv.move(POSE[:5] + [0.0], 20.0, False, accel=120.0, keep_offset=True)
        self.assertIsNone(srv.world.weld)

    def test_ideal_mode_has_no_sag(self):
        srv = make(dynamics=False)
        srv.move([0, 80, -60, 20, 0, 25], 15.0, False)
        self.assertLess(np.max(np.abs(srv.target - srv.world.reading())[:5]), 0.05)

    def test_compensation_reaches_tolerance_with_gravity(self):
        srv = make()
        srv.move([0, 80, -60, 20, 0, 25], 15.0, True)
        self.assertLessEqual(np.max(np.abs(srv.target - srv.world.reading())[:5]), 0.5)

    def test_ramp_contract(self):
        start = np.zeros(6)
        target = np.array([10, -20, 30, 5, -5, 25], dtype=float)
        rows = sm.plan_ramp(start, target, 15.0, sm.DT, max_accel=60.0)
        self.assertTrue(np.allclose(rows[-1], target))
        steps = np.diff(np.vstack([start, rows]), axis=0)
        self.assertLessEqual(np.max(np.abs(steps[:, :5])) / sm.DT, 15.0 * 1.02)
        self.assertEqual(sm.plan_ramp(target, target, 15.0, sm.DT).shape, (1, 6))

    def test_limits_and_gripper_law(self):
        with self.assertRaises(sm.MotionError):
            sm.validate_target([101, 0, 0, 0, 0, 25])
        with self.assertRaises(sm.MotionError):
            sm.validate_target([0, 0, 0, 0, 0, 101])
        q = sm.norm_to_joint_rad("gripper", 25.0)
        self.assertAlmostEqual(sm.joint_rad_to_norm("gripper", q), 25.0, places=9)
        self.assertAlmostEqual(sm.norm_to_joint_rad("gripper", 0.52), sm.GRIP_CLOSED_STOP, places=9)


if __name__ == "__main__":
    unittest.main()
