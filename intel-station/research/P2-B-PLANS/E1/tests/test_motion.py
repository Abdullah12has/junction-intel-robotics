"""Unit tests for app/motion. No hardware is touched.

The real SO-101 driver (physicalai_so101_plugin.so101.SO101, pinned plugin 0.3.0) runs
against FakeServoBus, which replaces the feetech scservo_sdk classes that the driver
imports (so101.py:38-43). The fake is installed unconditionally, so a real serial port
can never be opened from these tests.

Driver sources come only from P2R6_DRIVER_SRC (os.pathsep-separated, no default): the
physicalai core src/ and packages/physicalai-so101-plugin/src/ of
https://github.com/openvinotoolkit/physicalai.git rev 8381d439c2660d357b4b9d633f28ea7e7ed42c81.

Run from the folder that holds app/ and tests/:
  P2R6_DRIVER_SRC=<core src>:<plugin src> python3 -I -B -m pytest -p no:cacheprovider tests -q
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "app" / "motion"
EXAMPLE_POSES = APP / "poses.example.yaml"
FIXTURE = HERE / "fixtures" / "project_robots_sample.json"
DRIVER_SRC = [entry for entry in os.environ.get("P2R6_DRIVER_SRC", "").split(os.pathsep) if entry]
if not DRIVER_SRC or not all(Path(entry).is_dir() for entry in DRIVER_SRC):
    pytest.exit(
        "set P2R6_DRIVER_SRC to the physicalai src directory and packages/physicalai-so101-plugin/src "
        "of https://github.com/openvinotoolkit/physicalai.git rev 8381d439c2660d357b4b9d633f28ea7e7ed42c81",
        returncode=4,
    )
for entry in reversed(DRIVER_SRC):
    if entry not in sys.path:
        sys.path.insert(0, entry)
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


class FakeServoBus:
    """Per-servo present and goal positions, torque flags and an ordered event log."""

    def __init__(self, present: dict[int, int], *, max_step: int = 60) -> None:
        self.present = dict(present)
        self.goal = dict(present)
        self.torque = dict.fromkeys(present, 0)
        self.registers: dict[tuple[int, int], int] = {}
        self.events: list[tuple[object, ...]] = []
        self.open_ports: set[str] = set()
        self.max_step = max_step

    def set_goal(self, servo_id: int, value: int) -> None:
        self.goal[servo_id] = value
        self.events.append(("goal", servo_id, value))
        if self.torque[servo_id]:
            step = max(-self.max_step, min(self.max_step, value - self.present[servo_id]))
            self.present[servo_id] += step

    def set_torque(self, servo_id: int, value: int) -> None:
        self.torque[servo_id] = value
        self.events.append(("torque", servo_id, value))


ACTIVE: dict[str, FakeServoBus | None] = {"bus": None}


def _bus() -> FakeServoBus:
    bus = ACTIVE["bus"]
    if bus is None:
        msg = "no FakeServoBus installed for this test"
        raise RuntimeError(msg)
    return bus


# Fake Feetech SDK. It is installed before the driver module is first imported, and it
# always replaces any real scservo_sdk, so no real serial port can be opened here.
class FakePortHandler:
    def __init__(self, port: str) -> None:
        self.port = port

    def openPort(self) -> bool:  # noqa: N802 (SDK naming)
        _bus().open_ports.add(self.port)
        _bus().events.append(("open", self.port))
        return True

    def setPacketTimeoutMillis(self, _ms: float) -> None:  # noqa: N802
        return None

    def setBaudRate(self, baud: int) -> bool:  # noqa: N802
        _bus().events.append(("baud", baud))
        return True

    def closePort(self) -> None:  # noqa: N802
        _bus().open_ports.discard(self.port)
        _bus().events.append(("close", self.port))


class FakePacketHandler:
    def __init__(self, _protocol: int) -> None:
        return None

    def ping(self, _port: object, servo_id: int) -> tuple[int, int, int]:
        return (0, 0 if servo_id in _bus().present else -1, 0)

    def write1ByteTxRx(self, _port: object, servo_id: int, addr: int, value: int) -> tuple[int, int]:  # noqa: N802
        if int(addr) == 40:  # TORQUE_ENABLE, so101/constants.py
            _bus().set_torque(servo_id, value)
        else:
            _bus().registers[(servo_id, int(addr))] = value
        return (0, 0)

    def write2ByteTxRx(self, _port: object, servo_id: int, addr: int, value: int) -> tuple[int, int]:  # noqa: N802
        _bus().registers[(servo_id, int(addr))] = value
        return (0, 0)


class FakeGroupSyncRead:
    def __init__(self, _port: object, _packet: object, _addr: int, _length: int) -> None:
        self.ids: list[int] = []

    def addParam(self, servo_id: int) -> bool:  # noqa: N802
        self.ids.append(servo_id)
        return True

    def txRxPacket(self) -> int:  # noqa: N802
        return 0

    def isAvailable(self, servo_id: int, _addr: int, _length: int) -> bool:  # noqa: N802
        return servo_id in self.ids

    def getData(self, servo_id: int, _addr: int, _length: int) -> int:  # noqa: N802
        return _bus().present[servo_id]


class FakeGroupSyncWrite:
    def __init__(self, _port: object, _packet: object, _addr: int, _length: int) -> None:
        self.pending: dict[int, int] = {}

    def clearParam(self) -> None:  # noqa: N802
        self.pending = {}

    def addParam(self, servo_id: int, data: list[int]) -> bool:  # noqa: N802
        self.pending[servo_id] = data[0] + (data[1] << 8)
        return True

    def txPacket(self) -> int:  # noqa: N802
        for servo_id, value in self.pending.items():
            _bus().set_goal(servo_id, value)
        return 0


_fake_sdk = types.ModuleType("scservo_sdk")
_fake_sdk.PortHandler = FakePortHandler  # type: ignore[attr-defined]
_fake_sdk.PacketHandler = FakePacketHandler  # type: ignore[attr-defined]
_fake_sdk.GroupSyncRead = FakeGroupSyncRead  # type: ignore[attr-defined]
_fake_sdk.GroupSyncWrite = FakeGroupSyncWrite  # type: ignore[attr-defined]
sys.modules["scservo_sdk"] = _fake_sdk

import run_sequence  # noqa: E402
import so101_motion  # noqa: E402
import teach_poses  # noqa: E402
from physicalai.robot.transport._lock import acquire_locks  # noqa: E402
from so101_motion import (  # noqa: E402
    DEFAULT_MAX_ACCEL,
    HARD_MAX_SPEED,
    DeviceOwnedError,
    MotionError,
    MotionPlayer,
    MotionStalledError,
    Step,
    build_driver,
    claim_devices,
    describe_plan,
    device_ids_for,
    hold_and_enable_torque,
    is_placeholder,
    load_poses,
    load_sequences,
    load_studio_robot,
    open_follower_held,
    open_passive,
    plan_ramp,
    resolve_arm,
    validate_target,
    write_pose,
)

SERVO_IDS = [1, 2, 3, 4, 5, 6]  # calibration ids of the follower export (fixture)
START_TICKS = [2051, 1956, 2000, 1991, 1986, 2811]  # inside every recorded range of the follower
FOLLOWER_PORT = "/dev/serial/by-id/usb-TEST-ONLY_no_such_device-if00"  # fixture path, never opened (the fake bus takes it)
REAL_FOLLOWER_PORT_BY_ID = so101_motion.FOLLOWER_PORT_BY_ID  # read before the autouse fixture swaps it
FOLLOWER_SERIAL = "5B3D040939"  # project_robots_sample.json, SO101 Follower
TAUGHT_TEXT = (
    "unit: normalized\n"
    "poses:\n"
    "  rest: [0, 0, 0, 0, 0, 0]\n"
    "  up: [0, -10, 10, 0, 0, 20]\n"
    "sequences:\n"
    "  go:\n"
    "    - {pose: up, speed: 50}\n"
    "    - {pose: rest, speed: 50}\n"
)


@pytest.fixture(autouse=True)
def _no_bus_left_behind() -> Iterator[None]:
    yield
    ACTIVE["bus"] = None


@pytest.fixture(autouse=True)
def _follower_is_a_test_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the follower's by-id path at the non-existent test path and keep locks out of the live runtime dir."""
    monkeypatch.setattr(so101_motion, "FOLLOWER_PORT_BY_ID", FOLLOWER_PORT)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))


def install_bus(present: list[int] = START_TICKS, *, max_step: int = 60) -> FakeServoBus:
    bus = FakeServoBus(dict(zip(SERVO_IDS, present, strict=True)), max_step=max_step)
    ACTIVE["bus"] = bus
    return bus


def follower_calibration():  # noqa: ANN201
    return load_studio_robot(FIXTURE, "SO101 Follower").calibration


def use_follower_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test host has no USB enumeration of the arm, so the test supplies the serial number."""
    monkeypatch.setattr(so101_motion, "port_serial_number", lambda _port: FOLLOWER_SERIAL)


def write_taught(tmp_path: Path) -> Path:
    taught = tmp_path / "taught.yaml"
    taught.write_text(TAUGHT_TEXT, encoding="utf-8")
    return taught


class StallDriver:
    """Stand-in with the attributes MotionPlayer reads. Records when each setpoint is sent."""

    max_speed = 278.0

    def __init__(self, stall_s: float = 0.0) -> None:
        self.stall_s = stall_s
        self.sent: list[float] = []

    def send_action(self, _action: np.ndarray) -> None:
        self.sent.append(time.monotonic())
        if len(self.sent) == 1 and self.stall_s > 0:
            time.sleep(self.stall_s)


def _spawn_port_holder(path: Path) -> subprocess.Popen[str]:
    """Start a process of this user that keeps path open, like a Studio session holding the port."""
    code = "import sys, time\nhandle = open(sys.argv[1], 'rb')\nprint('ready', flush=True)\ntime.sleep(120)\n"
    child = subprocess.Popen([sys.executable, "-I", "-c", code, str(path)], stdout=subprocess.PIPE, text=True)
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "ready"
    return child


def _stop(child: subprocess.Popen[str]) -> None:
    child.kill()
    child.wait(timeout=10)
    if child.stdout is not None:
        child.stdout.close()


def test_plan_ramp_never_exceeds_max_speed_and_ends_on_target() -> None:
    rng = np.random.default_rng(0)
    speed, dt = 25.0, 1.0 / 30.0
    for _ in range(200):
        start = rng.uniform(-100, 100, 6)
        start[5] = rng.uniform(0, 100)
        target = rng.uniform(-100, 100, 6)
        target[5] = rng.uniform(0, 100)
        path = np.vstack([start, plan_ramp(start, target, speed, dt)])
        rates = np.abs(np.diff(path, axis=0)) / dt
        assert rates.max() <= speed * (1 + 1e-9)
        assert np.array_equal(path[-1], target)


def test_plan_ramp_respects_the_acceleration_limit_for_random_moves() -> None:
    rng = np.random.default_rng(1)
    speed, dt = 50.0, 1.0 / 30.0
    for _ in range(200):
        start = rng.uniform(-100, 100, 6)
        start[5] = rng.uniform(0, 100)
        target = rng.uniform(-100, 100, 6)
        target[5] = rng.uniform(0, 100)
        delta = float(np.max(np.abs(target - start)))
        setpoints = plan_ramp(start, target, speed, dt, max_accel=DEFAULT_MAX_ACCEL)
        assert len(setpoints) * dt >= np.sqrt(6.0 * delta / DEFAULT_MAX_ACCEL) - 1e-9
        path = np.vstack([start, setpoints])
        peak_accel = float(np.max(np.abs(np.diff(path, n=2, axis=0)))) / dt**2
        assert peak_accel <= DEFAULT_MAX_ACCEL * (1 + 1e-9)


def test_plan_ramp_zero_move_is_one_setpoint_and_bad_inputs_raise() -> None:
    pose = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert plan_ramp(pose, pose, 10.0, 0.1).shape == (1, 6)
    with pytest.raises(MotionError):
        plan_ramp(pose, pose + 1.0, 0.0, 0.1)
    with pytest.raises(MotionError):
        plan_ramp(pose, np.zeros(5), 10.0, 0.1)


def test_non_finite_limits_are_refused_by_plan_ramp_and_player() -> None:
    pose = np.zeros(6)
    for bad in (float("nan"), float("inf"), 0.0):
        with pytest.raises(MotionError):
            plan_ramp(pose, pose + 1.0, bad, 0.1)
    with pytest.raises(MotionError):
        plan_ramp(pose, pose + 1.0, 10.0, float("nan"))
    stub = StallDriver()
    for kwargs in ({"max_speed": float("nan")}, {"rate_hz": float("nan")}, {"max_accel": float("nan")}):
        with pytest.raises(MotionError):
            MotionPlayer(stub, **kwargs)


@pytest.mark.parametrize(
    "bad",
    [
        [0, 0, 0, 0, 100.5, 0],
        [0, 0, 0, 0, 0, -0.1],
        [0, 0, 0, 0, 0, 100.1],
        [0, 0, 0, float("nan"), 0, 0],
        [0, 0, 0, 0, 0],
    ],
)
def test_validate_target_rejects_out_of_limit_values(bad: list[float]) -> None:
    with pytest.raises(MotionError):
        validate_target(bad)


def test_example_poses_and_sequences_load_and_reference_known_poses() -> None:
    poses = load_poses(EXAMPLE_POSES)
    sequences = load_sequences(EXAMPLE_POSES)
    assert set(poses) == {"rest", "pick_spot", "inspection_spot", "blue_plate", "green_plate"}
    assert all(value.shape == (6,) for value in poses.values())
    assert set(sequences) == {"pick_to_inspection", "inspection_to_blue", "inspection_to_green"}
    for steps in sequences.values():
        assert all(step.pose in poses for step in steps)


def test_write_pose_roundtrip_and_rejects_invalid_without_touching_file(tmp_path: Path) -> None:
    out = tmp_path / "poses.yaml"
    write_pose(out, "rest", [0.0, -60.0, 60.0, 0.0, 0.0, 0.0])
    write_pose(out, "pick", [10.0, 20.0, -30.0, 5.0, 0.0, 99.0])
    before = out.read_text(encoding="utf-8")
    loaded = load_poses(out)
    assert np.allclose(loaded["pick"], [10.0, 20.0, -30.0, 5.0, 0.0, 99.0])
    with pytest.raises(MotionError):
        write_pose(out, "bad", [0.0, 0.0, 0.0, 0.0, 0.0, 120.0])
    assert out.read_text(encoding="utf-8") == before


def test_load_studio_robot_reads_port_unit_serial_and_calibration() -> None:
    follower = load_studio_robot(FIXTURE, "SO101 Follower")
    leader = load_studio_robot(FIXTURE, "SO101 Leader")
    assert follower.port == "/dev/ttyACM1"
    assert leader.port == "/dev/ttyACM2"
    assert follower.serial_number == FOLLOWER_SERIAL
    assert leader.serial_number == "5B90148378"
    assert follower.unit == "normalized"
    assert sorted(follower.calibration.joints) == sorted(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"])
    assert follower.calibration.joints["gripper"].range_max == 3577
    with pytest.raises(MotionError):
        load_studio_robot(FIXTURE, "SO101 Missing")


def test_resolve_arm_refuses_a_port_whose_serial_differs_from_the_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(so101_motion, "port_serial_number", lambda _port: "WRONG-SERIAL")
    with pytest.raises(MotionError, match="reports serial number"):
        resolve_arm(studio_export=FIXTURE, name="SO101 Follower", follower=True)
    use_follower_serial(monkeypatch)
    port, calibration = resolve_arm(studio_export=FIXTURE, name="SO101 Follower", follower=True)
    assert port == FOLLOWER_PORT
    assert calibration.joints["gripper"].range_max == 3577


def test_device_ids_match_the_real_driver_property() -> None:
    install_bus()
    driver = build_driver(FOLLOWER_PORT, follower_calibration(), role="follower")
    assert driver.device_ids == device_ids_for(FOLLOWER_PORT) == ("serial:usb-TEST-ONLY_no_such_device-if00",)


def test_the_follower_by_id_path_is_the_binding_rule() -> None:
    assert REAL_FOLLOWER_PORT_BY_ID == "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00"


def test_follower_port_is_the_by_id_path_and_the_export_connection_string_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    use_follower_serial(monkeypatch)
    port, _calibration = resolve_arm(studio_export=FIXTURE, name="SO101 Follower", follower=True)
    assert port == FOLLOWER_PORT
    assert port != "/dev/ttyACM1"
    with pytest.raises(MotionError, match="follower port is"):
        resolve_arm(studio_export=FIXTURE, name="SO101 Follower", port="/dev/ttyACM1", follower=True)


def test_device_ids_resolve_a_by_id_link_to_its_device(tmp_path: Path) -> None:
    target = tmp_path / "ttyFAKE9"
    target.write_bytes(b"")
    link = tmp_path / "usb-TEST-by-id-if00"
    link.symlink_to(target)
    assert device_ids_for(str(link)) == ("serial:ttyFAKE9",)


def test_claim_refuses_a_by_id_link_whose_device_another_process_has_open(tmp_path: Path) -> None:
    target = tmp_path / "ttyFAKE10"
    target.write_bytes(b"")
    link = tmp_path / "usb-TEST-by-id-if00"
    link.symlink_to(target)
    child = _spawn_port_holder(target)
    try:
        with pytest.raises(DeviceOwnedError, match="open in process"), claim_devices(device_ids_for(str(link)), ports=(str(link),)):
            pass
    finally:
        _stop(child)


def test_open_passive_keeps_torque_off() -> None:
    bus = install_bus()
    driver = open_passive(FOLLOWER_PORT, follower_calibration())
    assert driver.is_connected()
    assert not any(event[0] == "torque" and event[2] == 1 for event in bus.events)
    driver.disconnect()
    assert FOLLOWER_PORT not in bus.open_ports


def test_open_passive_turns_torque_off_when_it_was_on() -> None:
    bus = install_bus()
    bus.torque = dict.fromkeys(SERVO_IDS, 1)
    driver = open_passive(FOLLOWER_PORT, follower_calibration())
    try:
        assert list(bus.torque.values()) == [0] * len(SERVO_IDS)
    finally:
        driver.disconnect()


def test_open_follower_held_writes_present_goal_before_torque_enable() -> None:
    bus = install_bus()
    driver = open_follower_held(FOLLOWER_PORT, follower_calibration())
    first_torque_on = next(i for i, e in enumerate(bus.events) if e[0] == "torque" and e[2] == 1)
    goals_before = [e for e in bus.events[:first_torque_on] if e[0] == "goal"]
    assert len(goals_before) == len(SERVO_IDS)
    assert [value for _, _, value in goals_before] == START_TICKS
    assert driver.role == "follower"
    assert driver.torque_on_disconnect is True
    driver.disconnect()


def test_hold_refuses_an_out_of_range_present_pose_and_keeps_torque_off() -> None:
    present = list(START_TICKS)
    present[5] = 3700  # gripper range_max is 3577
    bus = install_bus(present)
    driver = open_passive(FOLLOWER_PORT, follower_calibration())
    try:
        with pytest.raises(MotionError, match="gripper"):
            hold_and_enable_torque(driver)
        assert not any(e[0] == "torque" and e[2] == 1 for e in bus.events)
        assert not any(e[0] == "goal" for e in bus.events)
        assert driver.role == "leader"
    finally:
        driver.disconnect()
    assert FOLLOWER_PORT not in bus.open_ports


def test_open_follower_held_closes_the_port_when_a_joint_is_out_of_range() -> None:
    present = list(START_TICKS)
    present[5] = 3700
    bus = install_bus(present)
    with pytest.raises(MotionError):
        open_follower_held(FOLLOWER_PORT, follower_calibration())
    assert FOLLOWER_PORT not in bus.open_ports
    assert not any(e[0] == "torque" and e[2] == 1 for e in bus.events)


def test_open_follower_held_closes_the_port_when_the_hold_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = install_bus()

    def failing_hold(_driver: object) -> None:
        msg = "bus fault"
        raise RuntimeError(msg)

    monkeypatch.setattr(so101_motion, "hold_and_enable_torque", failing_hold)
    with pytest.raises(RuntimeError, match="bus fault"):
        open_follower_held(FOLLOWER_PORT, follower_calibration())
    assert FOLLOWER_PORT not in bus.open_ports


def test_move_to_reaches_target_within_speed_limit() -> None:
    install_bus(max_step=40)
    driver = open_follower_held(FOLLOWER_PORT, follower_calibration())
    try:
        player = MotionPlayer(driver, max_speed=200.0, rate_hz=200.0, tolerance=2.0, settle_s=0.02)
        assert player.max_speed <= driver.max_speed
        target = np.array([30.0, -20.0, 10.0, 5.0, -10.0, 60.0])
        result = player.move_to(target)
        assert result.arrived
        assert result.max_error <= 2.0
        assert np.allclose(player.current(), target, atol=2.0)
    finally:
        driver.disconnect()


def test_motion_player_caps_speed_at_the_hard_ceiling() -> None:
    install_bus()
    driver = open_passive(FOLLOWER_PORT, follower_calibration())
    try:
        assert driver.max_speed > HARD_MAX_SPEED
        assert MotionPlayer(driver, max_speed=1000.0).max_speed == HARD_MAX_SPEED
        assert MotionPlayer(driver, max_speed=10.0).max_speed == 10.0
    finally:
        driver.disconnect()


def test_keyboard_interrupt_during_move_holds_the_present_pose() -> None:
    bus = install_bus(max_step=40)
    driver = open_follower_held(FOLLOWER_PORT, follower_calibration())
    try:
        player = MotionPlayer(driver, max_speed=50.0, rate_hz=200.0, tolerance=2.0, settle_s=0.02)
        real_send = driver.send_action
        sent = {"n": 0}

        def interrupted(action: np.ndarray) -> None:
            sent["n"] += 1
            if sent["n"] == 2:
                raise KeyboardInterrupt
            real_send(action)

        driver.send_action = interrupted  # type: ignore[method-assign]
        with pytest.raises(KeyboardInterrupt):
            player.move_to(np.array([30.0, -20.0, 10.0, 5.0, -10.0, 60.0]))
        last_goal: dict[int, int] = {}
        for event in bus.events:
            if event[0] == "goal":
                last_goal[int(event[1])] = int(event[2])
        assert [last_goal[servo_id] for servo_id in SERVO_IDS] == [bus.present[servo_id] for servo_id in SERVO_IDS]
        assert all(bus.torque[servo_id] == 1 for servo_id in SERVO_IDS)
    finally:
        driver.disconnect()


def test_pacing_does_not_burst_setpoints_after_a_stall() -> None:
    driver = StallDriver(stall_s=0.1)
    player = MotionPlayer(driver, max_speed=50.0, rate_hz=50.0, settle_s=0.0)
    player._send_paced(np.zeros((10, 6)))
    gaps = np.diff(driver.sent[1:])
    assert len(driver.sent) == 10
    assert gaps.min() >= 0.5 * player.dt


def test_move_gripper_changes_only_the_gripper() -> None:
    install_bus(max_step=40)
    driver = open_follower_held(FOLLOWER_PORT, follower_calibration())
    try:
        player = MotionPlayer(driver, max_speed=200.0, rate_hz=200.0, tolerance=2.0, settle_s=0.02)
        before = player.current()
        player.move_gripper(10.0)
        after = player.current()
        assert abs(after[5] - 10.0) <= 2.0
        assert np.allclose(after[:5], before[:5], atol=2.0)
    finally:
        driver.disconnect()


def test_move_rejects_out_of_limit_target_without_sending_anything() -> None:
    bus = install_bus()
    driver = open_follower_held(FOLLOWER_PORT, follower_calibration())
    try:
        player = MotionPlayer(driver)
        events_before = len(bus.events)
        with pytest.raises(MotionError):
            player.move_to([0.0, 0.0, 0.0, 0.0, 0.0, 150.0])
        assert len(bus.events) == events_before
    finally:
        driver.disconnect()


def test_disconnect_holds_follower_and_closes_port() -> None:
    bus = install_bus()
    driver = open_follower_held(FOLLOWER_PORT, follower_calibration())
    driver.disconnect()
    assert FOLLOWER_PORT not in bus.open_ports
    torque_events = [e for e in bus.events if e[0] == "torque"]
    assert torque_events[-1][2] == 1
    assert not driver.is_connected()


def test_claim_devices_reports_conflict_and_releases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    holder = acquire_locks("studio-owner", ("serial:ttyFAKE9",))
    try:
        with pytest.raises(DeviceOwnedError, match="serial:ttyFAKE9"), claim_devices(("serial:ttyFAKE9",), owner="motion-test"):
            pass
    finally:
        holder.release_all()
    with claim_devices(("serial:ttyFAKE9",), owner="motion-test"):
        pass


def test_claim_devices_nests_inside_a_claim_of_the_same_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with claim_devices(("serial:ttyFAKE5",), owner="outer"), claim_devices(("serial:ttyFAKE5",), owner="inner"):
        pass
    with claim_devices(("serial:ttyFAKE5",), owner="after"):
        pass


def test_claim_refuses_a_port_open_in_another_process_whatever_the_runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "other-runtime"))
    port = tmp_path / "ttyFAKE7"
    port.write_bytes(b"")
    child = _spawn_port_holder(port)
    try:
        with pytest.raises(DeviceOwnedError, match="open in process"), claim_devices(("serial:ttyFAKE7",), ports=(str(port),)):
            pass
    finally:
        _stop(child)
    with claim_devices(("serial:ttyFAKE7",), ports=(str(port),)):
        pass


def test_open_passive_refuses_a_port_another_process_has_open(tmp_path: Path) -> None:
    port = tmp_path / "ttyFAKE8"
    port.write_bytes(b"")
    child = _spawn_port_holder(port)
    try:
        with pytest.raises(DeviceOwnedError):
            open_passive(str(port), follower_calibration())
    finally:
        _stop(child)


def test_describe_plan_stays_under_step_speed() -> None:
    poses = load_poses(EXAMPLE_POSES)
    steps = load_sequences(EXAMPLE_POSES)["pick_to_inspection"]
    rows = describe_plan(steps, poses, start=poses["rest"], max_speed=25.0, rate_hz=30.0)
    assert len(rows) == len(steps)
    for row in rows:
        assert row["peak_rate_units_s"] <= row["speed"] * (1 + 1e-6)
    assert rows[0]["setpoints"] >= 1


def test_describe_plan_applies_the_speed_ceiling_and_reports_acceleration() -> None:
    poses = load_poses(EXAMPLE_POSES)
    steps = load_sequences(EXAMPLE_POSES)["pick_to_inspection"]
    rows = describe_plan(steps, poses, start=poses["rest"], max_speed=1000.0, rate_hz=30.0)
    assert all(row["speed"] <= HARD_MAX_SPEED for row in rows)
    assert all(row["peak_accel_units_s2"] <= DEFAULT_MAX_ACCEL * (1 + 1e-6) for row in rows)
    assert all(row["peak_rate_units_s"] <= row["speed"] * (1 + 1e-6) for row in rows)


@pytest.mark.parametrize("speed", ["51", ".nan", "0", "-1"])
def test_load_sequences_rejects_speeds_outside_the_ceiling(tmp_path: Path, speed: str) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        f"unit: normalized\nposes:\n  rest: [0, 0, 0, 0, 0, 0]\nsequences:\n  s:\n    - {{pose: rest, speed: {speed}}}\n",
        encoding="utf-8",
    )
    with pytest.raises(MotionError, match="speed"):
        load_sequences(bad)


def test_run_sequence_dry_run_opens_no_port(capsys: pytest.CaptureFixture[str]) -> None:
    bus = install_bus()
    assert run_sequence.main(["--poses", str(EXAMPLE_POSES), "--sequence", "pick_to_inspection"]) == 0
    assert bus.events == []
    assert bus.open_ports == set()
    assert "dry run" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["51", "nan", "0"])
def test_run_sequence_refuses_a_max_speed_outside_the_ceiling(value: str) -> None:
    assert run_sequence.main(["--poses", str(EXAMPLE_POSES), "--sequence", "pick_to_inspection", "--max-speed", value]) == 4


def test_run_sequence_unknown_sequence_is_config_error() -> None:
    assert run_sequence.main(["--poses", str(EXAMPLE_POSES), "--sequence", "nope"]) == 4


def test_run_sequence_returns_3_before_opening_when_the_device_is_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    use_follower_serial(monkeypatch)
    bus = install_bus()
    taught = write_taught(tmp_path)
    holder = acquire_locks("studio-owner", device_ids_for(FOLLOWER_PORT))
    try:
        code = run_sequence.main(["--poses", str(taught), "--sequence", "go", "--studio-export", str(FIXTURE), "--confirm-motion"])
    finally:
        holder.release_all()
    assert code == 3
    assert bus.events == []
    assert bus.open_ports == set()


def test_run_sequence_confirm_motion_on_taught_poses_returns_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    use_follower_serial(monkeypatch)
    bus = install_bus(max_step=60)
    taught = write_taught(tmp_path)
    code = run_sequence.main(["--poses", str(taught), "--sequence", "go", "--studio-export", str(FIXTURE), "--confirm-motion", "--rate-hz", "200"])
    assert code == 0
    assert bus.open_ports == set()
    assert any(e[0] == "goal" for e in bus.events)


def test_teach_loop_saves_valid_names_and_quits(tmp_path: Path) -> None:
    out = tmp_path / "taught.yaml"
    answers = iter(["rest", "?", "bad name!", "q"])
    pose = np.array([0.0, -60.0, 60.0, 0.0, 0.0, 0.0])
    saved = teach_poses.teach_loop(
        lambda: pose,
        out,
        input_fn=lambda _prompt: next(answers),
        print_fn=lambda _line: None,
    )
    assert saved == ["rest"]
    assert np.allclose(load_poses(out)["rest"], pose)


def test_run_executes_steps_in_order_and_reports_each() -> None:
    install_bus(max_step=40)
    driver = open_follower_held(FOLLOWER_PORT, follower_calibration())
    try:
        player = MotionPlayer(driver, max_speed=200.0, rate_hz=200.0, tolerance=2.0, settle_s=0.02)
        poses = {
            "a": np.array([10.0, 0.0, 0.0, 0.0, 0.0, 20.0]),
            "b": np.array([-10.0, 5.0, 0.0, 0.0, 0.0, 40.0]),
        }
        steps = [Step(pose="a"), Step(pose="b", dwell_s=0.01)]
        results = player.run(steps, poses)
        assert len(results) == 2
        assert all(result.arrived for result in results)
        assert np.allclose(player.current(), poses["b"], atol=2.0)
    finally:
        driver.disconnect()


def test_load_sequences_rejects_unknown_pose_names(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "unit: normalized\nposes:\n  rest: [0, 0, 0, 0, 0, 0]\nsequences:\n  s:\n    - {pose: missing}\n",
        encoding="utf-8",
    )
    with pytest.raises(MotionError, match="unknown pose"):
        load_sequences(bad)


def test_stalled_step_stops_the_sequence_before_later_steps() -> None:
    install_bus(max_step=1)
    driver = open_follower_held(FOLLOWER_PORT, follower_calibration())
    try:
        start = driver.get_observation().joint_positions.astype(np.float64)
        near = start.copy()
        near[0] += 20.0
        far = start.copy()
        far[0] -= 20.0
        player = MotionPlayer(driver, max_speed=100.0, rate_hz=30.0, tolerance=2.0, settle_s=0.05)
        with pytest.raises(MotionStalledError, match="step 1") as info:
            player.run([Step(pose="near"), Step(pose="far")], {"near": near, "far": far})
        assert len(info.value.results) == 1
        assert not info.value.results[0].arrived
    finally:
        driver.disconnect()


def test_allow_stall_step_lets_the_sequence_continue_to_the_next_step() -> None:
    install_bus(max_step=1)
    driver = open_follower_held(FOLLOWER_PORT, follower_calibration())
    try:
        start = driver.get_observation().joint_positions.astype(np.float64)
        near = start.copy()
        near[0] += 20.0
        far = start.copy()
        far[0] -= 20.0
        player = MotionPlayer(driver, max_speed=100.0, rate_hz=30.0, tolerance=2.0, settle_s=0.05)
        steps = [Step(pose="near", allow_stall=True), Step(pose="far")]
        with pytest.raises(MotionStalledError, match="step 2") as info:
            player.run(steps, {"near": near, "far": far})
        assert len(info.value.results) == 2
    finally:
        driver.disconnect()


def test_load_sequences_reads_allow_stall_and_rejects_non_boolean(tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    good.write_text(
        "unit: normalized\nposes:\n  rest: [0, 0, 0, 0, 0, 0]\n"
        "sequences:\n  s:\n    - {pose: rest, allow_stall: true}\n    - {pose: rest}\n",
        encoding="utf-8",
    )
    assert [step.allow_stall for step in load_sequences(good)["s"]] == [True, False]
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "unit: normalized\nposes:\n  rest: [0, 0, 0, 0, 0, 0]\nsequences:\n  s:\n    - {pose: rest, allow_stall: yes please}\n",
        encoding="utf-8",
    )
    with pytest.raises(MotionError, match="allow_stall"):
        load_sequences(bad)


def test_placeholder_poses_refuse_motion_before_any_port_is_opened() -> None:
    bus = install_bus()
    assert is_placeholder(EXAMPLE_POSES) is True
    code = run_sequence.main(
        ["--poses", str(EXAMPLE_POSES), "--sequence", "pick_to_inspection", "--studio-export", str(FIXTURE), "--confirm-motion"]
    )
    assert code == 4
    assert bus.events == []
    assert bus.open_ports == set()


def test_is_placeholder_is_false_when_the_flag_is_absent(tmp_path: Path) -> None:
    taught = tmp_path / "taught.yaml"
    taught.write_text("unit: normalized\nposes:\n  rest: [0, 0, 0, 0, 0, 0]\n", encoding="utf-8")
    assert is_placeholder(taught) is False


def test_hold_and_enable_torque_writes_the_goal_before_torque_on() -> None:
    bus = install_bus()
    driver = open_passive(FOLLOWER_PORT, follower_calibration())
    try:
        hold_and_enable_torque(driver)
        first_torque_on = next(i for i, e in enumerate(bus.events) if e[0] == "torque" and e[2] == 1)
        goals_before = [e for e in bus.events[:first_torque_on] if e[0] == "goal"]
        assert len(goals_before) == len(SERVO_IDS)
        assert driver.role == "follower"
        assert driver.torque_on_disconnect is True
    finally:
        driver.disconnect()


def test_hand_teaching_exit_leaves_the_arm_held_with_torque_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    bus = install_bus()
    args = argparse.Namespace(out=tmp_path / "poses.yaml", mode="hand", baudrate=1_000_000, capture="rest")
    assert teach_poses.run_teaching(args, FOLLOWER_PORT, follower_calibration()) == 0
    assert list(bus.torque.values()) == [1] * len(SERVO_IDS)
    assert FOLLOWER_PORT not in bus.open_ports
    assert "rest" in load_poses(args.out)
