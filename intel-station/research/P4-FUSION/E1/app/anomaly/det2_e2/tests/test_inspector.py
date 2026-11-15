"""Synthetic tests for the E2 inspector (run: python3 -I -B tests/test_inspector.py; pytest-compatible names).

Covers shape cases, sliver fail-safe, empty mat, hands refusal, fail-safe routing, the red target lookup,
merged blobs (two hues, white plus colour) and close-view tape (textured dark band vs clean brick).
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import inspector as I  # noqa: E402


def mat(h=480, w=640):
    return np.full((h, w, 3), 35, np.uint8)  # black mat, V about 35 (below the 70 mat/brick threshold)


def rect(img, x, y, w, h, bgr):
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), bgr, -1)
    return img


ORANGE = (0, 140, 255)
RED = (0, 0, 220)
YELLOW = (0, 220, 240)
WHITE = (255, 255, 255)
THR = {"defect": 0.02, "good": 0.01}  # test thresholds (independent of models/tape_thresholds.json)


def test_2x2_is_2x2():
    img = rect(mat(), 300, 250, 30, 30, ORANGE)
    bricks = I.scan(img)
    assert len(bricks) == 1, bricks
    assert bricks[0]["shape"] == "2x2" and bricks[0]["shape_conf"] >= I.CONF_SHAPE
    assert bricks[0]["verdict"] == "UNCERTAIN"  # top view cannot confirm tape: never a GOOD route


def test_2x4_is_not_2x2():
    img = rect(mat(), 300, 250, 70, 32, ORANGE)
    bricks = I.scan(img)
    assert len(bricks) == 1, bricks
    assert bricks[0]["shape"] == "not_2x2"
    assert bricks[0]["verdict"] == "CONFIDENT_DEFECT"


def test_sliver_is_never_a_defect_call():
    img = rect(mat(), 250, 300, 120, 20, ORANGE)  # aspect 6: cable or fragment in the real top view
    bricks = I.scan(img)
    assert bricks and all(b["shape"] != "not_2x2" for b in bricks), bricks
    assert all(b["verdict"] != "CONFIDENT_DEFECT" for b in bricks)


def test_two_hue_blob_is_not_2x4():
    img = rect(mat(), 300, 250, 30, 30, RED)
    img = rect(img, 330, 250, 30, 30, YELLOW)  # two touching bricks of different colour: a 2x4 footprint
    bricks = I.scan(img)
    assert len(bricks) == 1, bricks
    assert bricks[0]["merged"] and bricks[0]["merged_reason"] == "two_hues", bricks[0]
    assert bricks[0]["shape"] == "unknown" and bricks[0]["verdict"] == "UNCERTAIN"


def test_white_plus_colour_blob_is_not_2x4():
    img = rect(mat(), 300, 250, 30, 30, WHITE)
    img = rect(img, 330, 250, 30, 30, ORANGE)
    bricks = I.scan(img)
    assert len(bricks) == 1, bricks
    assert bricks[0]["merged"] and bricks[0]["merged_reason"] == "white_plus_colour", bricks[0]
    assert bricks[0]["verdict"] != "CONFIDENT_DEFECT"


def test_empty_mat_gives_no_bricks():
    res = I.scan_full(mat())
    assert res["bricks"] == [] and res["hands"] == []


def test_hands_present_refuses():
    img = mat()
    cv2.ellipse(img, (300, 300), (40, 28), 0, 0, 360, (140, 170, 220), -1)  # skin-like hue, saturation 92
    try:
        I.scan(img)
    except I.HandsPresent:
        return
    raise AssertionError("hands were not refused")


def test_uncertain_never_routes_blue():
    img = rect(mat(), 200, 200, 30, 30, ORANGE)
    img = rect(img, 400, 300, 30, 30, RED)
    for b in I.scan(img):
        assert b["verdict"] in ("UNCERTAIN", "CONFIDENT_DEFECT")
        assert b["verdict"] != "CONFIDENT_GOOD"


def test_red_target_lookup():
    img = rect(mat(), 300, 250, 30, 30, RED)
    bricks = I.scan(img)
    tgt = I.find_target(bricks, colour="red")
    assert tgt is not None and tgt["colour"] == "red"
    assert I.find_target(bricks, colour="blue") is None


def _close_brick(with_tape):
    rng = np.random.default_rng(1)
    img = rect(mat(480, 640), 200, 200, 90, 60, ORANGE)
    if with_tape:
        g = rng.integers(50, 90, size=(14, 70, 1)).astype(np.uint8)  # grey, textured dark tape band
        img[235:249, 210:280] = np.repeat(g, 3, axis=2)
    return img


def test_close_tape_defect_vs_clean():
    tape = I.segment_close(_close_brick(True))["bricks"]
    clean = I.segment_close(_close_brick(False))["bricks"]
    assert len(tape) == 1 and len(clean) == 1, (len(tape), len(clean))
    assert tape[0]["tape_score"] >= THR["defect"], tape[0]["tape_score"]
    assert clean[0]["tape_score"] <= THR["good"], clean[0]["tape_score"]
    assert I.triage(tape[0], THR)[0] == "CONFIDENT_DEFECT"
    assert I.triage(clean[0], THR)[0] == "CONFIDENT_GOOD"
    assert I.triage(clean[0], None)[0] == "UNCERTAIN"  # no threshold file: never a confident call


def test_single_brick_contract():
    img = rect(mat(), 300, 250, 30, 30, ORANGE)
    out = I.Inspector(device="CPU").inspect_rgb(img)
    assert out["label"] in ("GOOD", "DEFECTIVE", "UNCERTAIN")
    assert isinstance(out["reasons"], list)
    box = I.Inspector().locate_brick(img)
    assert box is not None and len(box) == 4


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, type(e).__name__, str(e)[:200])
    print("failures", fails)
    sys.exit(1 if fails else 0)
