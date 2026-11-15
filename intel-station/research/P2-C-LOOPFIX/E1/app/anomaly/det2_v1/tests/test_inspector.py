"""Synthetic tests for the inspector (run: python3 -I -B tests/test_inspector.py; pytest-compatible names).

E2 cases kept: shape cases, sliver fail-safe, empty mat, red target, close tape, single-brick contract.
V1 changes: two-colour blobs are split into colour pieces (no 2x4 call); a hand needs an in-mat skin blob that touches
the border or is large and not brick-shaped; pink bricks never refuse; an elongated close blob is never CONFIDENT_GOOD;
tape evidence still gives CONFIDENT_DEFECT on a merged blob.
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


def hsv_bgr(hh, ss, vv):
    px = np.uint8([[[hh, ss, vv]]])
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


ORANGE = (0, 140, 255)
RED = (0, 0, 220)
YELLOW = (0, 220, 240)
WHITE = (255, 255, 255)
SKIN = (140, 170, 220)  # BGR: hue 11, saturation 92 (skin-like)
PINK = hsv_bgr(4, 100, 247)  # the pink 2x2 that refused e09_end under E2 (median HSV 4, 100, 247)
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


def test_two_hue_blob_is_split_into_colour_pieces():
    img = rect(mat(), 300, 250, 30, 30, RED)
    img = rect(img, 330, 250, 30, 30, YELLOW)  # two touching bricks of different colour
    bricks = I.scan(img)
    assert len(bricks) == 2, bricks
    assert all(b["split"] and b["verdict"] == "UNCERTAIN" for b in bricks), bricks
    assert sorted(b["colour"] for b in bricks) == ["red", "yellow"], bricks
    assert all(b["shape"] == "2x2" for b in bricks), bricks  # each piece is one brick, never a 2x4


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
    cv2.ellipse(img, (300, 0), (50, 30), 0, 0, 360, SKIN, -1)  # a hand coming in over the top edge of the mat
    try:
        I.scan(img)
    except I.HandsPresent:
        return
    raise AssertionError("hands were not refused")


def test_interior_irregular_skin_refuses():
    img = rect(mat(), 250, 300, 150, 20, SKIN)  # 3000 px skin strip on the mat: large and not brick-shaped
    try:
        I.scan(img)
    except I.HandsPresent:
        return
    raise AssertionError("large interior skin strip was not refused")


def test_pink_brick_is_not_a_hand():
    img = rect(mat(), 300, 250, 30, 30, PINK)  # the e09_end false refusal: a pink 2x2 on the mat
    res = I.scan_full(img)
    assert res["hands"] == [], res["hands"]
    assert len(res["bricks"]) == 1 and res["bricks"][0]["colour"] == "pink", res["bricks"]


def test_skin_outside_mat_is_ignored():
    img = mat()
    cv2.ellipse(img, (60, 300), (30, 20), 0, 0, 360, SKIN, -1)  # skin over the wood table, left of the mat
    res = I.scan_full(img)
    assert res["hands"] == [] and res["bricks"] == [], res


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


def _close_rect(with_tape, w=70, h=66):
    rng = np.random.default_rng(1)
    img = rect(mat(480, 640), 200, 200, w, h, ORANGE)
    if with_tape:
        g = rng.integers(50, 90, size=(14, 46, 1)).astype(np.uint8)  # grey, textured dark tape band
        img[228:242, 212:258] = np.repeat(g, 3, axis=2)
    return img


def test_close_tape_defect_vs_clean():
    tape = I.segment_close(_close_rect(True))["bricks"]
    clean = I.segment_close(_close_rect(False))["bricks"]
    assert len(tape) == 1 and len(clean) == 1, (len(tape), len(clean))
    assert tape[0]["tape_score"] >= THR["defect"], tape[0]["tape_score"]
    assert clean[0]["tape_score"] <= THR["good"], clean[0]["tape_score"]
    assert I.triage(tape[0], THR)[0] == "CONFIDENT_DEFECT"
    assert I.triage(clean[0], THR)[0] == "CONFIDENT_GOOD"
    assert I.triage(clean[0], None)[0] == "UNCERTAIN"  # no threshold file: never a confident call


def test_close_elongated_blob_is_never_good():
    bricks = I.segment_close(rect(mat(480, 640), 200, 200, 100, 60, ORANGE))["bricks"]  # aspect 1.67
    assert len(bricks) == 1 and bricks[0]["merged"] and bricks[0]["merged_reason"] == "elongated_close", bricks
    assert I.triage(bricks[0], THR)[0] == "UNCERTAIN"


def test_close_merged_blob_with_tape_is_defect():
    img = rect(mat(480, 640), 200, 200, 100, 60, ORANGE)
    g = np.random.default_rng(2).integers(50, 90, size=(14, 46, 1)).astype(np.uint8)
    img[220:234, 212:258] = np.repeat(g, 3, axis=2)  # tape on an elongated (merged) blob
    bricks = I.segment_close(img)["bricks"]
    assert len(bricks) == 1, bricks
    assert I.triage(bricks[0], THR)[0] == "CONFIDENT_DEFECT", bricks[0]["tape_score"]


def test_single_brick_contract():
    img = rect(mat(), 300, 250, 30, 30, ORANGE)
    out = I.Inspector(device="CPU").inspect_rgb(img)
    assert out["label"] in ("GOOD", "DEFECTIVE", "UNCERTAIN")
    assert isinstance(out["reasons"], list)
    box = I.Inspector().locate_brick(img)
    assert box is not None and len(box) == 4


FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _live_153426():
    """Live stills 153426 (JPEG, robot cell). Ground truth: red 2x2 GOOD, white 2x4 DEFECT (not_2x2 plus tape), white 2x2
    GOOD, yellow 2x2 DEFECT (tape). Known miss (V1): the white 2x4 is pinned at UNCERTAIN until thresholds are re-calibrated."""
    top = cv2.imread(os.path.join(FIX, "live_top_153426.jpg"))
    close = cv2.imread(os.path.join(FIX, "live_close_153426.jpg"))
    return I.scan_full(top, close)


def test_live_153426_no_hands_and_no_rail_bricks():
    res = _live_153426()
    assert res["hands"] == []
    top = [b for b in res["bricks"] if b["view"] == "top"]
    assert len(top) == 4
    assert all(b["centre_px"][1] < I.TOP_Y1 for b in top)  # no clamp or blue-edge false bricks below the rail


def test_live_153426_top_shapes_and_verdicts():
    top = [b for b in _live_153426()["bricks"] if b["view"] == "top"]
    assert sorted(b["colour"] for b in top) == ["red", "white", "white", "yellow"]
    white = sorted((b for b in top if b["colour"] == "white"), key=lambda b: -b["area"])
    big, small = white[0], white[1]
    assert big["shape"] == "not_2x2" and big["verdict"] == "UNCERTAIN"  # known miss: expected CONFIDENT_DEFECT
    assert small["shape"] == "2x2" and small["verdict"] != "CONFIDENT_DEFECT"
    for b in top:
        if b["colour"] in ("red", "yellow"):
            assert b["verdict"] != "CONFIDENT_DEFECT"


def test_live_153426_close_tape_and_clean_bricks():
    close = [b for b in _live_153426()["bricks"] if b["view"] == "close"]
    assert len(close) == 4
    assert sum(1 for b in close if b["colour"] == "white" and b["bbox"][1] < 300) == 1  # white 2x4 kept as one brick
    yellow = [b for b in close if b["colour"] == "yellow"]
    assert len(yellow) == 1 and yellow[0]["verdict"] == "CONFIDENT_DEFECT"  # taped yellow brick
    clean = [b for b in close if b["colour"] == "red" or (b["colour"] == "white" and b["bbox"][1] >= 300)]
    assert len(clean) == 2 and all(b["verdict"] != "CONFIDENT_DEFECT" for b in clean)


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
