"""Fix of builder E1 (L39). (1) Replaces fixture servo/servo_181451_red.png with the frame chosen by the L37 window
search (w_20261114T181351Z.mkv at +59.020 s). (2) Updates the two MANIFEST.tsv rows that change: servo_181451_red.png,
and hand_labels.csv (its row no longer cites the other folder's feature table, which is not an input of this task).
Writes only inside the tests fixtures folder and the E1 raw folder.
"""
import hashlib
import os
import shutil

FIX = "/home/intel/intel-junction-2026/app/tools/tests/fixtures"
RAW = "/home/intel/intel-junction-2026/work/P2-B-TESTS/E1/raw"
MAN = os.path.join(FIX, "MANIFEST.tsv")
SERVO_NAME = "servo/servo_181451_red.png"
CAND = os.path.join(RAW, "L37_servo181451_candidate.png")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    old_path = os.path.join(FIX, SERVO_NAME)
    old_sha, new_sha = sha(old_path), sha(CAND)
    print("servo_181451 fixture before fix, sha256", old_sha)
    print("servo_181451 candidate (L37), sha256", new_sha)
    if old_sha != new_sha:
        shutil.copyfile(CAND, old_path)
        assert sha(old_path) == new_sha
        print("fixture replaced; sha256 now", sha(old_path))
    else:
        print("fixture already equal to the candidate; no file change")
    with open(MAN, encoding="utf-8") as f:
        lines = f.read().split("\n")
    n_servo = n_hand = 0
    for i, line in enumerate(lines):
        parts = line.split("\t")
        if parts[0] == SERVO_NAME:
            n_servo += 1
            assert len(parts) == 6, len(parts)
            parts[1] = ("logs/sort_any_any_1740.out line 277; video data/wrist/raw/w_20261114T181351Z.mkv at +59.020 s "
                        "(nominal 18:14:50.80Z; search window 18:14:50.50Z to 18:14:51.50Z)")
            parts[2] = "2026-11-14 18:14:51Z (log line 277)"
            parts[3] = "colour=red centre=(420.2,346.7) short=108.8 (log line 277)"
            parts[4] = ("replaced in L39: the earlier frame was the first frame of w_20261114T181451Z.mkv at 18:14:51.76Z "
                        "(+0.76 s, outside the +-0.5 s window; L37). New frame: centre 0.1 px and short diff 0.0 from the log "
                        "(L37); test_servo_replay_seeded_at_logged_centre")
            parts[5] = new_sha
            lines[i] = "\t".join(parts)
        elif parts[0] == "hand_labels.csv":
            n_hand += 1
            assert len(parts) == 6, len(parts)
            parts[1] = "Opus labels (coordinator message 18:53Z); fixtures/hand_labels.csv"
            parts[4] = ("not re-derived by E1; used by test_hand_labels_if_present (wrist rows skipped, hand_partial not "
                        "asserted); the feature-table path cited before L39 was removed (not an input of this task)")
            lines[i] = "\t".join(parts)
    assert n_servo == 1 and n_hand == 1, (n_servo, n_hand)
    with open(MAN, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))
    print("MANIFEST rows updated; MANIFEST sha256 now", sha(MAN))


if __name__ == "__main__":
    main()
