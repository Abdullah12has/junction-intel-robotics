# S9 item 11: candidate copy under E1 with two edits to the COPY only. R/app is not written.
import ast
import sys

E = "/home/intel/intel-junction-2026/work/P4-OVBENCH/E1"
P = E + "/app_copy/anomaly/anom3/infer.py"
REPS = [
    ('DEVICES = ("CPU",)  # CPU only: no GPU jobs on this laptop (coordinator update, 18:32Z)',
     'DEVICES = ("GPU", "CPU")  # candidate (S9): GPU first, INFERENCE_PRECISION_HINT f32 at compile, CPU fallback'),
    ('        bad = [d for d in devices if d != "CPU"]\n        if bad:\n            raise ValueError("anom3 runs on CPU only; refused devices %s" % bad)',
     '        bad = [d for d in devices if d not in ("CPU", "GPU")]\n        if bad:\n            raise ValueError("anom3 runs on CPU or GPU only; refused devices %s" % bad)'),
]


def main():
    with open(P, encoding="utf-8") as fh:
        s = fh.read()
    for old, new in REPS:
        n = s.count(old)
        if n != 1:
            print("REPLACE FAILED (count %d): %s" % (n, old[:60]))
            return 1
        s = s.replace(old, new)
    ast.parse(s)
    with open(P, "w", encoding="utf-8") as fh:
        fh.write(s)
    print("candidate edits applied to the COPY; ast.parse OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
