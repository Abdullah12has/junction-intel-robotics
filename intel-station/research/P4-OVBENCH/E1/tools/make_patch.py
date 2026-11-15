# Candidate edit on a COPY of app/ only (R/app is read). Writes a unified diff for patch -p1 from the R root.
import difflib
import os
import sys

orig_root, copy_root, out = sys.argv[1], sys.argv[2], sys.argv[3]
rel = os.path.join('anomaly', 'anom3', 'infer.py')
cp = os.path.join(copy_root, rel)
src = open(cp, encoding='utf-8').read()
edits = [
    ('DEVICES = ("CPU",)  # CPU only: no GPU jobs on this laptop (coordinator update, 18:32Z)',
     'DEVICES = ("NPU", "CPU")  # E1 candidate: NPU first, CPU fallback (validate before merge)'),
    ('bad = [d for d in devices if d != "CPU"]',
     'bad = [d for d in devices if d not in ("CPU", "NPU")]'),
]
for old, new in edits:
    n = src.count(old)
    if n != 1:
        raise SystemExit('edit anchor found %d times: %s' % (n, old))
    src = src.replace(old, new)
open(cp, 'w', encoding='utf-8').write(src)
orig = open(os.path.join(orig_root, rel), encoding='utf-8').read()
p = rel.replace(os.sep, '/')
text = ''.join(difflib.unified_diff(orig.splitlines(True), src.splitlines(True),
                                    fromfile='a/app/' + p, tofile='b/app/' + p))
open(out, 'w', encoding='utf-8').write(text)
changed = [l for l in text.splitlines() if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++', '---'))]
print('changed lines: %d' % len(changed))
