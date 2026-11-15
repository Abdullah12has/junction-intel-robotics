import glob
import os
from datetime import datetime, timezone

import av

ROOT = "/home/intel/intel-junction-2026"
FOLDERS = [("top", ROOT + "/logs/monitor", "top"), ("close", ROOT + "/logs/monitor", "close"),
           ("wrist", ROOT + "/data/wrist/raw", "w")]


def utc(stamp):
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()


for cam, folder, prefix in FOLDERS:
    files = sorted(glob.glob(os.path.join(folder, prefix + "_*.mkv")))
    n_abs, n_rel, bad, offsets = 0, 0, [], []
    first_abs, last_rel, transitions, creation = None, None, [], {}
    prev_mode = None
    for p in files:
        name = os.path.basename(p)
        ts = utc(name[len(prefix) + 1:-4])
        try:
            c = av.open(p)
            st = c.start_time
            md = dict(c.metadata)
            c.close()
        except Exception:
            bad.append(name)
            continue
        ab = bool(st) and st / 1e6 > 1e9
        mode = "abs" if ab else "rel"
        if ab:
            n_abs += 1
            offsets.append(st / 1e6 - ts)
            if first_abs is None:
                first_abs = name
        else:
            n_rel += 1
            last_rel = name
        if md.get("creation_time") and mode not in creation:
            creation[mode] = (name, md.get("creation_time"))
        if mode != prev_mode:
            transitions.append((name, mode))
            prev_mode = mode
    print(cam, "files", len(files), "absolute", n_abs, "relative", n_rel, "unreadable", len(bad), bad[:3])
    if offsets:
        print(cam, "absolute start minus filename second: min %.3f max %.3f" % (min(offsets), max(offsets)))
    print(cam, "mode transitions", transitions[:6], "first_abs", first_abs, "last_rel", last_rel)
    print(cam, "creation_time samples", creation)
