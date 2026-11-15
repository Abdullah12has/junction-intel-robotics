Sat Nov 14 08:35:09 PM UTC 2026 CMD: nice -n 19 envs/data-env/bin/python -I tools/meta_dump.py
Sat Nov 14 08:35:17 PM UTC 2026 CMD: find data+curated -type f \( -name '*.parquet' -o -name '*.mp4' -o -name '*.json' \) -print0 | xargs -0 nice -n 19 sha256sum > out/sha256.txt
Sat Nov 14 08:35:29 PM UTC 2026 CMD: summarize.py -> out/summary.txt
Sat Nov 14 08:35:43 PM UTC 2026 CMD: physical.py -> out/physical_map.tsv
Sat Nov 14 08:36:21 PM UTC 2026 CMD: windows.py -> out/windows.tsv
Sat Nov 14 08:36:38 PM UTC 2026 CMD: bash tools/sheets.sh out/windows.tsv (contact sheets, 4 times x 2 cams per window)
Sat Nov 14 08:44:22 PM UTC 2026 CMD: full_frames.py (native 640x480, 6 frames x 2 cams per window, montage in frames/F*.png)
