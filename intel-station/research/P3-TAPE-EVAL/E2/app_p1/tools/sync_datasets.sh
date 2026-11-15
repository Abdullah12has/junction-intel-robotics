#!/usr/bin/env bash
# Read-only snapshot of Studio datasets into the project folder every 120 s until 2026-11-15 01:00 Helsinki (Hassan's request: use every dataset).
SRC=/home/intel/.local/share/physicalai/datasets/
DST=/home/intel/intel-junction-2026/data/studio-datasets/
END=$(date -d '2026-11-14 23:00 UTC' +%s)
while [ "$(date +%s)" -lt "$END" ]; do
  rsync -a "$SRC" "$DST" && echo "$(date -u +%FT%TZ) synced $(ls -1 "$DST" | wc -l) datasets $(du -sh "$DST" | cut -f1)"
  sleep 120
done
