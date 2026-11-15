#!/usr/bin/env bash
# Read-only mirror of the team's simulation data into the project folder every 120 s until 2026-11-16 06:00Z (no deadline: I-66) (Hassan: keep the simulated data synced too). Skips rsync temp files.
SRC=/home/intel/junction-cell/simulation-data/
DST=/home/intel/intel-junction-2026/data/sim/live/
END=$(date -d '2026-11-16 06:00 UTC' +%s)
mkdir -p "$DST"
while [ "$(date +%s)" -lt "$END" ]; do
  rsync -a --exclude='.*' "$SRC" "$DST" && echo "$(date -u +%FT%TZ) synced sim $(du -sh "$DST" | cut -f1) files=$(find "$DST" -type f | wc -l)"
  sleep 120
done
