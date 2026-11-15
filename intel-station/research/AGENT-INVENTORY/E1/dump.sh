#!/bin/bash
# AGENT-INVENTORY E1: read-only dump of every R/work/<TASK>/<STAGE>/ folder.
# Prints per folder: path, first line of STATUS.md, file count, newest file mtime (UTC), top-level files with sizes.
R=/home/intel/intel-junction-2026
cd "$R/work" || exit 1
echo "# dump_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) root=$R/work"
n=0
for t in */; do
  [ -d "$t" ] || continue
  t=${t%/}
  for s in "$t"/*/; do
    [ -d "$s" ] || continue
    s=${s%/}
    n=$((n+1))
    first=""
    if [ -f "$s/STATUS.md" ]; then first=$(head -n 1 "$s/STATUS.md" 2>/dev/null | cut -c1-220); fi
    nfiles=$(find "$s" -type f 2>/dev/null | wc -l)
    newest=$(find "$s" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -n 1)
    nd="none"
    if [ -n "$newest" ]; then nd=$(date -u -d "@${newest%.*}" +%Y-%m-%dT%H:%MZ); fi
    echo "== $s | files=$nfiles | newest_utc=$nd"
    echo "   STATUS.md line1: ${first:-<no STATUS.md>}"
    find "$s" -maxdepth 1 -type f -printf '   %s bytes  %f\n' 2>/dev/null
  done
done
echo "# stage_folders=$n"
echo "# dump_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
