#!/bin/bash
# P2-B-PROBE E1 read-only gate. No hardware access: reads run/arm_state and /proc/*/fd symlinks only.
# GO when arm_state reads "FREE since <ts>" with age >= 60 s AND no holder of ttyACM0 (follower), ttyACM1 (leader), video8, video14.
# Polls every 60 s for up to 60 min, then ABORTED-BUSY.
R=/home/intel/intel-junction-2026
D=$R/work/P2-B-PROBE/E1
LOG=$D/gate.log
holders() {
  for p in /proc/[0-9]*; do
    for f in "$p"/fd/*; do
      t=$(readlink "$f" 2>/dev/null)
      case "$t" in
        /dev/ttyACM0|/dev/ttyACM1|/dev/video8|/dev/video14) echo "pid=${p#/proc/} dev=$t comm=$(cat "$p"/comm 2>/dev/null)";;
      esac
    done
  done
}
start=$(date -u +%s)
i=0
while :; do
  i=$((i+1))
  t0=$(date -u +%s)
  st=$(cat "$R/run/arm_state" 2>/dev/null | tr '\n' ' ')
  ts=$(printf '%s' "$st" | sed -n 's/^FREE since \([^ ]*\).*/\1/p')
  age=-1
  if [ -n "$ts" ]; then
    since=$(date -u -d "$ts" +%s 2>/dev/null)
    if [ -n "$since" ]; then age=$(( $(date -u +%s) - since )); fi
  fi
  h=$(holders | tr '\n' ';')
  verdict=WAIT
  if [ -n "$ts" ] && [ "$age" -ge 60 ] && [ -z "$h" ]; then verdict=GO; fi
  echo "$(date -u +%H:%M:%SZ) poll=$i arm_state=[$st] free_age_s=$age holders=[$h] verdict=$verdict" >> "$LOG"
  if [ "$verdict" = GO ]; then echo "GO $(date -u +%H:%M:%SZ)" >> "$LOG"; exit 0; fi
  if [ $(( $(date -u +%s) - start )) -ge 3600 ]; then echo "ABORTED-BUSY $(date -u +%H:%M:%SZ) 60 min elapsed without GO" >> "$LOG"; exit 1; fi
  el=$(( $(date -u +%s) - t0 ))
  if [ "$el" -lt 60 ]; then sleep $((60 - el)); fi
done
