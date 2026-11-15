#!/usr/bin/env bash
# Writes R/run/arm_state every 15 s: FREE since <utc> or BUSY <pids>, for the follower (by-id 5B3D040939) and both RealSense colour nodes.
R=/home/intel/intel-junction-2026
END=$(date -d '2026-11-14 23:00 UTC' +%s)
since=""
while [ "$(date +%s)" -lt "$END" ]; do
  F=$(readlink -f /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00 2>/dev/null)
  holders=""
  for p in /proc/[0-9]*; do
    for f in "$p"/fd/*; do
      l=$(readlink "$f" 2>/dev/null)
      if [ -n "$F" ] && { [ "$l" = "$F" ] || [ "$l" = /dev/video8 ] || [ "$l" = /dev/video14 ]; }; then holders="$holders $(basename "$p")"; break; fi
    done
  done
  now=$(date -u +%FT%TZ)
  if [ -z "$holders" ]; then
    [ -z "$since" ] && since=$now
    echo "FREE since $since (checked $now)" > "$R/run/arm_state.tmp"
  else
    since=""
    echo "BUSY$holders (checked $now)" > "$R/run/arm_state.tmp"
  fi
  mv -f "$R/run/arm_state.tmp" "$R/run/arm_state"
  sleep 15
done
