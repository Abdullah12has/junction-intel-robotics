#!/usr/bin/env bash
# Start, stop or check the 1 Hz telemetry sampler. Only the pid recorded by this script is signalled.
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # workspace root, from this script's location
D=$R/logs/telemetry
PIDF=$D/telemetry.pid
mkdir -p "$D"
running() {
  P=""
  [ -f "$PIDF" ] || return 1
  P=$(cat "$PIDF")
  [ -n "$P" ] && grep -q "telemetry.py" "/proc/$P/cmdline" 2>/dev/null
}
case "${1:-status}" in
  start)
    if running; then echo "already running pid $P"; exit 0; fi
    nohup nice -n 10 python3 -I -u "$R/app/telemetry/telemetry.py" run >> "$D/sampler.log" 2>&1 &
    sleep 3
    if running; then echo "started pid $P"; else echo "not running, see $D/sampler.log"; exit 1; fi
    ;;
  stop)
    if running; then kill -TERM "$P"; sleep 2; echo "sent TERM to pid $P"; else echo "not running"; fi
    ;;
  status)
    if running; then echo "running pid $P"; else echo "not running"; fi
    ;;
  *)
    echo "usage: telemetry_ctl.sh start|stop|status"; exit 2
    ;;
esac
