#!/bin/bash
# Bring the pick-anywhere stack back up after the robot was lent out, in a safe order.
#   app/tools/resume.sh            -> checks + starts recorders, hand watchdog, motion server; prints the loop command
#   app/tools/resume.sh --go NAME  -> same, then starts sort_any.py --run NAME (new run folder, --start 1)
# Run only when the robot is free for this stack. Never touches Studio; refuses if anything else holds the
# follower port. Every process is started with a pid file; stop them by pid file only.
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # workspace root, from this script's location
PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040939-if00
cd "$R" || exit 1
say() { echo "$(date -u +%H:%M:%SZ) $*"; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

# 1. the follower port must be free (no Studio session, no other process)
DEV=$(readlink -f "$PORT" 2>/dev/null)
if [ -z "$DEV" ]; then say "follower port $PORT not present - is the arm plugged in?"; exit 2; fi
HOLD=$(fuser "$DEV" 2>/dev/null | tr -s ' ')
if [ -n "$HOLD" ] && ! alive run/mserver/pid; then say "follower $DEV is held by pid(s)$HOLD - not ours, stopping"; exit 3; fi
SESS=$(envs/data-env/bin/python -B app/tools/studio_sessions.py 2>/dev/null | grep -ci follower)
if [ "${SESS:-0}" != "0" ]; then say "a Studio runtime session mentions the follower - stop it in Studio first"; exit 4; fi

# 2. cameras free? (the recorders need /dev/video8, /dev/video14 and /dev/video16)
for v in /dev/video8 /dev/video14 /dev/video16; do
  h=$(fuser "$v" 2>/dev/null | tr -s ' ')
  if [ -n "$h" ] && ! alive run/monitor_cams.pid && ! alive run/wrist_recorder.pid; then say "$v held by pid(s)$h - not ours, stopping"; exit 5; fi
done

# 3. recorders (they also feed run/live_*.jpg for the watchdog and the loop)
alive run/monitor_cams.pid || { nohup bash app/tools/monitor_cams.sh >> logs/monitor/monitor_cams.out 2>&1 & say "monitor_cams started"; }
alive run/wrist_recorder.pid || { nohup bash app/tools/wrist_rec.sh >> logs/wrist_rec.out 2>&1 & say "wrist recorder started"; }
sleep 4
for c in top close wrist; do
  a=$(( $(date +%s) - $(stat -c %Y run/live_$c.jpg 2>/dev/null || echo 0) ))
  [ "$a" -le 3 ] && say "live_$c.jpg fresh (${a}s)" || { say "live_$c.jpg is ${a}s old - camera not streaming"; exit 6; }
done

# 4. hand watchdog (shared detector app/tools/handdet.py)
alive run/hand_watch.pid || { nohup envs/data-env/bin/python -u -B app/tools/hand_watch.py >> logs/monitor/hand_watch.out 2>&1 & echo $! > run/hand_watch.pid; say "hand_watch started pid $(cat run/hand_watch.pid)"; }

# 5. motion server (torque on at the present pose; no motion)
if ! alive run/mserver/pid; then
  nohup envs/motion-env/bin/python -u -B app/tools/motion_server.py > run/mserver/stdout.log 2>&1 &
  for i in $(seq 1 30); do sleep 1; alive run/mserver/pid && grep -q '"ready"\|"ok"' run/mserver/state.json 2>/dev/null && break; done
fi
alive run/mserver/pid && say "motion server up: $(cut -c1-120 run/mserver/state.json)" || { say "motion server did not start - see run/mserver/stdout.log"; exit 7; }

CMD="cd $R/app/tools && nohup ../../envs/data-env/bin/python -u -B sort_any.py --run \$NAME --n 150 --start 1 >> ../../logs/sort_any_\$NAME.out 2>&1 &"
if [ "${1:-}" = "--go" ] && [ -n "${2:-}" ]; then
  NAME=$2
  rm -f run/sort_stop run/sort_pause
  echo "$NAME" > run/sort_run_name
  (cd app/tools && nohup ../../envs/data-env/bin/python -u -B sort_any.py --run "$NAME" --n 150 --start 1 >> "../../logs/sort_any_$NAME.out" 2>&1 &)
  say "loop started: run $NAME (log logs/sort_any_$NAME.out)"
else
  say "stack ready. Start the loop with:  NAME=<run>; $CMD"
fi
