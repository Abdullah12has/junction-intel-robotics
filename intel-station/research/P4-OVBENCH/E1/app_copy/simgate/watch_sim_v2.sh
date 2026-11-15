#!/bin/bash
# watch_sim.sh: runs sim_gate.py on every version folder under data/sim/live whenever
# its file tree changes. A change counts only after the tree is unchanged for one
# more poll, so a half-synced version is not judged. Stops 2026-11-16 06:00Z (no deadline: I-66).
R=/home/intel/intel-junction-2026
LIVE="$R/data/sim/live"
PY="$R/work/P2-D-SIM2/V1/venv/bin/python"
GATE="$R/app/simgate/sim_gate.py"
STOP_UTC="2026-11-16T06:00:00Z"
POLL=60
mkdir -p "$R/run" "$R/logs" "$R/data/sim/gate"
echo $$ > "$R/run/sim_gate.pid"

fingerprint() {
  (cd "$LIVE" 2>/dev/null && find . -type f ! -name '.*' ! -name '*.tmp' -printf '%p %s %T@\n' | sort | sha1sum | cut -d' ' -f1)
}

last_run=""
pending=""
while [ "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \< "$STOP_UTC" ]; do
  fp=$(fingerprint)
  if [ -n "$fp" ] && [ "$fp" != "$last_run" ]; then
    if [ "$fp" = "$pending" ]; then
      for root in "$LIVE"/*/; do
        root=${root%/}
        if ls -d "$root"/dataset-*-raw >/dev/null 2>&1; then
          nice -n 10 "$PY" -I "$GATE" --version-root "$root" >> "$R/logs/sim_gate.out" 2>&1
        fi
      done
      last_run="$fp"
      pending=""
    else
      pending="$fp"
    fi
  fi
  sleep "$POLL"
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) watch_sim stopped at deadline" >> "$R/logs/sim_gate.log"
