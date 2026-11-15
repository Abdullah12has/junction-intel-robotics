#!/usr/bin/env bash
set -euo pipefail
R=/home/intel/intel-junction-2026
P=ab767473-d6fd-47ff-812e-8239c455dabb
ENV=2ad7d643-b79b-417b-9948-6734e295bc21
NAME="Best-left X painted"
API=http://127.0.0.1:7860
C="curl -s --noproxy 127.0.0.1"
cd "$R"
envs/lerobot-env/bin/python -I - <<'EOF'
import sys, json, os
sys.path.insert(0, "app/data")
import cv2, numpy as np
import paint_cross as P
src = os.path.expanduser("~/.local/share/physicalai/datasets/554ce492-3fb1-496b-862a-347551ed38c0")
out = "data/painted/best_left_x"
info = json.load(open(os.path.join(src, "meta", "info.json")))
eps = P.episodes(src)
want = {0, 5, 12, 33, 40, 63}
picks = {}
for path, spans in P.video_files(out, info, P.TOP, eps).items():
    mids = {e: (a + b) // 2 for e, a, b in spans if e in want}
    for idx, img in P.frames(path, info["fps"]):
        for e, m in mids.items():
            if idx == m:
                picks[e] = img
tiles = []
for e in sorted(picks):
    t = cv2.resize(picks[e], (480, 360), interpolation=cv2.INTER_AREA)
    cv2.putText(t, f"painted ep {e} mid", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    tiles.append(t)
while len(tiles) % 3:
    tiles.append(np.zeros((360, 480, 3), np.uint8))
cv2.imwrite("data/painted/best_left_x_report/verify.jpg", np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print("verify.jpg written", sorted(picks))
EOF
rm -f data/painted/best_left_x.zip
(cd data/painted/best_left_x && zip -0 -qr ../best_left_x.zip .)
ls -la data/painted/best_left_x.zip
JOB=$($C -X POST -F format_hint=auto -F "dataset_name=$NAME" "$API/api/projects/$P/imports/datasets:prepare" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "job $JOB"
$C -X PUT -F "archive=@data/painted/best_left_x.zip" "$API/api/projects/$P/imports/datasets/$JOB:upload" | head -c 300; echo
step() { $C "$API/api/jobs/$JOB" | python3 -c "import json,sys; j=json.load(sys.stdin); p=j.get('payload') or {}; print(j.get('status'), p.get('step'), p.get('result_dataset_id'), (j.get('message') or '')[:200])"; }
for i in $(seq 1 120); do s=$(step); echo "$s"; case "$s" in *awaiting_user_review*|*failed*|*FAILED*) break;; esac; sleep 5; done
$C -X POST -H 'Content-Type: application/json' -d "{\"environment_id\":\"$ENV\",\"default_task\":\"\"}" "$API/api/projects/$P/imports/datasets/$JOB:finalize" | head -c 300; echo
for i in $(seq 1 240); do s=$(step); echo "$s"; case "$s" in completed*|*COMPLETED*|*failed*|*FAILED*) break;; esac; sleep 5; done
DS=$($C "$API/api/jobs/$JOB" | python3 -c "import json,sys; print((json.load(sys.stdin).get('payload') or {}).get('result_dataset_id') or '')")
echo "dataset $DS"
if [ -n "$DS" ]; then
  $C -X DELETE -H 'Content-Type: application/json' -d '[33]' "$API/api/dataset/$DS/episodes" | python3 -c "import json,sys; d=json.load(sys.stdin); print('episodes after delete', len(d) if isinstance(d, list) else d)"
fi
echo DONE
