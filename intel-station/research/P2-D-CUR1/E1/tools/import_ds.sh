#!/bin/bash
# Import one curated zip as a NEW Studio dataset in the Junction 2026 project: prepare, upload, finalize.
# Usage: bash import_ds.sh <zip path> <dataset name> <default task text>
set -u
B=http://127.0.0.1:7860
P=ab767473-d6fd-47ff-812e-8239c455dabb
ENVID=72206569-df16-4a8b-a98f-247bac4de63f
ZIP="$1"
NAME="$2"
TASK="$3"
R=$(curl -sS --noproxy '*' -X POST --data-urlencode "format_hint=auto" --data-urlencode "dataset_name=$NAME" "$B/api/projects/$P/imports/datasets:prepare")
echo "prepare response: $(printf '%s' "$R" | head -c 300)"
JOB=$(printf '%s' "$R" | python3 -I -c 'import json,sys; d=json.load(sys.stdin); print(d.get("id") or d.get("job_id") or "")')
echo "import job: $JOB"
[ -n "$JOB" ] || exit 1
U=$(curl -sS --noproxy '*' -X PUT -F "archive=@$ZIP" "$B/api/projects/$P/imports/datasets/$JOB:upload")
echo "upload response: $(printf '%s' "$U" | head -c 300)"
BODY=$(python3 -I -c 'import json,sys; print(json.dumps({"environment_id": sys.argv[1], "default_task": sys.argv[2]}))' "$ENVID" "$TASK")
F=$(curl -sS --noproxy '*' -X POST -H 'Content-Type: application/json' -d "$BODY" "$B/api/projects/$P/imports/datasets/$JOB:finalize")
echo "finalize response: $(printf '%s' "$F" | head -c 300)"
echo "IMPORT_JOB=$JOB NAME=$NAME"
