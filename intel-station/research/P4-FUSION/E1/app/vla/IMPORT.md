# Studio import for E1 datasets (P2-D-VLA1)

**Status: UNTESTED.** No step below has been run against Studio. The steps come from the Studio backend source (repos/physical-ai-studio/application/backend, read in this run) and from CUR1's import script. Running them needs Opus approval (request R1 in STATUS.md).

Rules:
- Each import creates a NEW dataset. Use a unique dataset name per import.
- The synthetic sets (SYNTH-A-3EP, SYNTH-B-150EP) are test data. Never import them into a real dataset.
- Studio accepts ZIP archives only.

## 1. Build the ZIP

    bash /home/intel/intel-junction-2026/app/vla/build.sh --runs <RUN_DIR> [<RUN_DIR> ...] --out <OUT_DIR> --variant team --only-success --zip

- `--out` must be inside /home/intel/intel-junction-2026 (build_dataset.py checks this).
- `--zip` writes `<OUT_DIR>.zip` next to the dataset folder, with MP4 stored and other files deflated (the zip block at the end of main() in build_dataset.py).

## 2. Prepare the import job

    curl -sS --noproxy '*' -X POST --data-urlencode "format_hint=auto" --data-urlencode "dataset_name=<UNIQUE_NAME>" "http://127.0.0.1:7860/api/projects/<PROJECT_ID>/imports/datasets:prepare"

- Route: POST /api/projects/{project_id}/imports/datasets:prepare (src/api/dataset_import.py lines 28 and 104).
- Form fields: `format_hint` (default `auto`) and `dataset_name` (must be non-empty; lines 108-126). The API file has no duplicate-name check, so use a unique name.
- The response is a Job with payload.step = awaiting_archive_upload (README lines 9 and 41). Take the job id from the response. CUR1's script reads `id` or `job_id`; verify this.

## 3. Upload the ZIP

    curl -sS --noproxy '*' -X PUT -F "archive=@<OUT_DIR>.zip" "http://127.0.0.1:7860/api/projects/<PROJECT_ID>/imports/datasets/<JOB_ID>:upload"

- The form file field is `archive` (line 139). ZIP only.
- On success the job becomes pending with payload.step = queued_for_detection (README lines 10 and 54).

## 4. Poll until the review step

    curl -sS --noproxy '*' "http://127.0.0.1:7860/api/jobs/<JOB_ID>"

- Route: GET /api/jobs/{job_id} (src/api/job.py line 31).
- Wait until payload.step = awaiting_user_review (README line 64). Do not finalize before this.

## 5. Finalize

    curl -sS --noproxy '*' -X POST -H "Content-Type: application/json" -d '{"environment_id": "<ENVIRONMENT_ID>", "default_task": "<TASK TEXT>"}' "http://127.0.0.1:7860/api/projects/<PROJECT_ID>/imports/datasets/<JOB_ID>:finalize"

- `environment_id` is required (UUID). `default_task` is optional (src/schemas/dataset_import_job.py lines 97-98).
- Success moves the job to status pending with payload.step = queued_for_import (README line 96).

## 6. Cancel (only if needed)

    curl -sS --noproxy '*' -X POST "http://127.0.0.1:7860/api/projects/<PROJECT_ID>/imports/datasets/<JOB_ID>:cancel"

- Cancel is allowed only in awaiting_user_review (README line 111).

## Values taken from CUR1's import_ds.sh (verify before use)

- Base URL: http://127.0.0.1:7860 (not checked against the running Studio)
- PROJECT_ID: ab767473-d6fd-47ff-812e-8239c455dabb
- ENVIRONMENT_ID: 72206569-df16-4a8b-a98f-247bac4de63f

## Not verified

- Every HTTP step. Nothing was sent to Studio in this run.
- Whether the prepare response uses the field name `id`.
- Whether the running Studio listens on port 7860.
