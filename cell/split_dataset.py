"""Copy a recorded Studio dataset once per task string, keeping only that task's episodes (for ACT).

ACT ignores the task string, so each ACT skill needs a dataset holding only its own episodes.
Studio has no merge or filter; this copies the dataset with Studio's own import-dir CLI and
deletes the other tasks' episodes through Studio's API. The source dataset is never modified.

Run on the laptop (any Python 3 with no extra packages), with the dataset NOT loaded in any session:
  python3 cell/split_dataset.py <dataset_id>
Untested: no recorded dataset existed when this was written (14 Nov).
"""
import json
import subprocess
import sys
import urllib.request

API = "http://127.0.0.1:7860/api"
BACKEND = "/home/intel/Documents/physical-ai-studio/application/backend"


def call(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


src_id = sys.argv[1]
src = call("GET", f"/dataset/{src_id}")
episodes = call("GET", f"/dataset/{src_id}/episodes")
datasets_dir = subprocess.check_output(
    [".venv/bin/python", "-c", "import sys; sys.path.insert(0,'src'); from settings import get_settings; "
                               "print(get_settings().datasets_dir)"], cwd=BACKEND, text=True).strip()
tasks = sorted({t for e in episodes for t in e["tasks"]})
print(f"{src['name']}: {len(episodes)} episodes, tasks: {tasks}")

for task in tasks:
    out = subprocess.check_output(
        [".venv/bin/physicalai-studio", "datasets", "import-dir", "--source-dir", f"{datasets_dir}/{src_id}",
         "--project-id", src["project_id"], "--environment-id", src["environment_id"],
         "--dataset-name", f"{src['name']} | {task}", "--default-task", task, "--copy"], cwd=BACKEND, text=True)
    new_id = next(line.split(":", 1)[1].strip() for line in out.splitlines() if line.startswith("Dataset ID:"))
    drop = [e["episode_index"] for e in episodes if task not in e["tasks"]]
    left = call("DELETE", f"/dataset/{new_id}/episodes", drop)
    assert all(task in e["tasks"] for e in left), f"{new_id} still has other tasks"
    print(f"{task!r}: dataset {new_id}, {len(left)} episodes")
