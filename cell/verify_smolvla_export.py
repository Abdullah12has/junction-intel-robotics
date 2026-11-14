"""Run a SmolVLA OpenVINO export on saved observations; never connects to hardware."""

import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from physicalai.inference import InferenceModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.export / "manifest.json").read_text())
    features = [f["init_args"] for f in manifest["model"]["input_features"]]
    cameras = [f["name"].removeprefix("images.") for f in features if f["ftype"] == "VISUAL"]
    rows = pq.read_table(args.dataset / "data/chunk-000/file-000.parquet").slice(0, 3).to_pylist()
    images = {}
    for camera in cameras:
        video = args.dataset / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        with av.open(str(video)) as container:
            images[camera] = []
            for frame in container.decode(video=0):
                images[camera].append(frame.to_ndarray(format="rgb24"))
                if len(images[camera]) == len(rows):
                    break
        assert len(images[camera]) == len(rows)
    report = {"export": str(args.export), "dataset": str(args.dataset), "devices": {}}
    outputs = {}
    for device in ("CPU",):
        started = time.monotonic()
        model = InferenceModel(args.export, backend="openvino", device=device)
        compiled_seconds = time.monotonic() - started
        outputs[device] = []
        timings = []
        try:
            for index, row in enumerate(rows):
                observation = {"task": ["move the brick to the inspection spot"], "state": np.asarray(row["observation.state"], dtype=np.float32)[None],
                               "images": {camera: images[camera][index].transpose(2, 0, 1)[None]
                                          for camera in cameras}}
                started = time.monotonic()
                actions = model.predict_action_chunk(observation)
                timings.append(time.monotonic() - started)
                assert actions.ndim == 2 and actions.shape[1] == 6, actions.shape
                assert np.isfinite(actions).all(), "Nonfinite actions"
                outputs[device].append(actions.copy())
        finally:
            model.close()
        report["devices"][device] = {"compiled_seconds": compiled_seconds,
                                      "inference_seconds": timings, "observations": len(rows),
                                      "output_shape": list(actions.shape), "all_finite": True}
    report["scope"] = "Offline inference only; no robot commands or task-success evaluation"
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
