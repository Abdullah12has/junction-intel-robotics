"""Make a batch-1 static copy of an anomalib 2.7.0 OpenVINO IR and check it against the source (P2-B-ANOM-LAPTOP, E1).

Why: anomalib 2.7.0 exports PatchCore with a fixed batch of 2 (IR input [2,3,256,256]) and PaDiM with a dynamic batch
(input [?,3,256,256]; the NPU plugin rejects it). OpenVINOInferencer feeds one image per call, so both need batch 1.
This reshapes the source to [1,3,256,256], writes model.xml and model.bin (FP16 weights), then compares every output
on one image on CPU: source fed the image twice (its own batch) against the batch-1 copy. Writes static_ir_check.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image


def image_batch(path, batch):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]
    return np.repeat(arr, batch, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source exported model.xml")
    ap.add_argument("--dst", required=True, help="output directory for the batch-1 IR")
    ap.add_argument("--image", required=True, help="one 256x256 test image for the check")
    ap.add_argument("--src_batch", type=int, default=2, help="batch used to feed the source model")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    report = {"src": str(src), "dst": str(dst), "src_batch_fed": args.src_batch, "openvino_version": ov.__version__}
    try:
        core = ov.Core()
        model = core.read_model(str(src))
        in_name = model.input(0).get_any_name()
        report["source_input_shape"] = str(model.input(0).get_partial_shape())
        report["source_outputs"] = [o.get_any_name() for o in model.outputs]

        ref_compiled = core.compile_model(model, "CPU")
        ref_res = ref_compiled({in_name: image_batch(args.image, args.src_batch)})
        ref_names = [o.get_any_name() for o in ref_compiled.outputs]
        ref_list = [np.asarray(ref_res[o]) for o in ref_compiled.outputs]

        model_b1 = core.read_model(str(src))
        model_b1.reshape({model_b1.input(0).get_any_name(): [1, 3, 256, 256]})
        ov.save_model(model_b1, str(dst / "model.xml"), compress_to_fp16=True)
        report["reshaped_input_shape"] = str(model_b1.input(0).get_partial_shape())

        b1_model = core.read_model(str(dst / "model.xml"))
        b1_compiled = core.compile_model(b1_model, "CPU")
        b1_res = b1_compiled({in_name: image_batch(args.image, 1)})
        diffs = {}
        for idx, o in enumerate(b1_compiled.outputs):
            key = ref_names[idx]
            b1_arr = np.asarray(b1_res[o])
            src_arr = ref_list[idx]
            a = src_arr[:1].astype(np.float64)
            b = b1_arr[:1].astype(np.float64)
            diffs[key] = {
                "src_shape": list(src_arr.shape),
                "b1_shape": list(b1_arr.shape),
                "max_abs_diff_first_sample": float(np.max(np.abs(a - b))) if a.shape == b.shape else None,
                "src_first_sample": a.reshape(-1)[:1].tolist(),
                "b1_first_sample": b.reshape(-1)[:1].tolist(),
            }
        report["cpu_output_comparison"] = diffs
        report["status"] = "ok"
    except Exception as exc:  # record the failure in the report
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    (dst / "static_ir_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
