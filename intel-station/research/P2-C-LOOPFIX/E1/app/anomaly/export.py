"""Export a trained anomalib 2.7.0 model to OpenVINO IR: FP16, static 256x256, batch dynamic (P2-B-ANOM-LAPTOP, E1).

No INT8 (orchestrator note: INT8 gave degenerate scores in sandbox builds). Writes <out>/weights/openvino/model.xml
and model.bin, plus <out>/export_summary.json with file sizes and sha256 hashes.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["padim", "patchcore"], required=True)
    ap.add_argument("--run", required=True, help="training run directory holding the .ckpt")
    ap.add_argument("--out", required=True, help="export_root, e.g. models/anomaly/<model>")
    ap.add_argument("--ckpt", default="", help="optional explicit checkpoint path")
    args = ap.parse_args()

    import anomalib
    import openvino as ov
    from anomalib.deploy import CompressionType
    from anomalib.engine import Engine
    from anomalib.models import Padim, Patchcore

    run, out = Path(args.run), Path(args.out)
    if args.ckpt:
        ckpt = Path(args.ckpt)
    else:
        ckpt = sorted(run.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)[-1]
    model_cls = Padim if args.model == "padim" else Patchcore
    model = model_cls.load_from_checkpoint(str(ckpt), map_location="cpu")

    out.mkdir(parents=True, exist_ok=True)
    exported = Engine().export(
        model=model,
        export_type="openvino",
        export_root=out,
        input_size=(256, 256),
        compression_type=CompressionType.FP16,
    )

    xml = out / "weights" / "openvino" / "model.xml"
    bin_file = out / "weights" / "openvino" / "model.bin"
    summary = {
        "model": args.model,
        "checkpoint": str(ckpt),
        "compression": "FP16",
        "input_size": [256, 256],
        "export_return": str(exported),
        "ir_xml": str(xml),
        "ir_bin": str(bin_file),
        "xml_bytes": xml.stat().st_size,
        "bin_bytes": bin_file.stat().st_size,
        "xml_sha256": sha256(xml),
        "bin_sha256": sha256(bin_file),
        "anomalib_version": anomalib.__version__,
        "openvino_version": ov.__version__,
    }
    (out / "export_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
