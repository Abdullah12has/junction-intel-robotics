"""UNTESTED DRAFT: API names checked against anomalib lib/v2.7.0 source, never run on the station.

Run in the anomalib venv:  python cell/train_export.py [patchcore|efficientad]

data/bricks/train/good    tray A        data/bricks/val/good    tray B
data/bricks/val/defect    tray D1       data/bricks/test_holdout/{good,defect}  trays C, D2 (never given to anomalib)
"""
import sys

import openvino as ov
import torch
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.deploy import CompressionType, ExportType
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import EfficientAd, Patchcore

NAME, SIZE = (sys.argv[1] if len(sys.argv) > 1 else "patchcore"), (256, 256)
assert torch.xpu.is_available(), "torch lacks XPU: install from https://download.pytorch.org/whl/xpu"

dm = Folder(name="bricks", root="data/bricks", normal_dir="train/good",
            normal_test_dir="val/good", abnormal_dir="val/defect",
            test_split_mode=TestSplitMode.FROM_DIR, val_split_mode=ValSplitMode.SAME_AS_TEST,
            train_batch_size=1 if NAME == "efficientad" else 16, eval_batch_size=16, num_workers=4)

if NAME == "patchcore":
    # ponytail: 2% coreset keeps the memory bank small enough for the NPU; raise it if accuracy drops
    model, extra = Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"),
                             coreset_sampling_ratio=0.02), {}
else:  # first run downloads ImageNette (~1.5 GB) and teacher weights
    model, extra = EfficientAd(model_size="small"), {"max_epochs": 20}

engine = Engine(accelerator=XPUAccelerator(), strategy=SingleXPUStrategy(), devices=1,
                default_root_dir=f"results/{NAME}", **extra)
engine.fit(model=model, datamodule=dm)   # validation pass sets the F1 threshold and min/max
engine.test(model=model, datamodule=dm)  # same images as validation: a sanity check, not our score

xml = engine.export(model=model.cpu(), export_type=ExportType.OPENVINO, export_root=f"export/{NAME}",
                    input_size=SIZE, compression_type=CompressionType.FP16)
m = ov.Core().read_model(xml)            # batch axis is exported dynamic; the NPU needs static
m.reshape([1, 3, *SIZE])
ov.save_model(m, str(xml).replace("model.xml", "model_static.xml"))
print("exported", xml)
