#!/usr/bin/env python3
"""E1 P4-OVBENCH: FP16-compressed copy of an IR so every model is benchmarked at the same weight precision.
usage: prep_fp16.py IN_XML OUT_XML (prints one JSON line)"""
import json
import os
import sys

import openvino as ov

SRC, DST = sys.argv[1], sys.argv[2]


def counts(path):
    text = open(path).read()
    return {"f16": text.count('element_type="f16"'), "f32": text.count('element_type="f32"')}


os.makedirs(os.path.dirname(DST), exist_ok=True)
info = {"src": SRC, "dst": DST, "src_counts": counts(SRC)}
model = ov.Core().read_model(SRC)
ov.save_model(model, DST, compress_to_fp16=True)
info["dst_counts"] = counts(DST)
info["dst_bin_mb"] = round(os.path.getsize(DST.replace(".xml", ".bin")) / 1e6, 1)
print(json.dumps(info), flush=True)
