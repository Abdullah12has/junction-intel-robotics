import json

import openvino as ov

core = ov.Core()
out = {"openvino": ov.__version__, "available_devices": core.available_devices, "full_device_name": {}}
for dev in core.available_devices:
    try:
        out["full_device_name"][dev] = str(core.get_property(dev, "FULL_DEVICE_NAME"))
    except Exception as exc:
        out["full_device_name"][dev] = "ERROR " + type(exc).__name__ + ": " + str(exc)[:300]
print(json.dumps(out, indent=1))
