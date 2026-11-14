# Anomalib Intel GPU installation recovery

Verified 2026-11-14 on the Intel laptop via Tailscale `100.112.85.77`.

The failed `anomalib[xpu]` install is no longer blocking: the separate environment `/home/intel/junction-cell/anomalib-env` already contained a completed fallback install (`install.log` ended with `INSTALL_DONE`). No packages were reinstalled during verification and Studio's environment was not modified.

Verified versions: Python 3.12.16, torch 2.15.0+xpu, torchvision 0.29.2+xpu, anomalib 2.7.0, OpenVINO 2026.4.1. The installed XPU stack includes `triton-xpu` 3.9.0. Do not try to solve the missing legacy extra dependency by installing arbitrary similarly named packages.

## Recreate the fallback in a separate environment

Use the official PyTorch XPU index explicitly for PyTorch. Installing an Anomalib extra does not itself ensure pip/uv uses that index.

```sh
uv venv --python 3.12 anomalib-env
uv pip install --python anomalib-env/bin/python \
  --index-url https://download.pytorch.org/whl/xpu \
  'torch==2.15.0+xpu' 'torchvision==0.29.2+xpu'
printf '%s\n' 'torch==2.15.0+xpu' 'torchvision==0.29.2+xpu' > xpu-constraints.txt
uv pip install --python anomalib-env/bin/python \
  --constraint xpu-constraints.txt 'anomalib[openvino]==2.7.0'
uv pip check --python anomalib-env/bin/python
```

These are recovery commands based on the verified installation, not a fresh clean-install test. The constraints preserve the XPU builds when resolving Anomalib dependencies. The full installed snapshot is on the laptop at `/home/intel/junction-cell/evidence/anomalib-verified-packages.txt`.

## Verification results

- `uv pip check`: all 112 installed packages compatible.
- Imports passed: Anomalib Engine, SingleXPUStrategy, XPUAccelerator, Patchcore and OpenVINOInferencer.
- `torch.xpu.is_available()`: True.
- Device: Intel(R) Graphics [0xb080].
- A float32 matrix multiplication on XPU completed, synchronized and matched the CPU result.
- OpenVINO enumerated CPU and GPU. **NPU was not listed** in this environment; do not use the draft inspection default `NPU` without separately resolving that issue.
- Only deprecation warnings were emitted by imports. Full training/export was not rerun during this check.

Official installation reference: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html

## NPU follow-up

NPU inference now works through `anomalib-env/bin/python-npu`; see [NPU recovery](NPU-SETUP.md). This supersedes the earlier NPU enumeration limitation for processes using that launcher.

## System installation completed (2026-11-14)

After successful sudo authentication, installed `intel-driver-compiler-npu` and `intel-level-zero-npu` v1.38.0 plus six distribution dependencies. Waited for an independently initiated apt upgrade to finish; did not interrupt it or reboot. Plain Anomalib environment Python with `LD_LIBRARY_PATH` explicitly removed now lists CPU/GPU/NPU and executes the existing PatchCore smoke export on NPU with finite outputs. The `python-npu` wrapper is no longer required. This supersedes the process-local-only limitation above. Credentials were not saved. Defect accuracy and sustained performance remain unverified.
