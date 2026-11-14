# Intel NPU recovery — 14 November 2026

## Working invocation on the laptop

```sh
/home/intel/junction-cell/anomalib-env/bin/python-npu your_script.py
```

Or, after activating the Anomalib environment, use `python-npu` instead of `python`. The launcher forwards all Python arguments and loads process-local driver/compiler libraries. Ordinary `python` without these library paths still does not expose NPU; this is not a system-wide install.

## Diagnosis and fix

Panther Lake NPU `8086:b03e`, kernel `intel_vpu`, `/dev/accel/accel0`, render-group access and August 2026 firmware were already working. OpenVINO's NPU plugin was present. The missing components were the Intel Level Zero NPU user driver and compiler.

Downloaded official Intel Linux NPU driver v1.38.0 for Ubuntu 26.04 and verified the package signatures against release fingerprint `EA267657A608300C296B8F8AD52C9665A4077678`. Extracted compiler and Level Zero packages under `/home/intel/junction-cell/npu-driver-1.38/runtime`. The launcher adds `runtime/usr/lib/x86_64-linux-gnu` and the Anomalib environment's `lib` (existing TBB runtime) to `LD_LIBRARY_PATH` for that process only. Existing Level Zero loader 1.28.2 worked in the checks below; Intel's release reference configuration lists 1.32.0.

System installation was attempted but sudo rejected the previously supplied password. No apt installation, firmware replacement, driver reload, reboot or Studio restart was performed. The process-local launcher is the working alternative.

Official package source and reference installation: https://github.com/intel/linux-npu-driver/releases/tag/v1.38.0

## Verified

- OpenVINO 2026.4.1 lists `CPU`, `GPU`, `NPU` through the launcher.
- NPU name: Intel(R) AI Boost.
- A small ReLU graph compiled and executed explicitly on NPU, matching NumPy output.
- Existing smoke PatchCore export `/home/intel/junction-cell/smoke/export/patchcore/weights/openvino/model.xml`, shape `[1,3,256,256]`, compiled explicitly for NPU in 2.04 seconds and inferred finite outputs. `EXECUTION_DEVICES` reported `NPU`.

The PatchCore check used a zero input and the existing smoke model: it proves compilation/execution, not defect accuracy, representative latency or sustained operation. No robot devices or camera sessions were opened. Do not switch production model defaults solely on this smoke test.

## Optional system installation

For default detection without the launcher, an operator with working sudo credentials can install the prepared compiler/Level Zero packages and their distribution dependencies:

```sh
cd /home/intel/junction-cell/npu-driver-1.38
sudo apt-get install ./intel-driver-compiler-npu*.deb ./intel-level-zero-npu*.deb
```

The dry run proposed eight new packages and no removals/upgrades. Recheck the transaction before installing; then verify using plain environment Python. No system-wide installation has been claimed here.
