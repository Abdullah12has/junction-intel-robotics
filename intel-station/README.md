# Intel station source archive

Read-only source capture from the Intel laptop on **15 November 2026**. This complements the main blue-plate training pipeline with the separately developed quality-sorting application and the actual station deployment scripts. No services were restarted and no robot code was executed during collection.

## Layout

| Directory | Original location under `/home/intel/` | Contents |
|---|---|---|
| `workspace/app/` | `intel-junction-2026/app/` | Anomaly detection, motion, control loops, dashboard v5, VLA adapters, perception, inspection, calibration tools and offline tests |
| `workspace/sim/` | `intel-junction-2026/sim/` | Separate digital twin, camera/dynamics fitting and simulation gates |
| `workspace/docs/`, `workspace/submission/` | Corresponding workspace folders | Intel-side design and submission notes |
| `deployment/` | `junction-cell/` | ACT/SmolVLA workers and bounded runners, recording helpers and station-specific variants |
| `dashboard-snapshot/` | `junction-dashboard-ui/` | Files differing from the canonical root `dashboard/` |
| `export-snapshot/` | `intel-junction-2026/export/junction-2026/` | Earlier curated export files differing from `workspace/app/` |
| `research/` | `intel-junction-2026/work/` | Historical experiments, diagnostics, candidate implementations and analyses |
| `operations/`, `setup/` | `junction-handoff/`, `Desktop/prometheus/` | Dated handoff differences and SSH setup script |

Start with [the Intel application overview](workspace/app/README.md) and [its calibration guide](workspace/app/docs/CALIBRATION.md). This application addresses GOOD-to-blue/DEFECT-to-green quality sorting, a different task from the blue-plate imitation-learning campaign. Its original documentation describes intended behavior; this import does not independently verify its task-success claims.

## Provenance and duplicates

[IMPORT-MANIFEST.json](IMPORT-MANIFEST.json) records original paths, SHA-256 hashes, byte sizes, import destinations and exclusions. Exact matches reuse existing repository files; differing versions do not overwrite canonical code. Consequently `deployment/`, `dashboard-snapshot/` and `export-snapshot/` are **partial source snapshots**, not standalone installations. Use the manifest or the reconstruction command below to combine their files with identical canonical copies.

Original agent instruction files, where included, are named `AGENTS.snapshot.md` or `CLAUDE.snapshot.md`; they are historical records, not new repository instructions.

```sh
python scripts/check_intel_snapshot.py
# Reconstruct one captured tree into a NEW directory outside this checkout:
python scripts/materialize_intel_snapshot.py \
  --source-root junction-cell/act-checkpoint-test \
  --output /tmp/intel-act-source
```

Reconstruction copies source only. It does not connect to hardware or install dependencies.

## External requirements and exclusions

The capture excludes credentials, hidden account state, installed environments, dependency repositories, model weights, videos, recordings, generated datasets, logs and most generated experiment output. Superseded workspace copies and bundled third-party packages are indexed as exclusions rather than duplicated. The full filtered capture is backed up outside Git on the development machine.

Calibration JSON files are preserved as observed station-specific evidence. They are not portable defaults and must not be used to replace measured calibration on another arm. Some application code also needs unbundled masks, camera images, fitted binary models, URDF/mesh assets and the original environment. Original relative links to excluded artifacts may not resolve. Do not run `start_all.sh`, motion tools or scripts named `test_*` merely to inspect this archive.

## Validation

[VALIDATION.json](VALIDATION.json) records syntax and inventory checks. `scripts/check_intel_snapshot.py` verifies every imported/reused source hash and parses imported Python without importing modules. This does not establish runtime compatibility or physical safety. Use the main repository's hardware-free test suite for its own tested utilities; native Intel application integration and physical trials remain separate.
