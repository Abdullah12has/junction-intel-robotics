# Verda training handoff

Updated 2026-11-14. User requested replacing blocked Modal with Verda and explicitly authorized an inspection-only SmolVLA pilot on the seven existing episodes.

## Training allocation (released after completion)

- Instance: `7c0a2762-b3ba-4259-a968-c1023b2114d9`, `junction-smolvla-fin03`, FIN-03.
- SSH: `root@95.133.254.44`, dedicated identity `~/.ssh/junction_verda_ed25519` on Mac and Intel laptop, mode 600.
- GPU verified with nvidia-smi: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver 580.178.04. Docker 29.9.3.
- Spot compute $1.418/hour, plus 100 GB OS storage. Spot can be evicted. Retained volumes still incur storage charges.
- Studio target: `b963d065-34f1-4740-8448-4bf9fdca2abd`, Junction Verda RTX Pro 6000. SSH tunnel local port 34131 on laptop, trainer remote loopback 8001. Pilot job `6e96ec9e-0253-4b5b-a4f1-041a7aef7531` completed all 2,320 steps / 10 epochs. Remote job `884288db-392a-480a-9bff-1ada339c5ee9`. Final validation loss 0.27096. OpenVINO export succeeded. Studio target now references released compute and must be updated to a new instance before more training.
- API client: `/Users/abdullah/Desktop/vd/infra/scripts/verda_api.py`. Its credentials stay in its private secrets directory. Never copy secrets into documentation/Git/logs.
- Existing WebUI CPU instances are unrelated; do not modify them.

## Training plan and limits

Use dataset `6d6bdce3-b382-478d-9697-fb9e573b97af`: seven episodes / 4,366 frames, inspection task only, both RealSense cameras. SmolVLA initializes from `lerobot/smolvla_base` in Studio. Planned pilot: 10 epochs, batch 16, BF16 mixed, validation split 0.15 (episode-level), OpenVINO export requested. Verify finite losses and export before robot use. No evidence yet of bowl-task capability or pick/place success.

Studio remote training cannot upload a base checkpoint for resume through its current protocol. Spot recovery is not automatic: preserve the OS volume and checkpoints, then recover explicitly. Do not describe it as verified resumable training.

## Cleanup

Copy artifacts back to laptop and Mac before deleting this dedicated allocation. **Shutdown alone does not stop compute billing.** Delete the instance when finished; explicitly choose whether to retain volumes. Never delete unrelated resources. See https://docs.verda.com/cpu-and-gpu-instances/shutdown-hibernate-and-delete/.

Two empty A100 40GB FIN-01 attempts never accepted SSH, including a restart of the first. Both were deleted with their empty OS volumes: `7d4fd5e3-5d18-42ae-a684-edb7fc57dd78` and `9b3b56d6-5c86-42e9-a743-16f47ad83b41`. FIN-03 is reachable from Mac and laptop.

Git stays local per user request. Modal authentication remains configured but its deployment was blocked by payment requirements.

## Verified result

Model `6540166d-c6c7-4087-ba07-ec5571c37d3b` is stored on the laptop under `~/.local/share/physicalai/models/`. Offline OpenVINO CPU inference passed on three real recorded observations plus the inspection task string. Each output was finite with shape 50x6; latency 1.48–1.59 seconds. No robot commands issued; GPU/NPU inference for this SmolVLA export and physical success remain untested.

Mac backup: `/Users/abdullah/Desktop/junction/station-backups/2026-11-14/verda-smolvla-pilot/`, eight files, 2,511,813,759 bytes. All SHA256 hashes matched the Verda originals. Evidence: `evidence/verda-backup-verification.json`, `evidence/smolvla-verda-export-check.json`, training request/completion JSON and metrics CSV.

GPU compute deletion requested successfully after backups and inference verification. Retained OS disk `2d4e8a6a-85e8-4fa2-a050-b24de1b9eb4e` for recovery; storage billing continues. Recreate a compatible instance from this volume and update Studio SSH hostname for future training. Never assume the old IP remains assigned.

## First physical SmolVLA trial — 2026-11-14

Operator confirmed demonstration starting pose, brick position and clear workspace. GPU offline preflight took 74–111 ms/chunk, but raw actions exceeded shoulder/elbow calibrated range (~−104/+104 normalized units). Added output constraints in normalized graph coordinates using the export's mean/std, mapping back to rotary −99.9..99.9 and gripper 0.1..99.9. Motor limits/calibration unchanged. Original export retained in `exports/openvino/original/`; original Mac backup remains unchanged. First in-place save failed with a bus error; recovered by loading preserved original and saving to a separate file before installing canonical files. Bounded GPU inference passed, finite outputs within calibrated bounds.

Stopped the existing recording session through Studio, confirmed no owners, then ran model control with both cameras at 30Hz, 3s gradual start, 20s duration. Monitoring guarded demonstrated joint/target range and stale feedback. Trial completed20.04s without guard faults, control stopped, and live session query returned an empty list. **Operator reported: moved but missed or dropped the brick. Task success is false.** No repeat motion authorized or performed afterward. This is a failed pick/place trial, not proof of task capability.

Evidence: `smolvla-gpu-preflight.json`, `smolvla-gpu-bounded-preflight.json`, `smolvla-robot-test.json`. Frames remain locally under `/tmp/smolvla-trial-frames/` and laptop `/home/intel/junction-cell/evidence/smolvla-test-frames/`; not committed. Scripts: `cell/test_smolvla_bounded_export.py`, `cell/bound_smolvla_export.py`. The latter rebuilds from already-preserved originals; do not use it against a different model's normalization.

## Requested 60-second repeat — 2026-11-14

Repeated the constrained SmolVLA GPU policy for60.05seconds at30Hz with3s gradual start, both cameras, stale-feedback and demonstrated-range guards. Started from the previous trial's ending pose, not a reset HOME. No guard fault; control stopped and subsequent live session list was empty. No operator outcome yet; do not claim pick/place success. Evidence: `evidence/smolvla-robot-60s-test.json`; script `cell/test_smolvla_60s.py`.

### Second 60-second repeat

User requested another repeat. Completed60.04s from current pose, same GPU export and guards, no range/feedback fault. Control stopped and subsequent session query empty. Task success remains unconfirmed. Evidence `evidence/smolvla-robot-60s-repeat2.json`; remote script `/home/intel/junction-cell/test_smolvla_60s_repeat2.py`.
