# Blue plate training — 15 November 2026

## Idle GPU deleted; checkpoints retained — 15 November, 08:10 Helsinki

At the user's request, deleted idle normal-image A100 instance `4b37dcdc-fec5-4f51-8070-17d1ecafe3d3` (former IP `86.38.238.100`). API deletion succeeded with explicit `volume_ids: []`; subsequent inventory confirms the instance is absent and its 400 GB OS volume `cfd3e072-b633-4446-8983-7432c5090071` is **detached**, with no instance attached. Compute billing of $1.699/hour has ended. Retained storage continues at $0.109589/hour ($80/month).

All normal-image checkpoints remain on that disk at `/root/blue-plate/campaign/runs/`: ACT steps 10k–50k every 10k; SmolVLA steps 2k–16k every 2k. Logs, campaign and software remain on the same disk. Restore/attach this existing FIN-02 volume to compatible compute only when needed; do not assume the old IP remains usable. Do not delete the recovery volume.

Local evidence: `../../station-backups/2026-11-15-idle-gpu/` contains the full remote SHA256 manifest, `idle-handoff.tgz` (logs, experiment specifications, campaign JSON and code), deletion request and verification. The extra local checkpoint transfer was cancelled to end compute billing sooner: **local checkpoint backup is incomplete** (19 files verified, 132 not copied, zero mismatches). Use the retained disk as the complete source, not the partial local `runs/` tree.

The marker ACT host `86.38.182.164`, instance `e25219b3-dea2-4991-9cdd-ad2b2d37e2af`, remains running and was not modified. This supersedes earlier references below to both servers being allocated.

## Single selected training — 15 November, 08:04 Helsinki

User narrowed the selection to one. **Only `act-X-A0-R-seed0` continues**, on marker host `86.38.182.164`; its existing trainer was not restarted. Choice is provisional: strongest observed ACT validation and explicit target markers, without proven physical success.

Stopped both normal-image trainers (`act-N-A0-R-seed0`, `smolvla-N-A0-R-seed0`) and their workers on `86.38.238.100`. Stopped the normal shard service and retired the old paused original supervisor without resuming its stale queue. Normal manifest now has no experiments/adoptions; its status records intentional stops. All existing checkpoints and logs remain. Both queues are empty. This supersedes the three-candidate section below.

The normal host is now idle but still allocated and billing; no instance or storage was deleted. Marker host continues training. Do not restart either retired normal/original scheduler from historical instructions.

## Reduced to three candidates — 15 November, 07:46 Helsinki

User requested keeping only the three experiments most likely to succeed. The active shard manifests now contain only:

- `smolvla-N-A0-R-seed0`: normal-image SmolVLA; keep its saved `checkpoints/002000` candidate (best observed validation loss 0.3417). Later validation worsened despite lower training loss, suggesting overfitting. Continuing training does not mean the final checkpoint should be selected.
- `act-X-A0-R-seed0`: marker-conditioned ACT; best observed validation loss 0.4570 at step 32,000.
- `act-N-A0-R-seed0`: normal-image ACT; best observed validation loss 0.4685 at step 36,000, retained as the simpler control.

ACT validation is every 2,000 steps but regular checkpoints are every 10,000; the best validation steps above are observations, not a claim that those exact weights were saved. Check available checkpoint directories before evaluation. Losses across ACT and SmolVLA are not directly comparable. This selection is a provisional judgment, not verified physical performance.

Stopped `smolvla-X-A0-R-seed0` and its data-loader processes. Removed all 12 queued A1/A2/A3 variants from dispatch. Preserved every existing checkpoint/log and all experiment specifications. The three selected trainers continued with their original PIDs; neither their training state nor immutable code was changed. Both shard services were restarted with explicit adoption manifests, and both queues were verified empty. The older paused original supervisor remains paused; do not resume it.

Remote selection records: `/root/blue-plate/campaign/selection.json`. Recovery snapshots: `/root/blue-plate/selection-1794721576` on the normal host and `/root/blue-plate/selection-1794721586` on the marker host. Current status remains `shard_status.json`. The earlier eight-job manifests and launch counts below are historical and superseded by this section. Both cloud instances remain allocated and billing; no compute or storage was deleted.

## Two-machine split (updated 04:07 UTC)

The authoritative queue on each host is now `/root/blue-plate/campaign/shard_status.json`. The old `queue_status.json` is a frozen historical snapshot and must not be used for current status.

- Original 40 GB A100: `root@86.38.238.100`, service `junction-blue-plate-normal.service`, eight N variants. Existing baseline training PIDs were adopted without restarting.
- New 80 GB A100: `root@86.38.182.164`, instance `e25219b3-dea2-4991-9cdd-ad2b2d37e2af` on the second account (Verda CLI default profile), FIN-03. Service `junction-blue-plate-marker.service`, eight X variants. GPU preflight passed and X-marker training started at approximately 04:07 UTC (06:07 Helsinki). On-demand $2.386/hour plus 250 GiB storage.
- Each host runs two jobs concurrently. The two manifests are disjoint and together cover all 16 R experiments. M experiments remain blocked.
- The original `junction-blue-plate.service` supervisor PID 259858 is intentionally SIGSTOP-paused to prevent dispatching its stale queue; its two training children continue. **Do not resume or restart this old supervisor.** The normal shard retires it after its work finishes.
- Shard code is `/root/blue-plate/shard_queue.py`, outside the frozen training code. Manifests are `/root/blue-plate/normal-shard.json` and `/root/blue-plate/marker-shard.json` on the respective hosts. Status records distinguish running, queued and finished jobs. No automatic service restart is configured. Stopping the normal shard service alone does not stop the adopted baseline workers in the old service.
- Both hosts use the existing dedicated SSH identity. No credentials copied to either server. Combined compute $4.085/hour, storage approximately $0.178/hour: **about $4.26/hour total** while both remain allocated. Deletion is required to end compute billing.

## Original single-host launch record

Campaign service started at approximately 03:39 UTC / 05:39 Helsinki. Initial jobs are `act-N-A0-R-seed0` (100,000 steps, batch 8) and `smolvla-N-A0-R-seed0` (20,000 steps, batch 32); the other 14 real-only variants queue automatically.

Dedicated Verda instance `4b37dcdc-fec5-4f51-8070-17d1ecafe3d3`, hostname `junction-blue-plate-20261115`, FIN-02, A100 SXM4 40 GB, 22 CPU cores, 120 GB RAM. On-demand compute $1.699/hour; 400 GB OS volume approximately $0.11/hour. Volume `cfd3e072-b633-4446-8983-7432c5090071`. Unrelated Verda resources were not changed.

SSH `root@86.38.238.100`, identity `/Users/abdullah/.ssh/junction_verda_ed25519`. Re-query instance before reconnecting after deletion/recreation; IPs are not permanent. No credentials are stored in this document.

Remote paths:

- `/root/blue-plate/campaign`: transferred campaign and outputs; all 1,046 bundle hashes verified before training.
- `/root/blue-plate/code`: deployment source, including shared-GPU queue and CUDA preflight.
- `/root/blue-plate/venv`: Python 3.12; torch 2.11.0+cu130 and LeRobot 0.6.0. All four validated LeRobot source hashes matched.
- `/root/blue-plate/campaign/queue_status.json`: current running/queued/finished IDs.
- `/root/blue-plate/campaign/logs/`: individual native trainer logs.
- `/root/blue-plate/queue.log`: supervisor output.
- `/root/blue-plate/campaign/runs/`: check experiment output_dir for exact checkpoint paths.

Service `junction-blue-plate.service` runs 16 real-only experiments, maximum two processes on GPU 0. ACT/SmolVLA start interleaved; whichever slot finishes takes the next experiment. Six loader workers per process. Actual batches, optimizer steps, grouped splits and augmentation definitions are unchanged. The 16 simulation runs remain excluded by the simulation compatibility gate.

Read-only monitoring:

```bash
systemctl status junction-blue-plate.service
cat /root/blue-plate/campaign/queue_status.json
nvidia-smi
```

Stopping the service stops its training processes. It does not stop Verda billing. Do not restart the service over existing outputs: use the documented per-experiment `run.py --resume CHECKPOINT` path after checking the latest complete checkpoint. The service intentionally has no automatic restart or boot enablement, to avoid overwriting or duplicating runs.

The queue ends after all experiments finish, but the instance remains allocated. Preserve checkpoints before deleting compute; shutdown alone does not stop compute charges. Retained volumes continue storage billing. No cloud credentials were copied to the training machine.

GPU preflight uses disposable optimizer updates and checkpoint writes, separate from campaign runs. Model quality is not established by finite training loss; real held-out evaluation is still needed before deployment.

Launch verification at approximately 03:41 UTC: ACT past step 791, SmolVLA past step 224; latest logged losses 2.691 and 0.421 respectively, both finite, no tracebacks. Two running / 14 queued. Concurrent GPU usage approximately 13.6 GB. These are startup checks, not completed-run results.
