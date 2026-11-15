# Blue-plate 32-run campaign: Verda cost and availability

Checked 15 November 2026, approximately 02:53 UTC / 04:53 Helsinki. This is a read-only price check, not a reservation or approved purchase. No instances, volumes, or paid jobs were created.

## Current account-visible rates

Authenticated catalog GET requests through the existing Verda client returned the following USD rates. These are **whole-instance hourly prices**, using `location_prices`; all three listed Finland locations had the same price for these types. Public live pricing agrees with the single-GPU rates. Search snippets containing lower rates were stale and were not used. Sources: [Verda pricing](https://verda.com/pricing), [API reference](https://api.verda.com/v1/docs).

| Instance type | GPU memory | CPUs / host RAM | On-demand/hour | Spot/hour | Capacity snapshot |
|---|---:|---:|---:|---:|---|
| `1RTXPRO6000.30V` | 1 × 96 GB | 30 / 90 GB | $2.836 | $1.418 | Unavailable |
| `1RTXPRO6000.30V.CC` | 1 × 96 GB | 30 / 90 GB | $2.893 | $1.446 | Listed in FIN-03 |
| `8RTXPRO6000.240V` | 8 × 96 GB | 240 / 720 GB | $22.69 | $11.34 | Unavailable |

The availability endpoint listed only the single CC type in FIN-03, with no types in FIN-01 or FIN-02. Per-type GET checks independently returned false for the normal single and eight-GPU types and true for single CC. This does not identify spot versus on-demand stock, reserve capacity, establish the number of available single instances, or guarantee deployment. Catalog presence alone is not stock availability. Recheck immediately before any subsequently authorized provisioning.

An eight-GPU node costs almost exactly eight ordinary single GPUs. It primarily saves elapsed time through parallel trials; it does not create an inherent compute discount. CC is approximately 2% more expensive than the ordinary single.

## What the campaign would cost

The proposed matrix contains 16 SmolVLA runs at **20,000 updates each** and 16 ACT runs at **100,000 updates each**. Let `s` be measured hours per SmolVLA run, `a` hours per ACT run, and `W = 16s + 16a` be useful GPU-hours. These formulas assume each run fits and uses one GPU. Run independent experiments on separate GPUs; do not silently change batch size or switch each experiment to distributed training.

- Available single CC: compute = `$1.446 × W` spot or `$2.893 × W` on-demand, before setup, idle time and retries.
- Eight-GPU node: compute = `$11.34 × node-hours` spot or `$22.69 × node-hours` on-demand. Ideal fully occupied node time is `W / 8`; actual time includes startup, data loading, contention, and the final partly occupied batch.
- Single ordinary RTX: `$1.418 × W` spot or `$2.836 × W` on-demand, if available later.

The only local throughput anchor is an earlier **15,000-step, batch-32 SmolVLA run taking 51 minutes on one RTX PRO 6000 CC**, documented in [VERDA-TRAINING.md](VERDA-TRAINING.md). That older dataset and pipeline differ from the proposed campaign. It is a rough reference, not a benchmark for these configurations. ACT runtime has not been measured. Different step counts scale these calculations only approximately; augmentations, cameras, decoding, batch size, checkpointing, and evaluation can change throughput.

Scaling the historical 15,000-update SmolVLA duration to the planned 20,000 updates gives `0.85 × 20,000 / 15,000 = 1.1333 hours` (68 minutes) per run, **if throughput is unchanged**. This scaling is an assumption, not a measurement. For sensitivity analysis only, hold SmolVLA at that scaled reference and vary the unknown runtime of each 100,000-update ACT run:

| Assumed ACT hours/run | Total GPU-hours | Ideal eight-GPU hours | Eight-GPU spot | Single CC spot | Single CC on-demand |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 26.13 | 3.27 | $37.04 | $37.79 | $75.60 |
| 1.0 | 34.13 | 4.27 | $48.38 | $49.36 | $98.75 |
| 2.0 | 50.13 | 6.27 | $71.06 | $72.49 | $145.04 |

These ACT durations are illustrative assumptions, not measured estimates or bounds. If the 20,000-update SmolVLA runs instead take twice the scaled reference (2.2667 hours each) and the 100,000-update ACT runs take two hours each, useful work reaches 68.27 GPU-hours: approximately $98.71 single-CC spot or $197.50 on-demand. Setup/retries/idle time and storage are additional. Each extra hour left running costs the full instance hourly rate; eight-GPU idle time is especially expensive. Taxes, account credits, transfer charges if any, and existing retained volumes are not included.

**Recommendation:** retain the 32 configs, then, after separate purchase authorization, benchmark one representative run of each model before committing to the full schedule. Measure steady-state steps/second plus startup/checkpoint time and update this formula from the actual step counts. Prefer independent single-GPU jobs with a queue that fills free GPUs. Eight-GPU capacity is currently absent; a single CC is the only available type visible in the snapshot. Spot budgeting must allow retries and verify resume behavior rather than assuming checkpoint recovery already works.

## Billing and storage

Verda's current billing documentation says Pay As You Go is prepaid in 10-minute increments, with unused time refunded within the next billing period after termination. This is not an assertion that ten minutes is an unrecoverable minimum. [Pricing and billing](https://docs.verda.com/welcome-to-verda/pricing-and-billing/).

**Shutdown continues charging.** Delete the instance to release compute; retained volumes remain billable. Hibernation has been removed, and deleting compute while keeping storage is the documented equivalent. Do not delete data before verified backup. [Shutdown and delete](https://docs.verda.com/cpu-and-gpu-instances/shutdown-hibernate-and-delete/).

The authenticated volume catalog reports NVMe and shared NVMe at $0.20 per provisioned GB-month, and HDD at $0.05. The public pricing page labels its storage units GiB-month; preserve the provider's provisioning units when quoting a final order. The API also supplies NVMe `cps_per_gb = 7.6103500761035e-08`, yielding these approximate NVMe charges:

| Provisioned size | Monthly rate | 10 hours | 24 hours |
|---:|---:|---:|---:|
| 100 provider units | $20 | $0.27 | $0.66 |
| 300 provider units | $60 | $0.82 | $1.97 |

Dataset/video/checkpoint sizes must determine the final disk allocation. These are examples, not a proposed volume size. Retaining a 300-unit disk for a month costs $60 even after training ends. Existing retained storage is separate from the incremental campaign cost. [Public storage pricing](https://verda.com/pricing), [volume API](https://api.verda.com/v1/docs).

## Reproducible evidence

- `evidence/blue-plate-campaign/verda-pricing-availability.json`: complete account-visible instance catalog and location availability, with timestamp.
- `evidence/blue-plate-campaign/verda-storage-and-capacity.json`: volume price catalog and independent per-type capacity booleans, with timestamp.
- `evidence/blue-plate-campaign/verda-cost-scenarios.json`: formulas, assumptions, and computed scenarios.

Operational calls were GET `/instance-types`, `/instance-availability`, `/volume-types`, and `/instance-availability/{type}`. The existing client obtains an OAuth token internally; credentials/tokens were neither printed nor saved in evidence. No resource mutation was requested.

## Storage sizing after implementation

The prepared datasets occupy approximately 1.7 GiB before packaging, but checkpoints dominate disk usage. Local model initialization measured 450,046,176 SmolVLA parameters, of which 99,880,992 are trainable, and 51,597,190 ACT parameters. With FP32 weights and two FP32 Adam moments, a rough full-checkpoint payload is 2.60 GB for SmolVLA and 0.62 GB for ACT, before metadata or implementation overhead. The current schedule retains ten checkpoints per run: about **516 GB across 32 runs**. This is a size calculation, not a measured CUDA checkpoint inventory.

Provision roughly **750 provider storage units** if retaining every scheduled checkpoint, allowing headroom for assets, temporary files and backups. At the current $0.20/unit-month catalog rate this is about $150/month, or approximately **$0.2055/hour** from the API per-second rate. For the illustrative single-GPU compute durations of 26.13–68.27 hours, storage adds approximately **$5.37–14.03** while attached/retained for that interval. Thus the earlier sensitivity scenarios become roughly **$43–113 spot or $81–212 on-demand including this disk allocation**, still excluding setup/idle/retry time, tax and transfer. Longer retention continues billing.

Reducing retention to selected checkpoints could lower this substantially, but the prepared runner does not silently delete checkpoints. No storage was purchased. Before launch, measure a representative full checkpoint and revise the disk size, throughput and total quote together.
