# Documentation

## Submission

- [Submission overview](SUBMISSION.md): implementation, results and limitations.
- [Code map](CODE-MAP.md): component entry points and verification boundaries.
- [Evidence index](../evidence/README.md): dated measurements and experiment records.
- [Training pipeline](../training/README.md): data preparation and experiment structure.

## Model and data research

- [Blue-plate processing plan](BLUE-PLATE-VLA-PROCESSING-PLAN.md)
- [Parallel experiment design](BLUE-PLATE-PARALLEL-EXPERIMENTS.md)
- [MuJoCo synthetic-data research](MUJOCO-SYNTHETIC-DATA-RESEARCH.md)
- [Action-limit audit](POLICY-ACTION-LIMIT-AUDIT.md)

## Historical planning

- **[Execution plan](EXECUTION-PLAN.md): original implementation plan.** Supersedes the build order in the two documents below.

- [Researched speed and reliability plan](Intel-speed-and-reliability-research.md): November 14 model decisions, Gemma assessment, optimizations, experiments and execution gates.

- [Winning plan](Intel-winning-plan.md): challenge, judging, schedule, evaluation and submission.
- [Build priorities](Intel-repo-priorities.md): proposed implementation order and gaps at the time of review.
- [Official setup guide](JUNCTION2026-official-setup.md): saved event instructions. Its relative `application/` links refer to the [upstream junction2026 repository](https://github.com/open-edge-platform/physical-ai-studio/tree/junction2026), not this repository.

## Official PDFs and Markdown versions

| Document | Original | Markdown text |
|---|---|---|
| Intel kickoff presentation, 15 pages | [PDF](Intel_Physical_AI_Hackathon.pdf) | [Markdown](Intel_Physical_AI_Hackathon.md) |
| Challenge terms, 1 page | [PDF](Ignite_Next_Challenge_Terms_Junction_2026.pdf) | [Markdown](Ignite_Next_Challenge_Terms_Junction_2026.md) |
| Original download alias, identical to the kickoff presentation | [j.pdf](j.pdf) | [j.md](j.md) |

Markdown versions preserve extracted text page by page. Consult the PDFs for diagrams, images and visual layout; the extractions are not manually reconstructed slide designs.

## Station operation

- [Verda CLI access and GPU availability](VERDA-CLI-HANDOFF.md)

- [Station handoff](ROBOT-STATION-HANDOFF.md)
- [Station runbook](ROBOT-STATION-RUNBOOK.md)
- [Dashboard instructions](ROBOT-DASHBOARD.md)

These are dated operational snapshots. Recheck live sessions, network addresses and hardware state before acting. Dashboard source is now included in [`../dashboard/`](../dashboard/README.md). The original running working copy remains outside the repo.

## Credits and resources

- [Shareable redemption guide](Junction-2026-credits-for-friends.md)
- [Redemption status ledger](Junction-credit-status.md)

Credit evidence screenshots remain outside this repository in `/Users/abdullah/Desktop/junction/`; some show personal account information. The ledger records observations at the time of each claim, not continuously refreshed balances.

## Live official sources

- [Intel challenge](https://hackjunction.app/participate/45/challenges/330)
- [Participant guide](https://hackjunction.app/participate/45/guide)
- [Intel Discord](https://discord.com/channels/1268184974784336004/1564622436773658624)
- [Upstream event setup](https://github.com/open-edge-platform/physical-ai-studio/blob/junction2026/JUNCTION2026.md)

Documents were collected November 13–14, 2026. Later organizer instructions may supersede saved rules and deadlines.
