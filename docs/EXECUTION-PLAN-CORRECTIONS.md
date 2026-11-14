# Corrections to Claude's execution plan

Reviewed 14 November 2026 (Helsinki). This document and [corrected plan copy](EXECUTION-PLAN-CORRECTED.md) are additions. The original plan and research are unchanged. This review inspected source; it did not operate the arms, run model inference, or independently revalidate external research/current event rules.

## Findings and required corrections

| Priority | Evidence in draft | Failure / correction |
|---|---|---|
| P1 | `cell/run_cell.py:44–56, 78` | HOME can be accepted from stale telemetry or without ever leaving HOME. Require fresh named-joint telemetry, finite values, departure and stable return; supervise receiver failures. |
| P1 | `cell/run_cell.py:90,104` | Both callers ignore skill timeout. Abort the cycle on timeout; do not proceed to classify or place. Retry only from a verified valid starting state. |
| P1 | `cell/run_cell.py:131–134` | Exceptions bypass the trial CSV. Persist attempt start and finalize errors/aborts in cleanup; preserve unknown outcomes. |
| P1 | `cell/inspection.py:18–45` | Camera resolution is not set; ROI ends at 780×540 despite verified 640×480 profile. Resolve stable camera identity, set/read back profile, reject out-of-bounds crops. Five dropped frames do not establish freshness; bound and timestamp acquisition. |
| P1 | Execution plan Phase 6 | Removing a brick during motion is an unsafe test procedure. Inject failures offline, then arrange physical test conditions before motion with hands clear. |
| P2 | `cell/run_cell.py:54,62,131` | Late ACK after timeout can fail the receiver; blocking input/capture/inference stalls asyncio processing. Clean up ACK futures, supervise pump, isolate blocking work and bound stop cleanup. |
| P2 | `cell/run_cell.py:106–111` | Empty inspection spot does not prove correct bowl placement. Distinguish commanded destination, observed placement and human-graded success. |
| P2 | Execution plan Phase 5 | Bad motion is not proof of an FP16 bug. Validate finite outputs and compare identical CPU/GPU inputs; investigate configuration and training too. |
| P2 | `cell/ov_prove.py` | Synthetic random inputs do not validate policy quality; replacing all dynamic dimensions with 1 can benchmark the wrong shape. Use representative correctly typed inputs and explicit validated shapes. |
| P2 | Execution plan energy section | Package energy coverage is platform-dependent. Verify domain coverage before attributing NPU energy or reporting system energy. |

## What remains useful

Retain the fixed inspection spot, learned stages, balanced language tasks, ACT fallback, early training/export smoke test, separate inspection camera, held-out evaluation and bounded recovery. Preserve the existing Studio hardware runtime and avoid competing ownership.

## Acceptance before autonomous operation

Implement the corrected plan's controller gate and offline failure tests. Then verify live message shapes, camera geometry/freshness, stop acknowledgement and one supervised skill. Only afterward expand to full cycles and dataset collection. Filling placeholders or passing syntax checks is insufficient.

## Scope

The corrected copy incorporates the findings and adds an explicit acceptance gate. It does not repair the scripts; they remain untested drafts. Original plan SHA-256: `485ea1e61a6904e813318063d6ca2fb96d3c5ea948552cfb0dd6394236659ed9`.
