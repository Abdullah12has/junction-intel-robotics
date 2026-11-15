# Modal training target — 14 November 2026

Modal CLI login completed as `intel` on the Intel laptop. Token verification succeeded for workspace `abdullah12-hassan7`; credentials remain only in `/home/intel/.modal.toml`. Do not copy that file into backups or Git. The user explicitly approved authorizing the laptop.

Studio target: `57e65864-ac75-4af2-b4d9-0af5246c6023`, **Junction Modal NVIDIA trainer**.

- Environment: `main`
- Modal app: `physicalai-974fa8e742ef4432b1345e855a13e8da`
- App ID: `ap-mKLssXMIdHLkHMGPj1Wz53`
- Function: `train`
- Volume: `physicalai-training-974fa8e742ef4432b1345e855a13e8da`
- Image: `ghcr.io/open-edge-platform/physicalai-trainer-cuda:main`
- Default GPU in the bundled trainer: A10G
- No Hugging Face secret configured; no training job submitted.

Deployment was requested through Studio's supported remote-trainer endpoint after successful login. The first image build succeeded in242.47seconds (image im-WeZ82LOmWFQ5Vugzj5Qhwz), but deployment failed afterward. A diagnostic retry against the same named target returned: `Please add a payment method to use A10G GPU functions.` Check `/api/remote-trainers/57e65864-ac75-4af2-b4d9-0af5246c6023/health` for readiness. Do not create duplicate targets while this build is active.

The Modal dashboard shows a payment-method verification notice and separately shows $100 credits. This does not establish what compute is available; deployment/training responses are authoritative. No payment information was entered or charge authorized.

Studio currently reports this target degraded. After payment verification, retry the same named deployment; Studio may retain process-local failure state until its deployment status is cleared or backend restarted safely. Do not restart an active robot session. No training was submitted.
