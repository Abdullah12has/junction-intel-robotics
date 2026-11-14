# Junction 2026 credits: redemption ledger

Checked 13 November 2026, Helsinki time. Source: Junction Discord server search for “credits after:2026-11-01” (40 results, both pages), event Developer Tools and Assets, sponsor redemption pages and the MLH event email. This is a snapshot; offers may be added later. Amounts are advertised values, not cash. No paid purchase or payment-method addition was performed.

## Confirmed actions

- **Codex: redeemed 2,500 usage credits ($100 advertised value).** Junction assigned the individual code, ChatGPT offered a 2,500-credit grant, Redeem was clicked, and Usage now displays 2,500 credits. Valid 30 days after redemption according to Junction. Personal ChatGPT/Codex usage, not API credit and not a paid subscription. Evidence: codex-credits-confirmed.jpg. Source https://hackjunction.app/participate/45/hackerpacks/70 .
- **Featherless: followed Junction $50 coupon; plan activation succeeded; current account balance shows $75.00.** Auto top-up is OFF. The initial $0 was a loading state; do not infer a $75 grant or a measured $50 balance increase. The account had prior usage. No paid top-up confirmed. Evidence: featherless-balance.jpg. Coupon https://featherless.ai/join/feather_request_pricing/JUNCTION26?amount_usd=50 .

- **Modal: $100 Junction grant active**, workspace `abdullah12-hassan7`, expires **30 November 2027**. User created account; claim submitted once and Credits page verified. Evidence: modal-credits-confirmed.jpg.
- **Exa: $50 coupon redeemed**, balance increased from $20 to **$70**. No payment method added. Evidence: exa-credits-confirmed.jpg.
- **Context.dev: 50,000 credits redeemed**, total **51,000**. UI reports **500 concurrent / 10 requests per minute** (different from advertised concurrency). Evidence: context-credits-confirmed.jpg.
- **Aiven: $50 trial active for 30 days**, zero running services. Evidence: aiven-credits-confirmed.jpg.
- **Gemini: one team request submitted**, confirmation recorded; fulfillment pending. Evidence: gemini-request-submitted.jpg.
- **Verda: request submitted and coupon received by email**, `JUNCTION-2026`; account creation and balance redemption still pending. Evidence: verda-request-submitted.jpg.

## Remaining claims and dependencies

| Provider | Advertised benefit | Current state / next dependency |
|---|---|---|
| Claude | $20 API credit per participant | Application submitted: confirmation observed in Chrome. No duplicate submitted. Evidence: claude-application-submitted.jpg. Grant pending, usually 1–2 business days; credits expire two weeks after grant. |
| Verda | €200 compute/inference | Coupon received: JUNCTION-2026. User must finish password signup in Chrome, then redeem. No credited balance verified. |
| CRACI | €1,000 compute per team | Discord directs teams to booth on third floor next to stage for invite code. No public code; no message sent or GitHub app installed. |
| ElevenLabs | About $50, per MLH post | MLH email found, but connector explicitly withholds authentication code. User must finish login in Chrome. No promo obtained yet. |
| Entire | $10, per MLH post | Same MLH sign-in dependency. No promo obtained yet. |
| Tiger Data | $1,000 for 30 days, new accounts only | Skipped at user request. No account created. |
| Vultr | $100 | Official MLH promo link opens an informational page with no credit-request form. Same problem reported in Discord. Requires organizer fix or direct valid form. |
| MongoDB Atlas | $50 via GitHub Student Developer Pack | Authorized read-only GitHub email/profile for student verification; returned to sign-in page, no eligibility result or grant confirmed. |
| Oxylabs | First month of $500 Flexible plan free | Code JUNCTION2026. Requires card and subscription cancellation to avoid subsequent monthly charges. Not activated; recurring commitment requires explicit approval. |
| Featherless Cloud | GPU credit for selected teams | Selection-dependent, amount unspecified. Not automatically eligible; talk to mentors. |
| Hostinger | VPS, AI and scraping resources | Discord says DM Ričardas or visit third-floor booth near Google Cloud. No message sent; eligibility/code not obtained. |

## Modal details for the Intel workflow

Use code **VVX-2QX-8HW** at https://modal.fillout.com/t/qMXCmRGseUus?code=VVX-2QX-8HW . Redemption deadline: 16 November 2026. Enter workspace slug from the dashboard URL, not email or display name. Each workspace once; do not submit duplicates while waiting for processing. Confirmation usually takes a few minutes. Verify in https://modal.com/settings/usage . Payment method required to use GPUs; set spend limit to $0 if you want workloads to stop once free credits run out. Browser signup is separate from backend CLI authentication. Do not put tokens in Git.

Guide: https://modal-com.notion.site/Modal-at-Junction-2026-use-cases-for-every-challenge-4161e7f16949816e9540de36da40a90a

## Claim links

- Claude: https://platform.claude.com/offers/0cfcfd8d-20af-4582-8324-1869c2af61a8
- Exa: https://dashboard.exa.ai/billing
- Context.dev: https://context.dev/dashboard/redeem-credits
- Aiven: https://console.aiven.io/signup
- Gemini: https://forms.gle/VbgQDB4ZiAG9w5io6
- Verda claim: https://fbo8m.share-eu1.hsforms.com/2uaUfG06UQz-VRFHFiMXFOA
- CRACI event instructions: https://hackjunction.app/participate/45/hackerpacks/64
- MLH event offers: https://www.mlh.com/events/junction-40/prizes
- Tiger Data: https://console.cloud.tigerdata.com/signup
- Vultr claim source: https://discover.vultr.com/mlh
- MongoDB student offer: https://www.mongodb.com/students
- Oxylabs event terms: https://hackjunction.app/participate/45/hackerpacks/61
- Featherless selected GPU offer: https://hackjunction.app/participate/45/hackerpacks/52

Non-credit tools also offered: Qualia dataset curation, IronShard artifact storage, Cloudinary free tier, Scailab, Wendy, Arduino hardware. These are not verified cash-equivalent balances and were not provisioned. They should not distract from Intel's required robot/Anomalib/Studio/OpenVINO integration.
