# Verda CLI access handoff

Recorded 15 November 2026 (Europe/Helsinki). Availability below is a snapshot, not a reservation.

## Configured account and host

- Account: `abdullah.hassan@talentadore.com`; console project **My Project**, ID `7c50e8a6-467e-48c3-90e6-e32f46862351`.
- Configured on Abdullah’s Mac, user `/Users/abdullah`; not installed or copied to the Intel laptop by this setup.
- Official Verda CLI v1.10.0: `/Users/abdullah/.local/bin/verda` (already on PATH). Official release download checksum verified.
- Default profile: `/Users/abdullah/.verda/credentials`, permissions `0600`, directory `0700`.
- Cloud API credential name: `local-cli`; expires **15 December 2026, 05:54 Europe/Helsinki**.
- API base URL: `https://api.verda.com/v1`.
- The CLI reads its client ID and client secret from the credentials file automatically. Do not print, copy into Git, or include that file in logs/artifacts. No secret values are stored in this handoff.
- This is a Cloud API credential for infrastructure management, not an inference API key.

Agents running as the same Mac user can use the commands below directly. Other hosts/users do not inherit this access. The older training handoff references a separate API client/secrets directory; do not assume its account/resources match this profile or overwrite it.

## Verified read-only commands

```sh
verda doctor
verda vm list
verda status
verda availability
verda availability --spot
verda availability --location FIN-03
verda availability --type 1A100.22V
verda instance-types --gpu
verda --help
```

If PATH is missing in an agent shell, use `/Users/abdullah/.local/bin/verda`.

`verda doctor` confirmed credentials found, API reachable and authentication valid. Its version check received GitHub HTTP 403; installation succeeded using the official latest-release redirect (v1.10.0). It also noted the non-default binary directory; that directory is intentional and on PATH. `verda vm list` returned **No instances found** for this profile at setup time. This does not establish the state of other accounts or retained storage.

## Live GPU availability observed 15 November 2026

| Billing | Location | Exact instance type | GPUs |
|---|---|---|---|
| On-demand | FIN-03 | `1A100.22V` | 1× A100 |
| On-demand | FIN-03 | `1RTXPRO6000.30V.CC` | 1× RTX PRO 6000 |
| Spot | FIN-01 | `1A100.22V` | 1× A100 |
| Spot | FIN-01 | `1H100.80S.30V` | 1× H100 |
| Spot | FIN-03 | `1RTXPRO6000.30V.CC` | 1× RTX PRO 6000 |
| Spot | FIN-03 | `8B200.240V.CC.INTEL` | 8× B200 |

FIN-02 reported no available types. CPU-only types are omitted from this table. Spot instances may be interrupted. Refresh availability and current prices before choosing compute; this setup created no instances and authorized no new training allocation.

## References

- [Official CLI installation](https://docs.verda.com/cli/)
- [CLI authentication and profiles](https://docs.verda.com/cli/getting-started/)
- [API credential types](https://docs.verda.com/welcome-to-verda/api-credentials/)
- [Existing training history and retained artifacts](VERDA-TRAINING.md)
