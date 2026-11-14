# Junction Intel Robotics

Team solution for the Junction 2026 Intel Physical AI challenge: learned robot manipulation, Anomalib inspection, and OpenVINO deployment.

## Documentation

- [Execution plan](docs/EXECUTION-PLAN.md): follow this
- [Documentation index](docs/README.md)
- [Winning plan](docs/Intel-winning-plan.md)
- [Speed and reliability research](docs/Intel-speed-and-reliability-research.md)
- [Robot station handoff](docs/ROBOT-STATION-HANDOFF.md)

## Intel dependency

[Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio) is included as a Git submodule at `vendor/physical-ai-studio`. It is pinned to **`17714488fa9c7d098a056da83f7c220cf983957d`**, the `junction2026` commit recorded in our station handoff. The submodule preserves upstream history, source and licensing; our integration belongs in this repository.

Install Git LFS before cloning if you want upstream demo media, then:

```sh
git clone --recurse-submodules https://github.com/Abdullah12has/junction-intel-robotics.git
cd junction-intel-robotics
```

For an existing clone:

```sh
git submodule update --init --recursive
```

These commands use the committed submodule pin once this local change is committed and pushed. Do not use `git submodule update --remote` for reproducible setup: it follows a newer branch tip instead of the recorded commit.

The initial local checkout retained Git LFS pointers for upstream video/GIF assets because Git LFS was unavailable. Source code is checked out. To retrieve media after installing Git LFS:

```sh
git -C vendor/physical-ai-studio lfs pull
```

Follow the pinned [event setup guide](vendor/physical-ai-studio/JUNCTION2026.md) for training and inference. Preserve existing credentials/configuration instead of overwriting `.env` from an example. Do not commit tokens, private keys or datasets containing private information.

## Existing station and dashboard

Our customized UI is included in [dashboard/](dashboard/README.md), with source, tests, lockfile and upstream licence. The original working copy at `/Users/abdullah/Desktop/junction/studio-ui` and the running laptop installation were left untouched. Use `dashboard/` for future solution development. See [dashboard notes](docs/ROBOT-DASHBOARD.md).

Before running robot code, follow the station handoff and confirm device ownership, calibration and live session state. Only one runtime may own an arm.
