# Third-party attribution

- `vendor/physical-ai-studio/` is the Intel/open-edge-platform Physical AI Studio submodule, pinned to the commit in `.gitmodules`/the Git tree. Its source and license remain upstream.
- `dashboard/` is derived from Physical AI Studio's `junction2026` UI. Original Apache-2.0 license and source headers are retained in that directory.
- Simulation asset downloads are handled by `simulation/setup_assets.py`; downloaded assets are excluded from Git. Respect the licenses accompanying the referenced MuJoCo Menagerie assets.
- Training uses Hugging Face LeRobot and pretrained SmolVLA. Model weights are not redistributed here; use the source/model license and the campaign's recorded revision.
- Organizer documents in `docs/` retain their original authorship. They are reference material, not project-authored implementation.

This notice does not relicense third-party code or grant a new license for project-authored code.
