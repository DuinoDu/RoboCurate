## Unreleased — device management and sync

- Add collection-node (stdlib agent for recording devices) and collection-hub (separate service on the storage server): device online and recording state, disk levels for devices and the server, anomaly alerts with an optional webhook.
- Recording state comes from the recording tree (`mcap/__RECORDING__` and MCAP growth); the PICO server's `recorder.active` is only compared, because it stays true after a recorder crash and reads false for a recorder orphaned by a server restart.
- Pull finished episodes with rsync over SSH (read-only rrsync on the device): never while a device records, after an idle period or when marked off-shift, with a shared bandwidth cap while any device records, a free-space reserve, per-file verification, and a trash folder for episodes deleted on the device.
- Optional, disabled-by-default node capabilities: power off after everything is synced, and delete already verified data on the device after a preview and confirmation code.
- RoboCurate: 设备管理 page, global disk banner, `/api/hub/*` proxy, incremental re-scan of synced sessions, a `device` field from the mirror layout, and task/user names taken from the episode directory when the sidecar disagrees.
- Add a user-level systemd unit for RoboCurate.

## v0.1.0-preview.3 — complete training source and model distribution

- Publish the remaining research/training, calibration, ablation, validation and historical delivery scripts with frozen protocols and a source index.
- Distribute the exact four installed DINOv2-based models plus the pinned backbone as a separate 93.68 MB model ZIP.
- Add checksum-verified, conflict-preserving online/offline installation and a real model forward-pass verifier.
- Exercise the previously external training helpers in source CI. Private recordings, labels, features, credentials and review state remain excluded.
- No new model training or accuracy improvement is claimed in this packaging release.

# Changes

## v0.1.0-preview.2 — 2026-09-18

- Update the interface gallery with measured pose ghosts, recorded control-target poses and target trajectories, captured directly from the application.
- Explain the visual overlays and provide a matching source download.

## v0.1.0-preview.1 — 2026-09-18

- Rename the application to RoboCurate and migrate saved browser preferences.
- Reduce review navigation to Task, Quality checks and Review; move advanced tools into disclosures.
- Keep one synchronized progress curve and switch quality/task timelines with the active view.
- Show current model evidence and actionable review intervals without duplicate summary cards.
- Change ratings, dimensions, API values and new exports to ten points; preserve grade gates and percentage metrics.
- Add a portable synthetic MCAP demo and an end-to-end review/export test with no private recording.
- Remove machine-specific default recording paths and inventory third-party origins.
- Provide source-only packaging, download instructions and GitHub Actions tests under Lee-hz/RoboCurate; keep the unresolved base-viewer license review explicit.
- Show actual G1 camera, robot and stage-model screenshots in the README, with the synthetic example kept in a separate disclosure.
- Include SciPy in development dependencies so WARP sampler parity checks run on a fresh CI installation.
