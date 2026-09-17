# Changes

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
