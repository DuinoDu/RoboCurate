# Candidate validation — 2026-09-17

This release changes interaction, rating units, branding and source packaging. It does not retrain the neural networks or claim improved model accuracy.

| Scope | Local result |
| --- | --- |
| Existing development workspace, full Python environment | 126 passed |
| Frontend helper tests | All 8 Node test files passed |
| Isolated source candidate, application environment | 75 passed, 35 skipped |
| Isolated source candidate, model environment | 102 passed, 24 skipped |
| Portable synthetic browser workflow | 17 checks passed, including a real 30-sample training-bundle export |
| Existing stage-model UI | 40 checks passed, including probability parity, missing-input evidence, review intervals, playback and responsive layouts |
| WARP browser checks | 16 fixture-based checks and 10 real-result checks passed |

Candidate tests run from a separate copy containing only the allowlisted source, using existing local Python environments. They are not a fresh package-index installation or a GitHub Actions run. Private-recording, unbundled-checkpoint, upstream-snapshot and external study checks explicitly skip when their inputs are absent. The application-only environment also skips unavailable optional model/video dependencies. Pytest can skip whole model modules before collecting their individual tests, so skip totals across environments are not directly comparable.

The synthetic MCAP generator was checked for known arm-tracking and camera-gap events, readable camera frames, explicit synthetic labelling, source immutability and review/export round trips. It is a software fixture, not evidence of visual-model accuracy.

On the existing 19-record development collection, every available technical score equals the previous value divided by ten. Grade results and human reviews match the pre-change snapshot. All 116 saved model-score JSON/NPZ files, model profiles and the checksums of five configured checkpoints were verified unchanged. Playback, robot geometry and trajectory calculation source files were also unchanged.

The source builder verifies ZIP CRC and per-file SHA-256. Third-party licenses and the unresolved base-viewer file inventory are included. License review remains pending as described in [LICENSE_STATUS.md](../LICENSE_STATUS.md).
