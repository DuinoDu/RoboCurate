# Training and research source

This directory publishes the original project training, selection, calibration,
ablation, audit, plotting and delivery scripts. `SOURCE_INDEX.json` records the
original and published source hashes. Training imports were adapted to the flat
GitHub layout. No network is called merely by importing the training engines.

| Directory | Work represented |
| --- | --- |
| `warp_rm_20260916` | Public WARP rollout replay and the initial input audit; WARP temporal training itself is in `warp_progress/train.py` and `scripts/warp_open_pilot.py` |
| `model_upgrade_20260917` | Feature extraction, sparse-stage supervision, v1 MLP/Transformer comparisons, selection and registration |
| `stage_v2_20260917` | Measured-state fusion, temporal/stage attention, five-fold comparisons, temperature calibration |
| `stage_v3_20260917` | Shared training engine, context ablations, random-walk transition supervision |
| `stage_v3_motion_20260917` | Motion-difference training, review safeguards and deployed v3 finalization |
| `stage_v4_20260917`, `stage_v4_decoupled_20260917` | Contrastive, boundary/segment and gradient-isolation experiments; none promoted |
| `interaction_20260917`, `simplify_release_20260917` | Historical UI validation and screenshot capture |

Frozen `protocol*.json` and `freeze.json` retain the actual experimental splits,
seeds, hyperparameters, selection gates and original input hashes. They are
research records, not a claim that the private data are included. To rerun the
original measurements one also needs the exact annotations, visual/state features,
source manifests and reference fold checkpoints. See [training inputs](../docs/TRAINING.md).

Historical `build_delivery.py`, `build_handoff.py`, registration and finalization
scripts describe one-time local handoffs. They require the corresponding old
archives/reports/workspace and are **not the current public packaging/install
commands**. Use `scripts/build_source_candidate.py`, `scripts/build_model_bundle.py`
and `scripts/install_models.py` for the public release. Old UI checks likewise
need their named local recordings, browser and running application.

The screenshot capture script writes only selected images; browser profiles,
logs, raw recordings and review databases are not shipped. The original label
mapping used by `prepare_study.py` must be supplied locally as `label_map.json`;
private annotations have not been embedded in the published source.
