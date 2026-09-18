# Training source and reproduction inputs

All project training algorithms are now present in the repository: WARP relative
temporal learning; v1 stage heads; v2 multimodal temporal fusion; v3 motion
differences and random-walk targets; v4 contrastive/boundary/segment objectives;
and the decoupled-progress experiment. The published model pack supplies the four
models available in the current application. Installing them does not require
training data or Hugging Face authentication.

## Use the trained models

Follow [model setup](MODELS.md). The released v3 is the exact checkpoint with the
review safeguard enabled (`bfd92eab9d50…`), not its earlier unguarded classifier.
The DINOv2 backbone and all four head files are checked against the same hashes
used in the local development installation. No v4 candidate is installed.

## Reproduce WARP training on public videos

The following entry points expose their options with `--help`:

```bash
workspace/warp-env/bin/python scripts/warp_fetch_public.py --help
workspace/warp-env/bin/python scripts/warp_open_pilot.py --help
workspace/warp-env/bin/python -m warp_progress.train --help
```

The pilot records its public data selection, episode splits, feature provenance,
training settings and temporal controls. The small DINOv2 adaptation is not the
original DINOv3 experiment and does not establish G1 task success.

## Reproduce the historical G1 experiments

The executable algorithms are in [research](../research/README.md); the frozen
protocols record the original settings. The following private **inputs** remain
necessary for numerical reproduction and are not included in either ZIP:

| Location relative to repository | Required inputs |
| --- | --- |
| `research/model_upgrade_20260917/` | `annotations.v1.json`, `features_manifest.json`, optional original frame index for re-extraction |
| `workspace/warp/features/` | Two-camera DINOv2 NPZ caches named by the visual manifest |
| `research/stage_v2_20260917/` | `state_manifest.json`, `states/*.npz` |
| `research/stage_v2_20260917/training/` | Five `fusion_tokens-fold*.pt` reference ensembles for the v3 comparisons |
| `research/stage_v3_motion_20260917/training/` | Five `motion_rw-fold*.pt` reference ensembles for the v4 comparisons |

Original manifests include hashes; a changed dataset must use its own new
protocol, annotations and output directory. Do not replace those hashes merely
to label new results as a reproduction. Held-out historical recordings have
already been inspected during development; reusing them does not make a new
independent test set.

After restoring the original inputs, run from this repository root:

```bash
# v1 supports a separate output directory directly.
workspace/warp-env/bin/python research/model_upgrade_20260917/train_stages.py --output workspace/reproduction-v1

# For a fresh v2 run, the study training/ directory must be empty.
workspace/warp-env/bin/python research/stage_v2_20260917/train_cv.py

# v3 uses the completed v2 reference folds.
workspace/warp-env/bin/python research/stage_v3_motion_20260917/train.py

# v4 studies reuse the v3 reference folds; they were not promoted.
workspace/warp-env/bin/python research/stage_v4_20260917/train.py
workspace/warp-env/bin/python research/stage_v4_decoupled_20260917/train.py
```

The v2/v3/v4 engines refuse to overwrite a fixed selection. Keep historical
outputs intact and use a separate checkout/workspace for another complete run.
`prepare.py` scripts describe how the original protocols and state arrays were
created; the published protocols are already frozen, so these scripts should
not be run over them. Plotting scripts additionally require Matplotlib.

For a different G1 dataset, the extraction/annotation code must be configured
for its recordings and labels. The research scripts are not a generic upload-
and-train service. Required array fields and the strong-label interpolation are
implemented in `model_upgrade_20260917/train_stages.py:read_study`; state alignment
is in `warp_progress/state_features.py`. Missing observations retain masks, and
whole-recording SHA-256 checks prevent duplicate recordings crossing splits.

## What the public tests establish

```bash
PYTHONPATH=. workspace/warp-env/bin/python -m pytest tests -q
workspace/warp-env/bin/python scripts/verify_models.py
```

Tests exercise the actual training helper functions, causal inference, missing
inputs and calibration/review semantics. The model verifier runs the real
released encoder and heads on generated inputs. These are functional checks,
not a measurement of task accuracy. The 95.14% v3 macro F1 is a same-session
development cross-validation result from repeatedly used provisional labels;
final placement in the basket remains a human review decision.
