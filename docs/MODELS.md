# Models and reproducibility scope

The application works without neural model weights: import MCAP, play synchronized frames/motion, inspect technical quality, save a review, and export selected clips. A fresh source installation reports that no task model is installed. It does not synthesize model predictions for the demo.

## Implemented model families

| Family | Implementation | Meaning of the output |
| --- | --- | --- |
| WARP relative progress | `warp_progress/core.py`, `model.py`, `train.py`, `worker.py`; upstream reference in `vendor/warp_rm` | Relative visual progress speed, regressions and action-chunk weights; not completion probability |
| G1 stage v1/v2/v3 | `stages.py`, `stages_v2.py`, `stages_v3.py`, stage workers | Five task phases, auxiliary stage progress, class probabilities and review reasons |
| Stage v4 experiments | `stages_v4.py` | Research implementation of extra training objectives; no promoted v4 checkpoint |

The existing development installation uses `g1-stage-v3` for the task “walk to the left table, pick up the banana prop, carry it to the right table, and place it in the basket.” Its frozen DINOv2-S/14 encoder extracts 384-dimensional features from the head left view and right-wrist camera at 2 Hz. The stage network fuses those features, 41 joint positions, adjacent valid-sample differences and validity masks. It uses eight causal samples, a two-layer/four-head Transformer with width 96, stage attention, and a three-member ensemble.

Missing inputs remain explicit. Both views missing produces no stage output; partial input loss triggers review. Calibrated class confidence is not task-success confidence. “Post-release observation” does not prove the banana is in the basket.

The trained **G1 v1/v2/v3 and open-dinov2-pilot heads plus the pinned DINOv2 backbone are published** in the [preview.3 model pack](https://github.com/Lee-hz/RoboCurate/releases/download/v0.1.0-preview.3/RoboCurate-v0.1.0-preview.3-models.zip) (93.68 MB). The source ZIP contains the matching profiles, SHA-256 manifest and installer; the binary weights are a separate Release asset. The original model files are unchanged.

All project training/selection/ablation code is included under `research/`, `warp_progress/` and `scripts/`. Private recordings, annotations, features and reference-fold experiment outputs are excluded. [Training instructions](TRAINING.md) explain the required inputs; source and weights alone do not reproduce the historical accuracy measurement.

## Optional model environment

From the source directory, using Python 3.10+:

```bash
python3 -m venv workspace/warp-env
workspace/warp-env/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
workspace/warp-env/bin/python -m pip install -r requirements-warp.txt
python3 scripts/install_models.py
workspace/warp-env/bin/python scripts/verify_models.py
```

The installer downloads the public GitHub model pack without login, verifies its archive and per-file hashes, and registers four models. The verifier checks the real encoder and all heads on generated inputs; it is a functional check, not an accuracy benchmark. Start/reload RoboCurate and choose `g1-stage-v3` for the banana-transfer task. CPU is the tested configuration.

If the browser is easier for downloading, save the model ZIP and install it offline:

```bash
python3 scripts/install_models.py --archive /path/to/RoboCurate-v0.1.0-preview.3-models.zip
```

Use `--workspace /path/to/workspace` consistently with the application if you choose a nondefault workspace; its model interpreter must be at `warp-env/bin/python`. Reinstallation is idempotent for identical assets and refuses conflicting local profiles or model files. The DINOv2 license is included in the model ZIP.

`scripts/warp_prepare_open.py` remains available for downloading only the backbone from Hugging Face; it does not install the trained heads.

For the optional Hugging Face backbone downloader, set `ROBOCURATE_HF_PROXY` if an HTTP proxy is required. The GitHub model installer uses standard `HTTPS_PROXY` / `HTTP_PROXY` environment variables. The legacy environment variable is accepted for existing installations. No proxy port is assumed. A Hugging Face login is not needed for this DINOv2 download. The optional `scripts/hf_login.sh` invokes the installed official CLI; it does not store credentials in the source tree.

## WARP training and registration

The source contains the temporal training implementation and scripts to prepare features and register a provenance-bearing checkpoint:

```bash
workspace/warp-env/bin/python scripts/warp_extract_local.py --help
workspace/warp-env/bin/python -m warp_progress.train --help
workspace/warp-env/bin/python scripts/warp_register_model.py --help
```

Registration verifies the backbone identity, revision, hashes, task, feature geometry and timing contract. `warp_register_model.py` registers WARP temporal checkpoints, not stage checkpoints. A DINOv3-trained checkpoint is incompatible with DINOv2 features.

Installed model assets live under `workspace/warp/models/`, with `profiles.json` describing the checkpoint and backbone files. Do not invent a profile for an incompatible checkpoint. A trusted stage-model release must supply its matching profile and preprocessing/provenance information as well as the weights. The server displays only configured models and checks file availability; inference rechecks model integrity.

## Tests

Runtime model tests check causality, missing-input handling, checkpoint round trips, review reasons and export weight alignment using generated fixtures. They require PyTorch; some WARP tests also require SciPy. They neither download weights nor claim task accuracy.

Training-helper tests now use the bundled research source. Tests requiring private recordings explicitly skip when those recordings are absent. `tests/test_demo.py` remains runnable without private inputs and exercises MCAP parsing, known fault detection, review and an actual training-bundle export.
