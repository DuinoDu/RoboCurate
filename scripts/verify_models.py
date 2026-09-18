#!/usr/bin/env python3
"""Check actual released weights, feature contracts and a numerical inference smoke test."""
import argparse
import json
from pathlib import Path
import sys

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))


def verify(workspace):
    import numpy as np
    import torch
    from warp_progress.features import DinoEncoder, preprocess_rgb, validate_training_contract, feature_contract
    from warp_progress.core import WarpConfig
    from warp_progress.model import file_sha256, load_model
    from warp_progress.stages import load_stage_checkpoint, infer_stage_models
    from warp_progress.stages_v2 import load_fusion_checkpoint, infer_fusion
    from warp_progress.stages_v3 import load_stage_v3, inference_v3

    torch.set_num_threads(2)
    root = Path(workspace).resolve() / 'warp/models'
    profiles = json.loads((root / 'profiles.json').read_text())['models']
    wanted = json.loads((APP / 'models/profiles.json').read_text())['models']
    encoder = DinoEncoder(root / wanted[0]['backbone_dir'], backbone_id=wanted[0]['backbone_id'],
                          checksums=wanted[0]['backbone_sha256'])
    rng = np.random.default_rng(817)
    # Generated images verify a real encoder forward pass, not task accuracy.
    images = [preprocess_rgb(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(2)]
    vector = encoder(images)
    x = np.tile(vector[None], (16, 1, 1))
    cv = np.ones((16, 2), bool)
    q = np.zeros((16, 41), np.float32)
    qv = np.ones(16, bool)
    results = []
    for profile in wanted:
        if profile not in profiles:
            raise ValueError('Missing or differing installed profile: ' + profile['id'])
        path = root / profile['checkpoint']
        if file_sha256(path) != profile['checkpoint_sha256']:
            raise ValueError('Checkpoint checksum mismatch')
        if profile.get('kind') != 'stage_progress':
            model, config, saved = load_model(path, expected_sha256=profile['checkpoint_sha256'])
            validate_training_contract(profile, config, saved)
            with torch.inference_mode():
                result = model(torch.from_numpy(np.tile(vector[0], (1, config.window_size, 1))))[0]
            if not torch.isfinite(result).all():
                raise ValueError('Non-finite WARP inference')
        else:
            version = profile.get('stage_version', 1)
            loader = {1: load_stage_checkpoint, 2: load_fusion_checkpoint, 3: load_stage_v3}[version]
            models, saved = loader(path, profile['checkpoint_sha256'])
            for camera, contract in zip(saved['cameras'], saved['feature_contracts']):
                expected = feature_contract(dict(profile, camera=camera,
                    view='left' if camera == 'head' else 'full'), WarpConfig(feature_stride=15))
                if contract != expected:
                    raise ValueError('Stage feature contract mismatch')
            if version == 1:
                result = infer_stage_models(models, x, cv.all(1), saved['priors'], saved['temperature'])
            elif version == 2:
                result = infer_fusion(models, x, cv, q, qv, saved['priors'], saved['temperature'])
            else:
                result = inference_v3(models, x, cv, q, qv, saved['priors'], saved['temperature'],
                                      saved.get('boundary_review', False), saved.get('review_guard', False))
            if not result['valid'].all() or not np.isfinite(result['probability']).all():
                raise ValueError('Invalid stage inference')
        results.append(dict(id=profile['id'], sha256=profile['checkpoint_sha256'], inference='passed'))
    return dict(models=results, backbone_forward='passed', task_accuracy_evaluated=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=APP / 'workspace')
    args = parser.parse_args()
    print(json.dumps(verify(args.workspace), ensure_ascii=False, indent=2))
