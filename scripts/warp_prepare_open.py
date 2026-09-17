#!/usr/bin/env python3
"""Install the ungated official DINOv2-S/14 backbone for WARP method adaptation.

This does not install or reuse the DINOv3-trained paper head. A new temporal
model must be trained with this backbone's features.
"""
import argparse
from pathlib import Path
import sys

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from warp_progress.features import OPEN_BACKBONE_ID, BACKBONES, atomic_json
from warp_progress.model import file_sha256

REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'


def prepare(workspace):
    from scripts.warp_network import configure_download_proxy
    configure_download_proxy()
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    spec = BACKBONES[OPEN_BACKBONE_ID]
    root = Path(workspace).resolve() / 'warp/models'
    target = root / spec['directory']
    info = HfApi(token=False).model_info(OPEN_BACKBONE_ID, revision=REVISION, files_metadata=True)
    if info.gated or info.card_data.get('license') != 'apache-2.0':
        raise ValueError('Expected the ungated Apache-2.0 official DINOv2 model')
    expected = next(f.lfs.sha256 for f in info.siblings if f.rfilename == 'model.safetensors')
    snapshot_download(OPEN_BACKBONE_ID, revision=REVISION, local_dir=target, token=False,
                      allow_patterns=['config.json', 'model.safetensors', 'preprocessor_config.json', 'README.md'])
    checksums = {name: file_sha256(target/name) for name in ('config.json', 'model.safetensors')}
    if checksums['model.safetensors'] != expected:
        raise ValueError('DINOv2 weights differ from the pinned upstream LFS checksum')
    provenance = dict(repo=OPEN_BACKBONE_ID, revision=REVISION, sha256=checksums,
                      license='Apache-2.0', obtained_via='Official public Hugging Face repository; token=False',
                      role='Alternative visual backbone; requires separately trained WARP temporal head',
                      feature_dimension=spec['dimension'])
    atomic_json(root/spec['provenance'], provenance)
    print(f"Installed {OPEN_BACKBONE_ID} at revision {REVISION}; weights SHA256 {expected}", flush=True)
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', default=str(APP/'workspace'))
    prepare(parser.parse_args().workspace)
