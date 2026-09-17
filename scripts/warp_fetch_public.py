#!/usr/bin/env python3
"""Fetch the pinned public video/metadata needed by the local WARP experiments."""
import argparse
import json
from pathlib import Path
import shutil
import sys

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from scripts.warp_network import configure_download_proxy
from warp_progress.features import atomic_json
from warp_progress.model import file_sha256
from warp_progress.selection import public_recipe_split

REPO = 'uynitsuj/sim-bottles-mjwarp-v1'
REVISION = '897b86679272a47897cc5ecca78c2205586a29d5'
FILES = [
    ('meta/episodes/chunk-000/file-00000.parquet', 'public_episodes.parquet',
     'f26eff70e106c9a76f66b0596d3a92dc758d1b8be56112a57f22fbe77be79e2e'),
    ('meta/object_counts.json', 'public_object_counts.json',
     'b1bcd5acc1dab237953abcf5eee2ae66832f5186c16b4f80194fec802fc8586f'),
    ('videos/top_camera-images-rgb/chunk-000/file-00007.mp4', 'public_top_camera_file_00007.mp4',
     '14ceccf471beb2b97ab4cd04dc6b5a0d986659afa17bc374323c7ad8d963fe4d'),
]


def fetch(data_root, metadata_only=False):
    configure_download_proxy()
    from huggingface_hub import hf_hub_download
    import pandas as pd
    root = Path(data_root).resolve()
    (root/'sources').mkdir(parents=True, exist_ok=True)
    rows = []
    for remote, local, digest in FILES[:2] if metadata_only else FILES:
        target = root/'sources'/local
        if not target.is_file():
            source = hf_hub_download(REPO, remote, repo_type='dataset', revision=REVISION,
                                     token=False, cache_dir=root/'download-cache')
            if file_sha256(source) != digest:
                raise ValueError('Pinned public data checksum mismatch: '+remote)
            shutil.copyfile(source, target)
        if file_sha256(target) != digest:
            raise ValueError('Existing public file changed: '+str(target))
        rows.append(dict(upstream_path=remote, local_path='sources/'+local, sha256=digest, bytes=target.stat().st_size))
        print('Verified '+local, flush=True)
    episodes = pd.read_parquet(root/'sources/public_episodes.parquet')[['episode_index', 'length']].to_dict('records')
    counts = json.loads((root/'sources/public_object_counts.json').read_text())['counts']
    train, validation, audit = public_recipe_split(episodes, counts)
    selection = dict(train=[e['episode_index'] for e in train], validation=[e['episode_index'] for e in validation],
                     audit=audit, video_sha256=FILES[2][2])
    path = root/'public_selection.json'
    if path.is_file():
        previous = json.loads(path.read_text())
        if previous['train'] != selection['train'] or previous['validation'] != selection['validation']:
            raise ValueError('Existing public split differs from the pinned recipe')
    else:
        atomic_json(path, selection)
    atomic_json(root/'public_download_audit.json', dict(repo=REPO, revision=REVISION, files=rows,
                training_episodes=len(train), validation_episodes=len(validation),
                scope='One video shard and full episode metadata; not the full multi-camera training corpus'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(APP.parent/'research/warp_rm_20260916'))
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    fetch(args.data_root, args.metadata_only)
