"""Build matching 2 Hz head/right-wrist DINOv2 caches without changing source data."""
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parent
APP = ROOT.parents[1]
sys.path.insert(0, str(APP))
import torch
from warp_progress.core import WarpConfig
from warp_progress.features import DinoEncoder, OPEN_BACKBONE_ID, installed_backbone_profile, extract_mcap, atomic_json
from warp_progress.model import file_sha256

def main():
    torch.set_num_threads(4)
    config = WarpConfig(feature_stride=15)
    modelroot = APP/'workspace/warp/models'
    head = installed_backbone_profile(modelroot, OPEN_BACKBONE_ID)
    encoder = DinoEncoder(head['backbone_dir'], checksums=head['backbone_sha256'], backbone_id=OPEN_BACKBONE_ID)
    metadata = {}
    for p in (APP/'workspace/cache').glob('*/meta.json'):
        meta = json.loads(p.read_text())
        metadata[meta['source']['path']] = (p, meta)
    manifest = []
    for i in range(124,142):
        episode = f'episode_{i:06d}'
        idx = json.loads((ROOT/'local_evidence'/episode/'index.json').read_text())
        path, meta = metadata[idx['source']]
        row = dict(episode=episode, source=idx['source'], meta=str(path), cameras={},
                   start_ns=meta['episode']['start_ns'],end_ns=meta['episode']['end_ns'])
        started = time.monotonic()
        row['source_sha256'] = file_sha256(idx['source'])
        for camera, view in [('head','left'),('right_wrist','full')]:
            profile = installed_backbone_profile(modelroot, OPEN_BACKBONE_ID, camera=camera, view=view)
            arrays, descriptor, cache = extract_mcap(idx['source'], meta, config, profile,
                APP/'workspace/warp/features', encoder, batch_size=8)
            row['cameras'][camera] = dict(cache=str(cache), contract=descriptor['contract'],
                valid=int(arrays['valid'].sum()), total=len(arrays['valid']))
            print(episode, camera, row['cameras'][camera]['valid'], 'features', flush=True)
        row['elapsed_s'] = time.monotonic()-started
        manifest.append(row)
        atomic_json(ROOT/'features_manifest.json', dict(complete=len(manifest)==18, episodes=manifest))
    print('Complete', flush=True)

if __name__ == '__main__':
    main()
