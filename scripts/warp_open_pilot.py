#!/usr/bin/env python3
"""Train and check a WARP-method pilot on real public video with ungated DINOv2.

Uses disjoint whole episodes from the published shortest-demo selection. This
small CPU experiment is an adaptation, not the paper's full training recipe.
Validation selects a checkpoint; an additional untouched test split measures
time-warp prediction and visual-input controls. No robot success is inferred.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import numpy as np
import pandas as pd
import torch
from warp_progress.core import WarpConfig, contiguous_runs
from warp_progress.features import (OPEN_BACKBONE_ID, DinoEncoder, atomic_json, atomic_npz,
                                    extract_video, feature_contract, installed_backbone_profile)
from warp_progress.model import file_sha256, load_model, infer_features
from warp_progress.train import train, sample_batch

VIDEO_SHA256 = '14ceccf471beb2b97ab4cd04dc6b5a0d986659afa17bc374323c7ad8d963fe4d'
TASK = '将场景中的瓶子放入容器（公开仿真任务）'


def make_plan(data_root, output):
    selection_path = data_root/'public_selection.json'
    meta_path = data_root/'sources/public_episodes.parquet'
    selection = json.loads(selection_path.read_text())
    meta = pd.read_parquet(meta_path).set_index('episode_index')
    available = meta[(meta['videos/top_camera-images-rgb/file_index'] == 7) &
                     meta.index.isin(selection['train'])]
    ids = np.random.default_rng(20260916).permutation(sorted(available.index))[:30]
    if len(ids) != 30:
        raise ValueError('The pinned public video needs at least 30 selected demonstrations')
    rows = []
    for i, ep in enumerate(ids):
        row = meta.loc[int(ep)]
        split = 'train' if i < 20 else 'validation' if i < 25 else 'test'
        rows.append(dict(id=int(ep), split=split, source_episode_id=row['source_episode_id'],
                         start_frame=round(float(row['videos/top_camera-images-rgb/from_timestamp']) * 30),
                         n_frames=int(row['length'])))
    plan = dict(episodes=rows, video_sha256=VIDEO_SHA256, metadata_sha256=file_sha256(meta_path),
                selection_sha256=file_sha256(selection_path), seed=20260916,
                split_policy='20 train / 5 validation / 5 untouched test whole episodes, sampled from the original train pool in file 7; original paper validation untouched',
                scope='CPU feasibility pilot on one public video shard; not a full paper reproduction or independent-domain validation')
    path = output/'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Existing pilot output has a different data plan')
    atomic_json(path, plan)
    return plan


def check_held_out(model, episodes, config, batches=20, batch_size=16):
    rng = np.random.default_rng(42091)
    sums = {name: dict(squared=0., tokens=0, positive_right=0, negative_right=0,
                       positive_count=0, negative_count=0) for name in ('real_features','shuffled_frames','constant_frame')}
    zero_squared = 0.
    model.eval()
    with torch.inference_mode():
        for _ in range(batches):
            x, y = sample_batch(episodes, config, rng, batch_size)
            dy = np.diff(y)
            positive, negative = dy > 1e-6, dy < -1e-6
            shuffled = np.stack([row[rng.permutation(len(row))] for row in x])
            constant = np.broadcast_to(x[:, :1], x.shape).copy()
            for name, observations in [('real_features', x), ('shuffled_frames', shuffled), ('constant_frame', constant)]:
                prediction = model(torch.from_numpy(observations))[0].numpy()
                dp = np.diff(prediction)
                s = sums[name]
                s['squared'] += float(np.square(prediction-y).sum())
                s['tokens'] += y.size
                s['positive_right'] += int(((dp > 0) & positive).sum())
                s['negative_right'] += int(((dp < 0) & negative).sum())
                s['positive_count'] += int(positive.sum())
                s['negative_count'] += int(negative.sum())
            zero_squared += float(np.square(y).sum())
    metrics = {}
    for name, s in sums.items():
        positive_accuracy = s['positive_right']/max(1, s['positive_count'])
        negative_accuracy = s['negative_right']/max(1, s['negative_count'])
        metrics[name] = dict(rmse=float(np.sqrt(s['squared']/s['tokens'])),
                             forward_accuracy=positive_accuracy, reverse_accuracy=negative_accuracy,
                             balanced_direction_accuracy=(positive_accuracy+negative_accuracy)/2,
                             tokens=s['tokens'])
    metrics['zero_progress_baseline'] = dict(rmse=float(np.sqrt(zero_squared/sums['real_features']['tokens'])))
    metrics['protocol'] = dict(warps=batches*batch_size, seed=42091,
        description='Same test warps and labels for all controls. Shuffling or freezing input frames keeps temporal positions but removes useful visual motion. Zero baseline always predicts zero relative progress.',
        limitation='Artificial time warps of held-out real video; not ground-truth task progress or policy success')
    return metrics


def run(args):
    torch.set_num_threads(4)
    data_root, output = Path(args.data_root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output/'status.json'
    started = time.time()
    def status(stage, **extra):
        atomic_json(status_path, dict(state=stage, started_at=started, updated_at=time.time(), **extra))
    try:
        plan = make_plan(data_root, output)
        video = data_root/'sources/public_top_camera_file_00007.mp4'
        if file_sha256(video) != VIDEO_SHA256:
            raise ValueError('Public video differs from the pinned source hash')
        profile = installed_backbone_profile(Path(args.workspace)/'warp/models', OPEN_BACKBONE_ID,
                                              camera='top_camera-images-rgb', view='full', crop='squash')
        config = WarpConfig(feature_stride=3, source_standard_stride=15)
        contract = feature_contract(profile, config); contract['decoder'] = 'OpenCV-RGB'
        encoder = None
        rows, test_rows, source_splits = [], [], {}
        for i, ep in enumerate(plan['episodes']):
            status('extracting', completed=i, total=len(plan['episodes']), episode=ep['id'])
            path = output/'features'/f"episode_{ep['id']}.npz"
            descriptor_path = path.with_suffix('.json')
            if path.exists() and descriptor_path.exists():
                d = json.loads(descriptor_path.read_text())
                if d['contract'] != contract or d['start_frame'] != ep['start_frame'] or d['n_frames'] != ep['n_frames']:
                    raise ValueError('Existing features have different preprocessing or episode boundaries')
            else:
                if encoder is None:
                    encoder = DinoEncoder(profile['backbone_dir'], checksums=profile['backbone_sha256'], backbone_id=OPEN_BACKBONE_ID)
                extract_video(video=video, output=path, config=config, profile=profile, encoder=encoder, batch_size=16,
                              start_frame=ep['start_frame'], n_frames=ep['n_frames'])
                d = json.loads(descriptor_path.read_text())
            digest = d['source_sha256']
            if digest in source_splits and source_splits[digest] != ep['split']:
                raise ValueError('Identical decoded source content leaks across pilot splits')
            source_splits[digest] = ep['split']
            row = dict(id=ep['id'], split=ep['split'], source_sha256=digest,
                       features=str(path.relative_to(output)), task=TASK,
                       selection='Original public stratified shortest-demo train pool; fixed-seed CPU subset')
            (test_rows if ep['split'] == 'test' else rows).append(row)
            print(f"Features {i+1}/{len(plan['episodes'])}: episode {ep['id']} {ep['split']}", flush=True)
        del encoder
        manifest = output/'manifest.json'
        atomic_json(manifest, dict(episodes=rows, split_policy=plan['split_policy'], warp_config=asdict(config)))
        atomic_json(output/'test_episodes.json', dict(episodes=test_rows, untouched_during_training=True))
        train_output = output/'train'
        status('training', manifest=str(manifest))
        run_path = train_output/'run.json'
        if run_path.exists():
            training = json.loads(run_path.read_text())
            if training['state'] != 'complete' or training['audit']['manifest_sha256'] != file_sha256(manifest):
                raise ValueError('Prior training is incomplete or uses different data; use a new output directory')
        else:
            training = train(argparse.Namespace(
                manifest=str(manifest), output=str(train_output), steps=args.steps, batch_size=32, lr=3e-4,
                warmup_steps=100, eval_every=200, eval_batches=8, seed=42, feature_stride=3,
                source_standard_stride=15, d_model=128, n_heads=4, n_layers=4,
                init=None, device='cpu', threads=4))
        status('testing')
        model, config, checkpoint = load_model(train_output/'best.pt', expected_sha256=training['best_sha256'])
        episodes, test_audit = [], []
        for row in test_rows:
            path = output/row['features']
            with np.load(path, allow_pickle=False) as a:
                x, valid, times = a['features'], a['valid'], a['timestamp_ns']
            runs = [(int(a), int(b)) for a,b in contiguous_runs(valid) if b-a >= config.window_size+1]
            episodes.append((x, runs))
            curve = infer_features(model, x, valid, config)
            atomic_npz(output/'test_curves'/f"episode_{row['id']}.npz", timestamp_ns=times,
                       velocity=curve['velocity'], valid=curve['valid'], coverage=curve['coverage'])
            test_audit.append(dict(**row, feature_sha256=file_sha256(path), scored_frames=int(curve['valid'].sum())))
        metrics = check_held_out(model, episodes, config)
        result = dict(state='complete', backbone=profile, warp_config=asdict(config), plan=plan,
                      checkpoint_sha256=checkpoint['sha256'], training_step=checkpoint['step'],
                      architecture=training['architecture'], paper_architecture=False, test=test_audit,
                      held_out_controls=metrics, started_at=started, completed_at=time.time(),
                      claim='WARP-method adaptation using official pretrained DINOv2 and a new temporal head; no original DINOv3 inference, policy training or G1 success evaluation')
        atomic_json(output/'report.json', result)
        status('complete', report=str(output/'report.json'))
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        status('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', default=str(APP/'workspace'))
    parser.add_argument('--data-root', default=str(APP.parent/'research/warp_rm_20260916'))
    parser.add_argument('--output', default=str(APP.parent/'research/warp_rm_20260916/open_pilot'))
    parser.add_argument('--steps', type=int, default=1600)
    run(parser.parse_args())
