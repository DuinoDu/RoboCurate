"""Offline scoring subprocess; never imports ML libraries into the web server."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
import numpy as np
from .core import curve_summary, VERSION, UPSTREAM
from .features import BACKBONE_ID, DinoEncoder, extract_mcap, atomic_json, atomic_npz, validate_training_contract
from .model import load_model, infer_features


def run(job_path):
    job_path = Path(job_path)
    job = json.loads(job_path.read_text())
    request = job['request']
    profile = request['profile']
    started = time.monotonic()
    torch.set_num_threads(request.get('threads', 4))
    def update(message, progress):
        job.update(state='running', message=message, progress=round(progress, 2), updated_at=time.time())
        atomic_json(job_path, job)
    try:
        update('正在载入已核验的模型', 0)
        if profile.get('kind') == 'stage_progress':
            if profile.get('stage_version',1) in (2,3,4):
                from .stage_worker_v2 import score_stage
            else:
                from .stage_worker import score_stage
            score_stage(request, update)
            job.update(state='complete', progress=100, message='阶段分析完成', finished_at=time.time())
            atomic_json(job_path, job)
            return
        model, config, checkpoint = load_model(profile['checkpoint'], request['device'], profile['checkpoint_sha256'])
        validate_training_contract(profile, config, checkpoint)
        encoder = DinoEncoder(profile['backbone_dir'], request['device'], profile.get('backbone_sha256'),
                              profile.get('backbone_id', BACKBONE_ID))
        data, descriptor, feature_path = extract_mcap(
            request['source'], request['meta'], config, profile, request['feature_root'], encoder,
            batch_size=request.get('feature_batch_size', 8),
            progress=lambda done, total: update(f'提取图像特征 {done}/{total}', 60 * done / max(1, total)))
        del encoder
        scored = infer_features(model, data['features'], data['valid'], config,
                                batch_size=request.get('score_batch_size', 16), device=request['device'],
                                progress=lambda done, total: update(f'计算进展窗口 {done}/{total}', 60 + 39 * done / max(1, total)))
        timestamps = data['timestamp_ns']
        seconds = (timestamps - int(request['meta']['episode']['start_ns'])) / 1e9
        valid, velocity = scored['valid'], scored['velocity']
        output = Path(request['output'])
        atomic_npz(output.with_suffix('.npz'), timestamp_ns=timestamps, velocity=velocity,
                   valid=valid, coverage=scored['coverage'])
        result = dict(version=VERSION, upstream_commit=UPSTREAM, signature=request['signature'],
                      episode_id=request['episode_id'], model_id=profile['id'], model_label=profile['label'],
                      validation_status=profile['validation_status'], task_scope=profile['task_scope'],
                      domain_note=profile['domain_note'], source_fingerprint=request['source_fingerprint'],
                      model_signature=request['model_signature'], checkpoint_sha256=checkpoint['sha256'],
                      feature_contract=descriptor['contract'], feature_cache=str(feature_path),
                      warp_config=asdict(config), created_at=time.time(), elapsed_s=time.monotonic()-started,
                      frame_count=len(timestamps), feature_valid_count=int(data['valid'].sum()),
                      windows=scored['windows'], shortened_windows=scored['shortened_windows'],
                      span_s_min=scored['span_s_min'], span_s_max=scored['span_s_max'],
                      times_s=seconds.tolist(), velocity=[float(v) if ok else None for v, ok in zip(velocity, valid)],
                      coverage=scored['coverage'].tolist(),
                      summary=curve_summary(seconds, velocity, valid, signature=request['signature']))
        atomic_json(output, result)
        job.update(state='complete', progress=100, message='评分完成', finished_at=time.time())
    except Exception as exc:
        job.update(state='failed', message=str(exc), finished_at=time.time())
        atomic_json(job_path, job)
        raise
    atomic_json(job_path, job)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    run(parser.parse_args().job)
