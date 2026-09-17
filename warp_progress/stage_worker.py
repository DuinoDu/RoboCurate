"""Offline task-stage inference using the same feature and model code as evaluation."""
from pathlib import Path
import time
import numpy as np
from .core import WarpConfig, VERSION
from .features import DinoEncoder, extract_mcap, feature_contract, atomic_json, atomic_npz
from .stages import load_stage_checkpoint, infer_stage_models


def stage_segments(times, stages, confidence, review, duration):
    segments = []
    start = 0
    for stop in range(1, len(stages)+1):
        if stop < len(stages) and stages[stop] == stages[start]:
            continue
        if stages[start] >= 0:
            segments.append(dict(stage=int(stages[start]), start_s=float(times[start]),
                end_s=float(times[stop]) if stop < len(times) else float(min(duration,times[-1]+.5)),
                mean_confidence=float(np.mean(confidence[start:stop])),
                review_fraction=float(np.mean(review[start:stop]))))
        start = stop
    return segments


def score_stage(request, update):
    started = time.monotonic()
    profile = request['profile']
    models, saved = load_stage_checkpoint(profile['checkpoint'], profile['checkpoint_sha256'])
    if saved['task_id'] != profile.get('task_id') or saved['cameras'] != profile.get('cameras'):
        raise ValueError('阶段模型的任务或相机配置不匹配')
    if len(saved['cameras']) != saved['config']['views'] or len(saved['feature_contracts']) != len(saved['cameras']):
        raise ValueError('阶段模型的相机特征约定不完整')
    config = WarpConfig(feature_stride=15)
    descriptors, arrays, paths = [], [], []
    for camera, trained in zip(saved['cameras'], saved['feature_contracts']):
        current = dict(profile, camera=camera, view='left' if camera == 'head' else 'full')
        if feature_contract(current,config) != trained:
            raise ValueError('阶段模型的图像特征与训练约定不匹配')
    encoder = DinoEncoder(profile['backbone_dir'], checksums=profile['backbone_sha256'],backbone_id=profile['backbone_id'])
    for index,camera in enumerate(saved['cameras']):
        current = dict(profile, camera=camera, view='left' if camera == 'head' else 'full')
        data, descriptor, path = extract_mcap(request['source'],request['meta'],config,current,
            request['feature_root'],encoder,batch_size=request.get('feature_batch_size',8),
            progress=lambda done,total:update(f'提取 {camera} 图像特征 {done}/{total}',
                80*(index+done/max(1,total))/len(saved['cameras'])))
        arrays.append(data);descriptors.append(descriptor['contract']);paths.append(str(path))
    del encoder
    for data in arrays[1:]:
        if not np.array_equal(data['timestamp_ns'],arrays[0]['timestamp_ns']):
            raise ValueError('多个相机没有对齐到相同的时间网格')
    features = np.stack([d['features'] for d in arrays],1)
    valid = np.stack([d['valid'] for d in arrays],1).all(1)
    update('识别任务阶段并估计不确定片段',85)
    scored = infer_stage_models(models,features,valid,saved['priors'],saved['temperature'])
    timestamps = arrays[0]['timestamp_ns']
    seconds = (timestamps-request['meta']['episode']['start_ns'])/1e9
    output = Path(request['output'])
    atomic_npz(output.with_suffix('.npz'), timestamp_ns=timestamps, **scored)
    serial = lambda values:[float(v) if np.isfinite(v) else None for v in values]
    summary = dict(coverage=float(valid.mean()),mean_confidence=float(scored['confidence'][valid].mean()) if valid.any() else None,
        review_fraction=float(scored['review'].mean()),terminal_verdict='unverified',suggestions=[])
    result = dict(version=VERSION,kind='stage_progress',unit='stage_aligned_proxy',signature=request['signature'],
        episode_id=request['episode_id'],model_id=profile['id'],model_label=profile['label'],
        model_signature=request['model_signature'],source_fingerprint=request['source_fingerprint'],
        validation_status=profile['validation_status'],task_scope=profile['task_scope'],domain_note=profile['domain_note'],
        checkpoint_sha256=profile['checkpoint_sha256'],feature_contract={'cameras':descriptors},feature_cache=paths,
        annotation_status=saved['annotation_status'],annotation_sha256=saved['annotation_sha256'],
        protocol_sha256=saved['protocol_sha256'],test_report_sha256=saved['test_report_sha256'],
        created_at=time.time(),elapsed_s=time.monotonic()-started,frame_count=len(seconds),feature_valid_count=int(valid.sum()),
        windows=int(valid.sum()),shortened_windows=0,times_s=seconds.tolist(),stage=scored['stage'].tolist(),
        stage_names=saved['stage_names'],progress=serial(scored['progress']),confidence=serial(scored['confidence']),
        needs_review=scored['review'].tolist(),summary=summary,
        stage_segments=stage_segments(seconds,scored['stage'],scored['confidence'],scored['review'],request['meta']['episode']['duration_s']))
    atomic_json(output,result)
    return result
