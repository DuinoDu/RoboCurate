"""Shared multimodal scoring with versioned identity and missing-input review."""
from pathlib import Path
import time
import numpy as np
from .core import WarpConfig,VERSION
from .features import DinoEncoder,extract_mcap,feature_contract,atomic_json,atomic_npz
from .stages_v2 import load_fusion_checkpoint,infer_fusion
from .state_features import native_fingerprint,read_state
from .stage_worker import stage_segments


def check_inputs(request):
    if native_fingerprint(request.get('native_path'))!=request.get('native_fingerprint'):
        raise ValueError('关节状态缓存已变化，请重新提交评分')
    if native_fingerprint(request['source'])!=request['source_fingerprint']:
        raise ValueError('原始记录已变化，请重新解析后评分')


def score_stage(request,update):
    started=time.monotonic();profile=request['profile'];check_inputs(request)
    version=profile.get('stage_version',2)
    if version==4:
        from .stages_v4 import load_stage_v4
        models,saved=load_stage_v4(profile['checkpoint'],profile['checkpoint_sha256'])
    elif version==3:
        from .stages_v3 import load_stage_v3
        models,saved=load_stage_v3(profile['checkpoint'],profile['checkpoint_sha256'])
    else:
        models,saved=load_fusion_checkpoint(profile['checkpoint'],profile['checkpoint_sha256'])
    if (saved['task_id']!=profile.get('task_id') or saved['cameras']!=['head','right_wrist'] or
        saved['cameras']!=profile.get('cameras') or len(saved['feature_contracts'])!=2 or
        saved['state_contract']!=profile.get('state_contract')):
        raise ValueError('多模态阶段模型的任务、相机或状态约定不匹配')
    config=WarpConfig(feature_stride=15)
    for camera,trained in zip(saved['cameras'],saved['feature_contracts']):
        current=dict(profile,camera=camera,view='left' if camera=='head' else 'full')
        if feature_contract(current,config)!=trained:
            raise ValueError('阶段模型的图像特征与训练约定不匹配')
    encoder=DinoEncoder(profile['backbone_dir'],checksums=profile['backbone_sha256'],backbone_id=profile['backbone_id'])
    arrays={};paths=[]
    for index,camera in enumerate(saved['cameras']):
        if camera not in request['meta']['cameras']:
            paths.append(None);continue
        current=dict(profile,camera=camera,view='left' if camera=='head' else 'full')
        data,_,path=extract_mcap(request['source'],request['meta'],config,current,request['feature_root'],encoder,
            batch_size=request.get('feature_batch_size',8),
            progress=lambda done,total:update(f'提取 {camera} 图像特征 {done}/{total}',40*index+40*done/max(1,total)))
        arrays[camera]=data;paths.append(str(path))
    del encoder
    if not arrays:raise ValueError('多模态阶段模型至少需要一个相机')
    timestamps=next(iter(arrays.values()))['timestamp_ns'];n=len(timestamps)
    if not n:raise ValueError('记录没有可采样的时间范围')
    features=np.full((n,2,384),np.nan,np.float32);camera_valid=np.zeros((n,2),bool)
    for j,camera in enumerate(saved['cameras']):
        if camera not in arrays:continue
        data=arrays[camera]
        if not np.array_equal(data['timestamp_ns'],timestamps):raise ValueError('相机没有对齐到相同的时间网格')
        features[:,j]=data['features'];camera_valid[:,j]=data['valid']
    state=read_state(request.get('native_path'),timestamps,request['meta'],saved['state_contract'])
    update('融合视觉与关节状态，识别任务阶段',85)
    if version==4:
        from .stages_v4 import inference_v4
        scored=inference_v4(models,features,camera_valid,state['state'],state['state_valid'],
            saved['priors'],saved['temperature'],saved['review_boundary'])
    elif version==3:
        from .stages_v3 import inference_v3
        scored=inference_v3(models,features,camera_valid,state['state'],state['state_valid'],
            saved['priors'],saved['temperature'],saved['boundary_review'],saved.get('review_guard',False))
    else:
        scored=infer_fusion(models,features,camera_valid,state['state'],state['state_valid'],saved['priors'],saved['temperature'])
    check_inputs(request)
    seconds=(timestamps-request['meta']['episode']['start_ns'])/1e9;valid=scored['valid']
    output=Path(request['output'])
    atomic_npz(output.with_suffix('.npz'),timestamp_ns=timestamps,camera_valid=camera_valid,
        state_valid=state['state_valid'],state_source_timestamp_ns=state['state_source_timestamp_ns'],**scored)
    serial=lambda a:[float(x) if np.isfinite(x) else None for x in a]
    summary=dict(coverage=float(valid.mean()),mean_confidence=float(scored['confidence'][valid].mean()) if valid.any() else None,
        review_fraction=float(scored['review'].mean()),terminal_verdict='unverified',suggestions=[],
        camera_coverage={c:float(camera_valid[:,j].mean()) for j,c in enumerate(saved['cameras'])},
        state_coverage=float(state['state_valid'].mean()),degraded_fraction=float(scored['degraded'].mean()))
    result=dict(version=VERSION,kind='stage_progress',stage_version=version,unit='stage_aligned_proxy',
        signature=request['signature'],episode_id=request['episode_id'],model_id=profile['id'],model_label=profile['label'],
        model_signature=request['model_signature'],source_fingerprint=request['source_fingerprint'],
        native_fingerprint=request.get('native_fingerprint'),modalities=['vision','measured_joint_state'],
        validation_status=profile['validation_status'],task_scope=profile['task_scope'],domain_note=profile['domain_note'],
        checkpoint_sha256=profile['checkpoint_sha256'],feature_contract=dict(cameras=saved['feature_contracts'],state=saved['state_contract']),
        feature_cache=paths,annotation_status=saved['annotation_status'],annotation_sha256=saved['annotation_sha256'],
        protocol_sha256=saved['protocol_sha256'],test_report_sha256=saved['test_report_sha256'],
        created_at=time.time(),elapsed_s=time.monotonic()-started,frame_count=n,feature_valid_count=int(valid.sum()),
        windows=int(valid.sum()),shortened_windows=0,times_s=seconds.tolist(),stage=scored['stage'].tolist(),stage_names=saved['stage_names'],
        progress=serial(scored['progress']),confidence=serial(scored['confidence']),needs_review=scored['review'].tolist(),
        degraded=scored['degraded'].tolist(),summary=summary,
        stage_segments=stage_segments(seconds,scored['stage'],scored['confidence'],scored['review'],request['meta']['episode']['duration_s']))
    if version>=3:
        result.update(review_reasons=scored['review_reasons'].tolist(),boundary_review_enabled=saved.get('boundary_review',False),
            review_guard_enabled=saved.get('review_guard',False),review_guard_report_sha256=saved.get('review_guard_report_sha256'))
    if version==4:
        result.update(boundary_probability=serial(scored['boundary_probability']),
            boundary_head_available=saved['config']['boundary'],learned_boundary_review_enabled=saved['review_boundary'])
    atomic_json(output,result)
    return result
