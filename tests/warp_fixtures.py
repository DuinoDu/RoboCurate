"""Explicitly synthetic scores for isolated API/export/UI acceptance tests.

These fixtures must never be written to the user's production workspace.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
import time
import numpy as np
from warp_progress.core import WarpConfig, curve_summary
from warp_progress.features import atomic_json, atomic_npz

MODEL_ID='synthetic-test-only'


def install_profile(service, python):
    root=service.model_root
    head=root/'test-head.pt';head.write_bytes(b'fixture only - not a trained checkpoint')
    backbone=root/'test-backbone';backbone.mkdir(exist_ok=True)
    (backbone/'config.json').write_text('{}')
    (backbone/'model.safetensors').write_bytes(b'fixture only')
    env=service.library.workspace/'warp-env/bin';env.mkdir(parents=True,exist_ok=True)
    if not (env/'python').exists():(env/'python').symlink_to(python)
    profile=dict(id=MODEL_ID,label='测试曲线（非模型输出）',checkpoint=head.name,
                 checkpoint_sha256=hashlib.sha256(head.read_bytes()).hexdigest(),backbone_dir=backbone.name,
                 backbone_revision='test-fixture',camera='head',view='left',crop='squash',
                 backbone_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in backbone.iterdir()},
                 validation_status='test_fixture',task_scope='automated tests only',
                 domain_note='自动化验收用合成曲线，不代表模型推理或任务判别效果。')
    atomic_json(root/'profiles.json',dict(version=1,models=[profile]))
    return profile


def install_score(service, ep):
    req=service._request(ep,MODEL_ID)
    episode=req['meta']['episode'];duration=episode['duration_s']
    times=np.arange(int(duration*30))/30
    timestamps=int(episode['start_ns'])+np.rint(times*1e9).astype(np.int64)
    velocity=(1.5+.25*np.sin(times)).astype(np.float32)
    velocity[(times>=10)&(times<13)]=0
    velocity[(times>=16)&(times<19)]=-1
    valid=np.ones(len(times),bool);valid[(times>=5.5)&(times<5.8)]=False
    velocity[~valid]=np.nan
    config=asdict(WarpConfig())
    result=dict(version=1,signature=req['signature'],episode_id=ep,model_id=MODEL_ID,
                model_label='测试曲线（非模型输出）',validation_status='test_fixture',task_scope='automated tests only',
                source_fingerprint=req['source_fingerprint'],model_signature=req['model_signature'],
                checkpoint_sha256=req['profile']['checkpoint_sha256'],
                feature_contract=dict(fixture=True,fps=30,feature_stride=1),warp_config=config,
                domain_note=req['profile']['domain_note'],created_at=time.time(),elapsed_s=0,
                frame_count=len(times),feature_valid_count=int(valid.sum()),windows=20,shortened_windows=0,
                span_s_min=15.5,span_s_max=15.5,times_s=times.tolist(),
                velocity=[float(v) if ok else None for v,ok in zip(velocity,valid)],coverage=valid.astype(int).tolist(),
                summary=curve_summary(times,velocity,valid,signature=req['signature']))
    atomic_npz(Path(req['output']).with_suffix('.npz'),timestamp_ns=timestamps,velocity=velocity,valid=valid,coverage=valid.astype(int))
    atomic_json(req['output'],result)
    return result
