import hashlib
import json
import sys
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from warp_progress.stages import StageNet, infer_stage_models, load_stage_checkpoint
from warp_progress.stage_worker import stage_segments
from warp_progress.features import atomic_json
from progress_service import ProgressService
from test_core import real_library
from warp_fixtures import install_profile, MODEL_ID


def test_stage_inference_missing_camera_and_causal_future():
    torch.set_num_threads(1);torch.manual_seed(11)
    model=StageNet().eval()
    x=np.random.default_rng(2).normal(size=(24,2,384)).astype(np.float32)
    valid=np.ones(24,bool);valid[10]=False;x[10]=np.nan
    prior=[.2,.3,.3,.2,0.]
    a=infer_stage_models([model],x,valid,prior)
    x[17:]*=-3
    b=infer_stage_models([model],x,valid,prior)
    np.testing.assert_allclose(a['probability'][:17],b['probability'][:17],equal_nan=True)
    assert a['stage'][10]==-1 and a['review'][10] and np.isnan(a['progress'][10])
    assert np.isfinite(a['progress'][valid]).all()
    assert ((a['progress'][valid]>=0)&(a['progress'][valid]<=1)).all()
    with pytest.raises(ValueError,match='camera count'):
        infer_stage_models([model],x[:,:1],valid,prior)


def test_all_missing_images_never_become_a_stage():
    model=StageNet(temporal=False).eval()
    out=infer_stage_models([model],np.full((4,2,384),np.nan,np.float32),np.zeros(4,bool),[.2,.3,.3,.2,0.])
    assert np.all(out['stage']==-1) and out['review'].all() and np.isnan(out['confidence']).all()


def test_model_hash_and_strict_architecture_roundtrip(tmp_path):
    model=StageNet(temporal=False)
    p=tmp_path/'stage.pt'
    torch.save(dict(kind='stage_progress',version=1,config=model.config,states=[model.state_dict()]),p)
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    models,_=load_stage_checkpoint(p,sha)
    assert not models[0].training
    with pytest.raises(ValueError,match='校验失败'):load_stage_checkpoint(p,'0'*64)


def test_stage_segments_keep_missing_gaps_and_post_release_semantics():
    spans=stage_segments(np.arange(6)*.5,np.array([2,2,-1,4,4,4]),
                         np.array([.9,.8,np.nan,.85,.9,.9]),np.array([0,0,1,1,0,0]),2.8)
    assert [(r['start_s'],r['end_s']) for r in spans]==[(0.,1.),(1.5,2.8)]
    assert spans[-1]['stage']==4 and 'success' not in spans[-1]


def test_stage_profile_requires_all_cameras_and_cannot_export_warp_weights(real_library):
    lib,ref=real_library;s=lib.progress_service
    p=install_profile(s,sys.executable)
    p.update(kind='stage_progress',cameras=['head','missing_camera'])
    atomic_json(s.model_root/'profiles.json',dict(models=[p]))
    with pytest.raises(ValueError,match='缺少阶段模型'):s._request(ref.id,MODEL_ID)
    assert s.models()['models'][0]['supports_weights'] is False
    with pytest.raises(ValueError,match='阶段进度不能'):s.for_export(ref.id,dict(model_id=MODEL_ID))
    with pytest.raises(ValueError,match='单位不同'):
        ProgressService.apply_to_clip(dict(kind='stage_progress'),{}, {}, {},30)
