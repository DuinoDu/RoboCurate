"""Causal multimodal alignment and missing-input behavior, not learned-accuracy tests."""
import copy
import hashlib
import numpy as np
import pytest
from warp_progress.state_features import STATE_KEYS,state_contract,sample_state,read_state
from test_core import real_library


def state_fixture():
    channels={k:dict(names=[f'{k}.{i}' for i in range(w)],unit='rad' if w==29 else '0=open 1=closed')
              for k,w in zip(STATE_KEYS,[29,6,6])}
    native={}
    for k,w in zip(STATE_KEYS,[29,6,6]):
        native[k+'.t']=np.array([100,200,200,400],np.int64)*1_000_000
        native[k+'.v']=np.repeat(np.array([1,2,3,4],np.float32)[:,None],w,axis=1)
    return dict(channels=channels),native


def test_state_alignment_uses_no_future_no_stale_and_last_duplicate():
    meta,native=state_fixture()
    t=np.array([99,100,199,200,300,301,400],np.int64)*1_000_000
    out=sample_state(native,t,state_contract(meta))
    assert out['state_valid'].tolist()==[False,True,True,True,True,False,True]
    np.testing.assert_array_equal(out['state'][out['state_valid'],0],[1,1,3,3,4])
    assert (out['state_source_timestamp_ns'][out['state_valid']]<=t[out['state_valid'],None]).all()
    assert np.isnan(out['state'][~out['state_valid']]).all()


def test_state_missing_channel_is_unknown_schema_changes_rejected(tmp_path):
    meta,native=state_fixture();contract=state_contract(meta);times=np.array([200_000_000])
    del native[STATE_KEYS[1]+'.v']
    assert not sample_state(native,times,contract)['state_valid'].any()
    path=tmp_path/'native.npz';np.savez(path,**native)
    incomplete=copy.deepcopy(meta);del incomplete['channels'][STATE_KEYS[1]]
    assert not read_state(path,times,incomplete,contract)['state_valid'].any()
    assert not read_state(None,times,meta,contract)['state_valid'].any()
    wrong=copy.deepcopy(meta);wrong['channels']['state.q']['names'].reverse()
    with pytest.raises(ValueError,match='名称或单位'):read_state(path,times,wrong,contract)
    native['state.q.t']=native['state.q.t'][::-1]
    with pytest.raises(ValueError,match='时间顺序'):sample_state(native,times,contract)


def test_fusion_missing_modalities_causality_and_roundtrip(tmp_path):
    torch=pytest.importorskip('torch')
    from warp_progress.stages_v2 import FusionStageNet,infer_fusion,load_fusion_checkpoint
    torch.set_num_threads(1);torch.manual_seed(7)
    model=FusionStageNet(stage_tokens=True).eval()
    rng=np.random.default_rng(11)
    x=rng.normal(size=(24,2,384)).astype(np.float32);q=rng.normal(size=(24,41)).astype(np.float32)
    cv=np.ones((24,2),bool);qv=np.ones(24,bool)
    cv[3,0]=False;x[3,0]=np.nan
    qv[5]=False;q[5]=np.nan
    cv[8]=False;x[8]=np.nan
    prior=[.2,.3,.3,.2,0.]
    a=infer_fusion([model],x,cv,q,qv,prior)
    assert a['valid'][3] and a['valid'][5] and a['degraded'][3] and a['degraded'][5]
    assert a['review'][3] and a['review'][5]
    assert a['stage'][8]==-1 and a['review'][8] and np.isnan(a['progress'][8])
    assert np.isfinite(a['probability'][a['valid']]).all()
    x[17:]*=3;q[17:]*=-2
    b=infer_fusion([model],x,cv,q,qv,prior)
    np.testing.assert_allclose(a['probability'][:17],b['probability'][:17],equal_nan=True)
    np.testing.assert_allclose(a['progress'][:17],b['progress'][:17],equal_nan=True)
    path=tmp_path/'model.pt'
    torch.save(dict(kind='stage_progress',version=2,config=model.config,states=[model.state_dict()]),path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    loaded,_=load_fusion_checkpoint(path,digest)
    c=infer_fusion(loaded,x,cv,q,qv,prior)
    np.testing.assert_allclose(b['probability'],c['probability'],equal_nan=True)
    cv[:]=False;x[:]=np.nan
    d=infer_fusion(loaded,x,cv,q,qv,prior)
    assert np.all(d['stage']==-1) and d['review'].all()
    with pytest.raises(ValueError,match='校验失败'):load_fusion_checkpoint(path,'0'*64)


def test_v2_native_cache_changes_signature_and_one_camera_can_continue(real_library,monkeypatch):
    import sys,json
    from progress_service import ProgressService
    from warp_progress.features import atomic_json
    from warp_fixtures import install_profile,MODEL_ID
    lib,ref=real_library;service=lib.progress_service
    p=install_profile(service,sys.executable)
    p.update(kind='stage_progress',stage_version=2,cameras=['head','missing_camera'])
    atomic_json(service.model_root/'profiles.json',dict(models=[p]))
    first=service._request(ref.id,MODEL_ID)
    assert first['native_fingerprint'] is not None
    # No changes to the symlinked real cache: alter only the metadata returned by the fingerprint helper.
    monkeypatch.setattr('progress_service.native_fingerprint',lambda _:dict(size=1,mtime_ns=2))
    second=service._request(ref.id,MODEL_ID)
    assert first['signature']!=second['signature']
    p['cameras']=['absent_a','absent_b'];atomic_json(service.model_root/'profiles.json',dict(models=[p]))
    with pytest.raises(ValueError,match='至少需要一个'):service._request(ref.id,MODEL_ID)
    with pytest.raises(ValueError,match='不能作为 WARP'):service.for_export(ref.id,dict(model_id=MODEL_ID))


def test_v2_worker_rejects_state_change_since_request(tmp_path):
    pytest.importorskip('torch')
    from warp_progress.stage_worker_v2 import check_inputs
    from warp_progress.state_features import native_fingerprint
    source=tmp_path/'source.mcap';source.write_bytes(b'synthetic-test')
    native=tmp_path/'native.npz';native.write_bytes(b'synthetic-test')
    req=dict(source=str(source),source_fingerprint=native_fingerprint(source),
             native_path=str(native),native_fingerprint=native_fingerprint(native))
    check_inputs(req)
    native.write_bytes(b'changed-synthetic-test')
    with pytest.raises(ValueError,match='关节状态缓存已变化'):check_inputs(req)
