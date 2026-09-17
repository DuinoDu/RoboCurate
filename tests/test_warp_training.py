"""Optimization and checkpoint checks. Synthetic fixtures are not real-video evidence."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import json
import numpy as np
import pytest
from warp_progress.core import WarpConfig,two_hot
from warp_progress.features import atomic_json,atomic_npz


def feature_manifest(tmp_path):
    rows=[]
    for i,split in enumerate(('train','validation')):
        t=np.linspace(0,1,200,dtype=np.float32)
        x=np.stack([t,t**2,np.sin(t),np.cos(t),np.sin(2*t),np.cos(2*t),np.ones_like(t)*i/10,t**3],axis=1)
        path=tmp_path/(split+'.npz')
        atomic_npz(path,features=x,valid=np.ones(len(x),bool))
        atomic_json(path.with_suffix('.json'),dict(contract=dict(dimension=8,fps=30,feature_stride=1,source='synthetic unit test')))
        rows.append(dict(id=split,split=split,features=path.name,source_sha256=hashlib.sha256(split.encode()).hexdigest(),
                         task='synthetic temporal ramp; test only'))
    path=tmp_path/'manifest.json';atomic_json(path,dict(episodes=rows))
    return path


def test_feature_split_leakage_is_rejected(tmp_path):
    pytest.importorskip('torch')
    from warp_progress.train import load_manifest
    path=feature_manifest(tmp_path)
    m=json.loads(path.read_text());m['episodes'][1]['source_sha256']=m['episodes'][0]['source_sha256'];atomic_json(path,m)
    with pytest.raises(ValueError,match='leaks'):load_manifest(path,WarpConfig())


def test_training_optimizes_and_safe_checkpoint_roundtrips(tmp_path):
    torch=pytest.importorskip('torch')
    from warp_progress.train import train
    from warp_progress.model import load_model,infer_features
    from vendor.warp_rm.loss import soft_bins
    values=np.linspace(-3.1,3.1,200,dtype=np.float32)
    assert np.allclose(two_hot(values),soft_bins(torch.from_numpy(values),-3,3,30).numpy(),atol=1e-6)
    path=feature_manifest(tmp_path)
    args=argparse.Namespace(manifest=str(path),output=str(tmp_path/'run'),steps=80,batch_size=8,lr=.002,
                            warmup_steps=0,eval_every=40,eval_batches=2,seed=42,feature_stride=1,source_standard_stride=15,
                            d_model=32,n_heads=4,n_layers=2,init=None,device='cpu',threads=2)
    run=train(args)
    assert run['state']=='complete' and run['paper_architecture'] is False
    assert run['validation'][-1]['cross_entropy'] < run['validation'][0]['cross_entropy']
    model,config,meta=load_model(tmp_path/'run/best.pt',expected_sha256=run['best_sha256'])
    with np.load(tmp_path/'validation.npz') as a:r=infer_features(model,a['features'],a['valid'],config,batch_size=16)
    assert r['valid'].sum()>150 and np.isfinite(r['velocity'][r['valid']]).all()
    with pytest.raises(ValueError,match='校验'):load_model(tmp_path/'run/best.pt',expected_sha256='0'*64)


def test_published_checkpoint_strict_load_and_forward():
    torch=pytest.importorskip('torch')
    from warp_progress.model import load_model
    app=Path(__file__).resolve().parents[1]
    path=app/'workspace/warp/models/paper_sim_sss15.pt'
    if not path.exists():pytest.skip('official model has not been downloaded')
    torch.set_num_threads(2)
    model,config,meta=load_model(path,expected_sha256='9c74aa3934b12dd6b169f8945dc2501a8c625ef0becb9f27e4f297725e2c775f')
    assert config==WarpConfig() and len(model.transformer.layers)==12 and meta['step']==14400
    with torch.inference_mode():value,_,logits=model(torch.zeros(1,32,768))
    assert value.shape==(1,32) and logits.shape==(1,32,30) and torch.isfinite(value).all()


def test_bc_weights_control_chunk_gradient():
    torch=pytest.importorskip('torch')
    from warp_progress.bc import weighted_bc_loss
    prediction=torch.ones(2,3,4,requires_grad=True)
    loss=weighted_bc_loss(prediction,torch.zeros_like(prediction),torch.tensor([0.,2.]))
    loss.backward()
    assert float(loss.detach())==1 and torch.count_nonzero(prediction.grad[0])==0
    assert torch.allclose(prediction.grad[1],torch.full((3,4),1/6))


def test_backbone_adapter_loads_local_transformers_model(tmp_path):
    torch=pytest.importorskip('torch')
    transformers=pytest.importorskip('transformers')
    from warp_progress.features import DinoEncoder
    from warp_progress.model import file_sha256
    torch.set_num_threads(2)
    # Explicit small random architecture: tests the API only, not pretrained quality.
    config=transformers.DINOv3ViTConfig(hidden_size=768,num_hidden_layers=1,num_attention_heads=12,intermediate_size=1536)
    model=transformers.DINOv3ViTModel(config);model.save_pretrained(tmp_path/'random-test-backbone');del model
    root=tmp_path/'random-test-backbone'
    checksums={p.name:file_sha256(p) for p in (root/'config.json',root/'model.safetensors')}
    encoder=DinoEncoder(root,checksums=checksums)
    features=encoder([np.zeros((3,224,224),np.float32)])
    assert features.shape==(1,768) and np.isfinite(features).all()


def test_open_backbone_api_and_dimension_guard(tmp_path):
    torch=pytest.importorskip('torch')
    transformers=pytest.importorskip('transformers')
    from warp_progress.features import DinoEncoder,OPEN_BACKBONE_ID
    torch.set_num_threads(2)
    config=transformers.Dinov2Config(hidden_size=384,num_hidden_layers=1,num_attention_heads=6,intermediate_size=768)
    model=transformers.Dinov2Model(config);model.save_pretrained(tmp_path/'random-dinov2');del model
    encoder=DinoEncoder(tmp_path/'random-dinov2',backbone_id=OPEN_BACKBONE_ID)
    features=encoder([np.zeros((3,224,224),np.float32)])
    assert features.shape==(1,384) and np.isfinite(features).all()
    with pytest.raises(ValueError,match='feature contract'):
        DinoEncoder(tmp_path/'random-dinov2')


def test_scoring_never_silently_changes_visual_training_contract():
    from warp_progress.features import OPEN_BACKBONE_ID,feature_contract,validate_training_contract
    profile=dict(backbone_id=OPEN_BACKBONE_ID,backbone_revision='test',backbone_sha256={},
                 camera='head',view='left',crop='squash')
    config=WarpConfig()
    with pytest.raises(ValueError,match='DINOv3'):
        validate_training_contract(profile,config,{})
    original=feature_contract(profile,config)
    checkpoint=dict(provenance=dict(feature_contract=original))
    assert validate_training_contract(profile,config,checkpoint)==original
    with pytest.raises(ValueError,match='crop'):
        validate_training_contract({**profile,'crop':'center'},config,checkpoint)
    public={**original,'camera':'top_camera-images-rgb','view':'full','decoder':'OpenCV-RGB'}
    checkpoint=dict(provenance=dict(feature_contract=public))
    with pytest.raises(ValueError,match='camera'):
        validate_training_contract(profile,config,checkpoint)
    assert validate_training_contract({**profile,'training_feature_contract':public},config,checkpoint)==original
    with pytest.raises(ValueError,match='backbone_revision'):
        validate_training_contract({**profile,'backbone_revision':'different','training_feature_contract':public},config,checkpoint)
