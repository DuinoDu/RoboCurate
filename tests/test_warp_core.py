from dataclasses import replace
import numpy as np
import pytest
from warp_progress.core import (WarpConfig,sample_warp,relative_targets,two_hot,plan_windows,
                                aggregate_windows,align_velocity,chunk_weights,curve_summary)


@pytest.mark.parametrize('feature_stride',[1,3,15])
def test_ar_sampler_exact_upstream_parity(monkeypatch,feature_stride):
    from vendor.warp_rm.samplers import ARSampler
    factory=np.random.default_rng
    config=WarpConfig(feature_stride=feature_stride)
    sampler=ARSampler(window_size=32,feature_stride=feature_stride)
    for seed in range(100):
        for size in (45,100,1000,3000):
            rng=factory(seed)
            monkeypatch.setattr(np.random,'default_rng',lambda:factory(seed))
            assert sample_warp(size,config,rng).tolist()==sampler.sample_indices(size)


def test_relative_labels_canonical_forward_reverse_and_pause():
    config=WarpConfig()
    indices=np.arange(32)*15+100
    assert np.allclose(relative_targets(indices,config),np.linspace(0,1,32))
    assert np.allclose(relative_targets(indices[::-1],config),-np.linspace(0,1,32))
    assert np.all(relative_targets(np.full(32,45),config)==0)
    assert np.array_equal(relative_targets(indices//3,replace(config,feature_stride=3)),relative_targets(indices,config))


def test_two_hot_mass_mean_and_endpoints():
    values=np.array([-10,-3,-2.1,0,1,2.999,3,10],np.float32)
    distribution=two_hot(values)
    assert np.allclose(distribution.sum(-1),1)
    assert np.all((distribution>0).sum(-1)<=2)
    assert np.allclose(distribution@np.linspace(-3,3,30),np.clip(values,-3,3),atol=8e-7)
    with pytest.raises(ValueError):two_hot([np.nan])


@pytest.mark.parametrize('length',[33,100,465,466,467,900])
def test_window_planning_and_overlap_match_release(length):
    pytest.importorskip('torch')
    from vendor.warp_rm.inference import _plan_episode_windows,_aggregate_relative
    config=WarpConfig()
    ids,scales,_=plan_windows(np.ones(length,bool),config)
    reference,step,scale=_plan_episode_windows(length,32,15)
    assert ids.tolist()==reference and np.all(scales==scale)
    if not len(ids):return
    pred=np.random.default_rng(4).normal(size=ids.shape).astype(np.float32)
    velocity,count=aggregate_windows(ids,pred,length,scales)
    _,expected=_aggregate_relative(reference,pred,length,32,step,scale)
    assert np.allclose(velocity[count>0],expected[count>0],atol=1e-5)
    assert np.isnan(velocity[count==0]).all()  # explicit deployment change


def test_missing_frames_split_windows_and_weight_eligibility():
    mask=np.ones(1000,bool);mask[490:510]=False
    ids,scales,_=plan_windows(mask,WarpConfig())
    assert mask[ids].all()
    assert not ((ids[:,0]<490)&(ids[:,-1]>=510)).any()
    prediction=(ids-ids[:,:1])/465
    velocity,count=aggregate_windows(ids,prediction,1000,scales)
    assert (count[490:510]==0).all()
    weights,eligible=chunk_weights(velocity,count>0,horizon=30,threshold=.9)
    assert not eligible[461:510].any()
    assert np.allclose(weights[eligible],1)
    assert not weights[-29:].any()


def test_strict_terminal_gate_continuous_binary_and_tail_padding():
    velocity=np.array([3,3,1,2,0,4],np.float32)
    valid=np.ones(6,bool)
    w,eligible=chunk_weights(velocity,valid,horizon=3,threshold=1)
    assert w.tolist()==[0,2,0,4,0,0]
    assert eligible.tolist()==[True,True,True,True,False,False]
    binary,_=chunk_weights(velocity,valid,3,1,'binary',pad_tail=True)
    assert binary.tolist()==[0,1,0,1,1,1]
    valid[4]=False
    w,_=chunk_weights(velocity,valid,3,1)
    assert w.tolist()==[0,2,0,0,0,0]


def test_nan_is_uncovered_and_epoch_alignment_is_precise():
    base=1_789_033_864_925_712_875
    t=np.array([base,base+10,base+20],np.int64)
    v=np.array([1,np.nan,2],np.float32)
    grid=base+np.array([-1,0,5,10,15,20,21],np.int64)
    out,valid=align_velocity(t,v,[True,True,True],grid,6)
    assert valid.tolist()==[False,True,True,False,False,True,False]
    assert out[2]==1 and out[5]==2
    with pytest.raises(ValueError):align_velocity(t[::-1],v,[True]*3,grid,10)


def test_suggestions_have_stable_ids_and_no_fake_completion():
    t=np.arange(90)/30
    v=np.r_[np.ones(30),np.zeros(30),-np.ones(30)]
    r=curve_summary(t,v,np.ones(90,bool),signature='a')
    assert [s['kind'] for s in r['suggestions']]==['forward','stall','regression']
    assert r==curve_summary(t,v,np.ones(90,bool),signature='a')
    assert r['suggestions'][0]['id']!=curve_summary(t,v,np.ones(90,bool),signature='b')['suggestions'][0]['id']
    assert 'completion' not in r


def test_camera_preprocess_separates_stereo_and_rgb():
    pytest.importorskip('cv2')
    from warp_progress.features import preprocess_rgb,MEAN,STD
    rgb=np.zeros((10,40,3),np.uint8);rgb[:,:20,0]=255;rgb[:,20:,2]=255
    left=preprocess_rgb(rgb,'left','squash');right=preprocess_rgb(rgb,'right','center')
    assert left.shape==right.shape==(3,224,224)
    assert np.allclose(left[:,0,0],(np.array([1,0,0])-MEAN)/STD)
    assert np.allclose(right[:,0,0],(np.array([0,0,1])-MEAN)/STD)
