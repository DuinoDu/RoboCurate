import numpy as np
import pytest
from trajectories import wrist_positions, segmented_track, build_trajectories
from urdf import parse_urdf
from pathlib import Path


def chain():
    return dict(root='root', joints=[dict(name='shoulder_joint',type='revolute',parent='root',child='arm',xyz=[1,0,0],rpy=[0,0,0],axis=[0,0,1]),dict(name='tip_fixed',type='fixed',parent='arm',child='tip',xyz=[1,0,0],rpy=[0,0,0],axis=[0,0,1])])


def test_fk_known_chain_and_invalid_ancestor():
    p,v=wrist_positions(chain(),['shoulder','unrelated'],np.array([[0,np.nan],[np.pi/2,0],[np.nan,0]]),'tip')
    np.testing.assert_allclose(p[:2],[[2,0,0],[1,1,0]],atol=1e-12)
    assert v.tolist()==[True,True,False] and np.isnan(p[2]).all()


def test_origin_rotation_precedes_joint_and_preserves_length():
    d=chain();d['joints'][0]['rpy']=[np.pi/2,0,0]
    p,_=wrist_positions(d,['shoulder'],[[np.pi/2]],'tip')
    np.testing.assert_allclose(p,[[1,0,1]],atol=1e-12)


def test_no_line_across_invalid_gap_or_duplicate_even_after_decimation():
    base=1_789_033_864_925_712_875
    t=base+np.array([0,10,20,30,40,500,510,510,520])*1_000_000
    v=np.ones(9,dtype=bool);v[2]=False
    p=np.zeros((9,3));p[2]=np.nan
    r=segmented_track(t,p,v,base)
    assert len(r['segments'])==4
    assert [s['times_s'] for s in r['segments']]==[[0,.01],[.03,.04],[.5,.51],[.51,.52]]
    assert r['invalid_samples']==1 and r['gap_count']==1
    with pytest.raises(ValueError):segmented_track(t[::-1],p,v,base)


def test_motion_facts_do_not_count_jump_across_gap():
    t=np.array([0,50,100,1000,1050])*1_000_000
    p=np.array([[0,0,0],[1,0,0],[2,0,0],[100,0,0],[101,0,0]])
    r=segmented_track(t,p,np.ones(5,dtype=bool),0)
    assert r['motion']['travel_m']==3 and r['motion']['observed_s']==pytest.approx(.15)


def test_real_g1_symmetry_zero_pose_and_missing_foot_is_explicit():
    d=parse_urdf(Path('robot/g1_29dof_rev_1_0.urdf'))
    names=[j['name'][:-6] for j in d['joints'] if j['type']=='revolute']
    left,_=wrist_positions(d,names,np.zeros((1,len(names))),'left_wrist_yaw_link')
    right,_=wrist_positions(d,names,np.zeros((1,len(names))),'right_wrist_yaw_link')
    np.testing.assert_allclose(left[0]*[1,-1,1],right[0],atol=1e-5)
    result=build_trajectories({},dict(joints=dict(names=names),timeline=dict(t0_ns=0)),d)
    assert len(result['unavailable'])==8 and result['tracks']==[] and result['frame']=='pelvis' and not result['imu_applied']
