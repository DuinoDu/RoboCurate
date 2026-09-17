"""Display evidence must reflect the signed score without changing model outputs."""
import hashlib,json
import numpy as np
import pytest
from progress_service import stage_evidence

def test_probability_and_input_masks_are_causal_display_data_only(tmp_path):
    p=tmp_path/'score.npz';np.savez(p,probability=np.array([[.8,.2],[np.nan,np.nan],[.1,.9]],np.float32),
        camera_valid=np.array([[True,True],[False,False],[True,False]]),state_valid=np.array([True,False,True]))
    result=dict(times_s=[0,.5,1],stage_names=['a','b'],stage=[0,-1,1],confidence=[.8,None,.9])
    before=json.dumps(result);digest=hashlib.sha256(p.read_bytes()).hexdigest();e=stage_evidence(result,p)
    assert e['available'] and e['probability'][1] is None
    np.testing.assert_allclose(e['probability'][2],[.1,.9]);assert e['camera_valid'][2]==[True,False]
    assert e['state_valid']==[True,False,True] and before==json.dumps(result)
    assert digest==hashlib.sha256(p.read_bytes()).hexdigest()

def test_legacy_evidence_does_not_invent_missing_input_masks(tmp_path):
    p=tmp_path/'score.npz';np.savez(p,probability=np.array([[.6,.4]],np.float32))
    e=stage_evidence(dict(times_s=[0],stage_names=['a','b'],stage=[0],confidence=[.6]),p)
    assert e['available'] and 'camera_valid' not in e and 'state_valid' not in e
    np.savez(p,other=[1]);assert not stage_evidence({},p)['available']

@pytest.mark.parametrize('probability,mask,confidence',[
    ([[.7,.3],[.7,.3]],[[True,True]],.7), # wrong length
    ([[.7,.5]],[[True,True]],.7), # not a distribution
    ([[.7,.3]],[[True]],.7), # wrong modality axis
    ([[.7,.3]],[[True,True]],.9), # mismatch with cached confidence
])
def test_invalid_evidence_is_not_displayed_as_model_certainty(tmp_path,probability,mask,confidence):
    p=tmp_path/'score.npz';np.savez(p,probability=np.array(probability,np.float32),camera_valid=np.array(mask,bool))
    with pytest.raises(ValueError,match='阶段'):
        stage_evidence(dict(times_s=[0],stage_names=['a','b'],stage=[0],confidence=[confidence]),p)
