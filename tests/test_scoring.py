import numpy as np
from scoring import technical_rating, WEIGHTS
from grading import grade_data, WIDTHS
from quality import DEFAULT_RULES


def data():
    t0=1_789_000_000_123_456_789;duration=2
    native={}
    for k,width in WIDTHS.items():
        hz=500 if k=='state.q' else 50
        native[k+'.t']=t0+np.arange(duration*hz,dtype=np.int64)*int(1e9/hz)
        native[k+'.v']=np.zeros((duration*hz,width),dtype=np.float32)
    cams={n:dict(count=60) for n in ['head','left_wrist','right_wrist']}
    for n in cams:native['camera.'+n+'.t']=t0+np.rint(np.arange(60)*1e9/30).astype(np.int64)
    meta=dict(episode=dict(start_ns=t0,end_ns=t0+duration*1_000_000_000),cameras=cams)
    info=dict(cameras=cams)
    return info,native,meta


def rate(info,native,meta):
    g=grade_data(info,[],native=native,meta=meta)
    return technical_rating(g,native,meta,DEFAULT_RULES)


def test_score_is_weighted_measured_evidence_not_issue_count():
    info,native,meta=data();r=rate(info,native,meta)
    assert r['state']=='measured' and 9.9<r['score']<=10
    assert r['max_score']==10 and r['version']==3 and r['method_version']==2
    assert all(0<=d['score']<=10 for d in r['dimensions'])
    assert r['score']==round(sum(d['score']*WEIGHTS[d['key']] for d in r['dimensions']),2)
    g=grade_data(info,[dict(code='operator_failure',level='review',label='人工标记',detail='失败')],native=native,meta=meta)
    assert technical_rating(g,native,meta,DEFAULT_RULES)['score']==r['score']


def test_degrading_motion_or_camera_coverage_reduces_score():
    info,native,meta=data();baseline=rate(info,native,meta)
    native['action.q.v'][:,15:22]=1
    worse=rate(info,native,meta)
    assert worse['score']<baseline['score']
    tracking=next(d for d in worse['dimensions'] if d['key']=='tracking')
    assert next(g for g in tracking['evidence']['groups'] if g['label']=='左臂')['fraction']==0
    native['camera.head.t']=native['camera.head.t'][15:]
    assert rate(info,native,meta)['score']<worse['score']


def test_missing_or_unknown_dimensions_never_receive_clean_score():
    info,native,meta=data()
    pending=grade_data(info,[])
    assert technical_rating(pending)['score'] is None
    del info['cameras']['right_wrist']
    r=rate(info,native,meta)
    assert r['state']=='blocked' and r['score'] is None and r['grade_gate']=='D'
    assert r['max_score']==10


def test_nonfinite_and_timestamp_reversal_cannot_hide_in_average():
    info,native,meta=data();native['state.q.v'][10,0]=np.nan
    r=rate(info,native,meta)
    assert r['grade_gate']=='C'
    native['action.q.t'][[10,11]]=native['action.q.t'][[11,10]]
    r=rate(info,native,meta)
    assert r['state']=='unavailable' and r['score'] is None
