import json
from pathlib import Path

import numpy as np
import pytest

from diagnostics import build_diagnostics, spans
from exporter import plan_clip, preflight
from g1_joints import BODY_JOINT_NAMES, HAND_JOINT_NAMES
from quality import DEFAULT_RULES
from storage import Store, Conflict
from studio import StudioLibrary


@pytest.fixture
def diagnostic_cache(tmp_path):
    t0=1_789_000_000_123_456_789
    seconds=np.arange(0,4,.02)
    t=t0+np.rint(seconds*1e9).astype(np.int64)
    native={}
    channels={}
    for key in ['state.q','action.q','state.temp','hand.left.state_q','hand.right.state_q','hand.left.cmd_q','hand.right.cmd_q']:
        width=6 if key.startswith('hand.') else 29
        values=np.zeros((len(t),width),dtype=np.float32)
        if key=='state.q':
            values[(seconds>=1)&(seconds<1.6),26]=1.2
            values[5,0]=2  # one brief spike must not become a sustained event
        if key=='state.temp':values[:]=45;values[(seconds>=2)&(seconds<2.6),3]=72
        native[key+'.t']=t;native[key+'.v']=values
        channels[key]=dict(names=list(HAND_JOINT_NAMES if width==6 else BODY_JOINT_NAMES))
    cams={}
    for cam in ['head','left_wrist','right_wrist']:
        ct=np.arange(.5,4,1/30)
        if cam=='head':ct=ct[(ct<.8)|(ct>=1.3)]
        native['camera.'+cam+'.t']=t0+np.rint(ct*1e9).astype(np.int64)
        cams[cam]=dict(topic=f'/observations/camera/{cam}/color/image')
    native['topic:/action/humanoid']=t
    meta=dict(episode=dict(start_ns=t0,end_ns=t0+4_000_000_000,duration_s=4),source=dict(path='synthetic.mcap',mtime_ns=1),
              channels=channels,cameras=cams,joints=dict(names=list(BODY_JOINT_NAMES),hand_names=list(HAND_JOINT_NAMES)),
              bands=dict(policy_mode=[dict(start_s=0,end_s=4,value='motion_enabled')],robot_state=[dict(start_s=0,end_s=4,value='POLICY')]))
    np.savez(tmp_path/'native.npz',**native)
    (tmp_path/'meta.json').write_text(json.dumps(meta))
    return tmp_path,meta,native


def test_events_use_episode_clock_and_body_joint(diagnostic_cache):
    cache,_,_=diagnostic_cache
    result=build_diagnostics(cache,DEFAULT_RULES)
    tracking=[e for e in result['events'] if e['kind']=='tracking']
    assert len(tracking)==1
    event=tracking[0]
    assert event['group']=='right_arm' and event['joint_index']==26
    assert event['start_s']==1.0 and event['end_s']==1.6
    assert event['context']['policy_mode']=='motion_enabled'
    gap=next(e for e in result['events'] if e['kind']=='gap')
    assert gap['start_s']>.7 and gap['end_s']>=1.29  # never relative to camera's 0.5s start
    thermal=next(e for e in result['events'] if e['kind']=='temperature')
    assert thermal['joint_index']==3 and thermal['value']==72
    assert thermal['start_s']==2.0


def test_single_spike_before_gap_is_not_sustained():
    assert spans(np.array([True,False]),np.array([0.,3.]),minimum=.25,end=4)==[]


def test_rule_change_invalidates_event_ids(diagnostic_cache):
    cache,_,_=diagnostic_cache
    a=build_diagnostics(cache,DEFAULT_RULES)
    b=build_diagnostics(cache,{**DEFAULT_RULES,'tracking_p99_rad':.7})
    assert a['signature']!=b['signature']
    assert not ({e['id'] for e in a['events']} & {e['id'] for e in b['events']})


def test_alignment_plan_keeps_causal_frames_and_clip_boundaries(diagnostic_cache):
    _,meta,native=diagnostic_cache
    plan,_=plan_clip(native,meta,.5,2,30)
    assert plan['action'].shape==(45,41)
    valid=plan['valid.camera.head'];stamp=plan['camera.head.timestamp_ns']
    assert np.all(stamp[valid]<=plan['timestamp_ns'][valid])
    assert np.all(stamp[valid]>=meta['episode']['start_ns']+500_000_000)
    assert not plan['valid'][(plan['timestamp_s']>1)&(plan['timestamp_s']<1.2)].any()
    with pytest.raises(ValueError):plan_clip(native,meta,2,1,30)


def test_old_review_schema_retains_new_decisions(tmp_path):
    s=Store(tmp_path/'reviews.sqlite3')
    with s.connect() as db:db.execute('INSERT INTO reviews VALUES (?,?)',('old',json.dumps(dict(grade='B',revision=2))))
    assert s.all()['old']['event_decisions']=={}
    key='a'*20
    saved=s.save_many([('old',dict(revision=2,event_decisions={key:'accepted'}))])
    assert saved['old']['grade']=='B' and saved['old']['event_decisions'][key]=='accepted'
    assert saved['old']['revision']==3


def test_interrupted_export_is_visible_after_restart(tmp_path):
    lib=StudioLibrary(tmp_path/'workspace')
    lib.persist_job(dict(id='export_interrupted',state='running',message='writing',progress=20))
    lib.recover_jobs()
    jobs=lib.exports()
    assert jobs[0]['state']=='error' and '中断' in jobs[0]['message']
    assert not list(lib.export_root.glob('*.tmp'))
    lib.pool.shutdown();lib.export_pool.shutdown()
