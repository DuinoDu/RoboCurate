"""Time-local evidence for the G1 + BrainCo collection workflow.

Threshold excursions are review candidates, not automatic failure labels.
All event times are relative to the MCAP episode clock, never to a topic's
first sample. Numeric comparisons use native samples and bounded alignment.
"""
import hashlib
import json
import numpy as np

from exporter import align, plan_clip, STATE_KEYS, ACTION_KEYS
from extract import T_ACTION, T_LOWSTATE
from g1_joints import JOINT_GROUPS

VERSION = 2
CACHE_SCHEMA_VERSION = 2
GROUP_LABELS = dict(left_leg='左腿', right_leg='右腿', waist='腰部', left_arm='左臂', right_arm='右臂', left_hand='左手', right_hand='右手')


def interval_end(seconds, hi, end):
    last=float(seconds[hi-1])
    following=float(seconds[hi]) if hi<len(seconds) else float(end)
    # A missing interval does not prove that an excursion persisted through it.
    return following if following-last<=.100001 else last


def spans(mask, seconds, minimum=0, merge_gap=.06, end=None):
    """Return [first,last+1) sample bounds for sustained candidate intervals."""
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(seconds[indices]) > merge_gap) + 1
    result = []
    for block in np.split(indices, breaks):
        lo, hi = int(block[0]), int(block[-1])+1
        last = interval_end(seconds,hi,end if end is not None else seconds[hi-1])
        if last-float(seconds[lo]) >= minimum:
            result.append((lo,hi))
    return result


def coverage_summary(plan, fps):
    valid = plan['valid']
    return dict(samples=len(valid), aligned_samples=int(valid.sum()),
                fraction=float(valid.mean()) if len(valid) else 0, fps=fps,
                invalid_by_channel={k[6:]:int((~v).sum()) for k,v in plan.items() if k.startswith('valid.')},
                note='仅表示数值与时间覆盖；未判定任务成功，也未逐张解码图像。')


def build_diagnostics(cache, rules):
    meta = json.loads((cache/'meta.json').read_text())
    t0 = int(meta['episode']['start_ns'])
    duration = (meta['episode']['end_ns']-t0)/1e9
    signature = hashlib.sha256(json.dumps(dict(version=VERSION,source=meta['source'],rules=rules),sort_keys=True).encode()).hexdigest()[:16]
    events, groups = [], []

    def mode_at(t):
        return {name:next((b['value'] for b in bands if b['start_s'] <= t < b['end_s']), '未记录')
                for name,bands in meta.get('bands',{}).items()}

    def emit(kind, label, start, end, peak, value, unit, detail, channel='', group='', joint=None, threshold=None, severity='review'):
        start, end, peak = max(0,float(start)),min(duration,float(end)),float(peak)
        if end < start:return
        event = dict(kind=kind,label=label,start_s=round(start,6),end_s=round(end,6),peak_s=round(max(start,min(end,peak)),6),
                     value=None if value is None else round(float(value),6),unit=unit,detail=detail,channel=channel,group=group,
                     joint_index=joint,threshold=threshold,severity=severity,context=mode_at(peak))
        event['id'] = hashlib.sha256((signature+json.dumps(event,sort_keys=True)).encode()).hexdigest()[:20]
        events.append(event)

    with np.load(cache/'native.npz',allow_pickle=False) as native:
        # Check every relevant stream from native timestamps. Topic-local
        # offsets in the legacy display statistics are deliberately not used.
        streams = [(T_ACTION,'动作',rules['action_gap_ms']), (T_LOWSTATE,'身体状态',100.)]
        streams += [(c['topic'],name+' 相机',rules['camera_gap_ms']) for name,c in meta['cameras'].items()]
        streams += [(f'/action/brainco/{side}/cmd',GROUP_LABELS[side+'_hand']+'指令',100.) for side in ('left','right')]
        for topic,label,limit in streams:
            cam=next((name for name,c in meta['cameras'].items() if c['topic']==topic),None)
            key='camera.'+cam+'.t' if cam else 'topic:'+topic
            if key not in native:continue
            ts=native[key];gaps=np.diff(ts)/1e6
            for i in np.flatnonzero(gaps > limit):
                a,b=(int(ts[i])-t0)/1e9,(int(ts[i+1])-t0)/1e9
                emit('gap',label+'采样中断',a,b,(a+b)/2,gaps[i],'ms',f'相邻消息间隔超过 {limit:g} ms；查看前后画面与状态。',topic,threshold=limit,severity='warning')

        for key in STATE_KEYS+ACTION_KEYS:
            if key+'.v' not in native:continue
            seconds=(native[key+'.t']-t0)/1e9
            invalid=~np.isfinite(native[key+'.v']).all(axis=1)
            for lo,hi in spans(invalid,seconds,end=duration):
                end=interval_end(seconds,hi,duration)
                emit('invalid','数值无效',seconds[lo],end,seconds[lo],None,'',key+' 存在 NaN/Inf；这些时刻不能作有效训练样本。',key,severity='error')

        if all(k in native for k in ['state.q.t','state.q.v','action.q.t','action.q.v']):
            ts=native['action.q.t'];seconds=(ts-t0)/1e9
            q,valid=align(native['state.q.t'],native['state.q.v'],ts)
            delta=np.abs(q-native['action.q.v']);delta[~valid]=np.nan
            joint_p99=[]
            for j,name in enumerate(meta['joints']['names']):
                v=delta[:,j];finite=np.isfinite(v)
                joint_p99.append(dict(index=j,name=name,p99_rad=float(np.percentile(v[finite],99)) if finite.any() else None))
            for group,lo,hi in JOINT_GROUPS:
                block=delta[:,lo:hi]
                finite=np.isfinite(block).all(axis=1)
                values=np.where(finite, np.max(np.nan_to_num(block,nan=-np.inf),axis=1),np.nan)
                threshold=rules['tracking_p99_rad']
                over=finite & (values > threshold)
                flat=block[np.isfinite(block)]
                worst=max(joint_p99[lo:hi],key=lambda x:x['p99_rad'] or 0)
                groups.append(dict(group=group,label=GROUP_LABELS[group],unit='rad',p99=float(np.percentile(flat,99)) if flat.size else None,
                                   peak=float(np.max(flat)) if flat.size else None,over_fraction=float(over.sum()/max(finite.sum(),1)),
                                   joint_index=worst['index'],joint_name=worst['name']))
                for a,b in spans(over,seconds,minimum=rules.get('tracking_min_s',.25),merge_gap=.06,end=duration):
                    peak=a+int(np.nanargmax(values[a:b]));j=lo+int(np.nanargmax(block[peak]))
                    end=interval_end(seconds,b,duration)
                    emit('tracking',GROUP_LABELS[group]+'持续跟踪偏差',seconds[a],end,seconds[peak],values[peak],'rad',
                         f"峰值关节 {meta['joints']['names'][j]}；持续超过提醒线至少 {rules.get('tracking_min_s',.25):g} 秒。结合动作速度、模式与控制参数复核。",'state.q',group,j,threshold)
            if 'cmd.q.v' in native:
                cmd,valid_cmd=align(native['cmd.q.t'],native['cmd.q.v'],ts,hold=True)
                comparison=np.abs(cmd-native['action.q.v']);v=comparison[valid_cmd & np.isfinite(comparison).all(axis=1)]
                consistency=float(np.percentile(v,99)) if v.size else None
            else:consistency=None
        else:joint_p99=[];consistency=None

        for side in ('left','right'):
            state=f'hand.{side}.state_q';cmd=f'hand.{side}.cmd_q'
            if state+'.v' not in native or cmd+'.v' not in native:continue
            ts=native[cmd+'.t'];seconds=(ts-t0)/1e9
            measured,valid=align(native[state+'.t'],native[state+'.v'],ts)
            delta=np.abs(measured-native[cmd+'.v'])
            values=np.max(np.nan_to_num(delta,nan=-np.inf),axis=1)
            valid &= np.isfinite(delta).all(axis=1)
            threshold=rules.get('hand_error',.3);group=side+'_hand'
            flat=delta[valid]
            groups.append(dict(group=group,label=GROUP_LABELS[group],unit='闭合量',p99=float(np.percentile(flat,99)) if flat.size else None,
                               peak=float(flat.max()) if flat.size else None,over_fraction=float(((values>threshold)&valid).sum()/max(valid.sum(),1)),joint_index=0,joint_name=''))
            for a,b in spans(valid&(values>threshold),seconds,minimum=rules.get('tracking_min_s',.25),merge_gap=.08,end=duration):
                peak=a+int(np.argmax(values[a:b]));j=int(np.argmax(delta[peak]));end=interval_end(seconds,b,duration)
                emit('hand_tracking',GROUP_LABELS[group]+'指令与实测偏差',seconds[a],end,seconds[peak],values[peak],'闭合量',
                     f"BrainCo {meta['joints']['hand_names'][j]}；单位为归一化闭合量，接触物体时也可能产生偏差，需要结合腕部图像。",state,group,j,threshold)

        if 'state.temp.v' in native:
            ts=native['state.temp.t'];seconds=(ts-t0)/1e9;temp=native['state.temp.v']
            values=np.max(np.nan_to_num(temp,nan=-np.inf),axis=1)
            for a,b in spans(values>rules['temperature_c'],seconds,minimum=.2,merge_gap=.04,end=duration):
                peak=a+int(np.argmax(values[a:b]));j=int(np.argmax(temp[peak]));end=interval_end(seconds,b,duration)
                emit('temperature','电机温度提醒',seconds[a],end,seconds[peak],values[peak],'°C',
                     meta['joints']['names'][j]+' 超过提醒线，复核运动负载与采集条件。','state.temp','',j,rules['temperature_c'],'warning')
        try:
            plan,_=plan_clip(native,meta,0,duration,30)
            coverage=coverage_summary(plan,30);valid=plan['valid'];seconds=plan['timestamp_s']
            runs=spans(valid,seconds,minimum=.5,merge_gap=1/30+.0001,end=duration)
            candidates=[dict(start=float(seconds[a]),end=float(seconds[b]) if b<len(seconds) else duration,
                             label=f'覆盖完整区间 {i+1}',grade='B') for i,(a,b) in enumerate(runs)]
        except (KeyError,ValueError) as e:
            coverage=dict(samples=0,aligned_samples=0,fraction=0,fps=30,error=str(e));candidates=[]
    events.sort(key=lambda e:(e['start_s'],e['group'],e['kind']))
    return dict(version=VERSION,signature=signature,source=meta['source'],duration_s=duration,events=events,groups=groups,joints=joint_p99,
                action_lowcmd_p99_rad=consistency,coverage=coverage,candidates=candidates,bands=meta['bands'],
                event_count=len(events),rules=rules,
                note='事件是待复核线索。覆盖完整区间只检查时序和有限数值，不自动判断任务完成、接触效果或图像质量。')
