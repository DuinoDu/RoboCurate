"""Explainable technical QC, separate from human task-success judgements."""
import json
from pathlib import Path
import numpy as np
from mcap.reader import make_reader
from extract import T_ACTION, T_LOWSTATE
from grading import grade_data
from scoring import technical_rating

DEFAULT_RULES = dict(min_duration_s=5.0, camera_min_hz=24.0, camera_gap_ms=150.0,
                     state_min_hz=300.0, action_min_hz=40.0, action_gap_ms=100.0,
                     tracking_p99_rad=0.5, temperature_c=70.0, tracking_min_s=.25, hand_error=.3)
RULE_LABELS = dict(min_duration_s='最短记录时长（秒）', camera_min_hz='相机最低平均帧率（Hz）',
                   camera_gap_ms='相机最大间隔（ms）', state_min_hz='身体状态最低频率（Hz）',
                   action_min_hz='动作最低频率（Hz）', action_gap_ms='动作最大间隔（ms）',
                   tracking_p99_rad='关节跟踪误差提醒线（rad）', temperature_c='电机最高温度（°C）',
                   tracking_min_s='持续偏差的最短时长（秒）', hand_error='双手闭合量误差提醒线（0–1）')


def probe(ref):
    p = ref.path
    result = dict(duration_s=0, message_count=0, topics=[], cameras={}, source_eligible=None,
                  source_reason='', source_instruction=ref.instruction, expected_cameras=[], error='')
    try:
        with p.open('rb') as f:
            summary = make_reader(f).get_summary()
            if not summary or not summary.statistics:
                raise ValueError('MCAP 缺少完整索引 / 录制可能未完成')
            s = summary.statistics
            duration = (s.message_end_time - s.message_start_time) / 1e9
            result.update(duration_s=duration, message_count=s.message_count,
                          start_ns=s.message_start_time, end_ns=s.message_end_time)
            for cid, c in summary.channels.items():
                count = s.channel_message_counts.get(cid, 0)
                schema = summary.schemas.get(c.schema_id)
                result['topics'].append(dict(topic=c.topic, count=count, rate_hz=count/max(duration,1e-9)))
                if schema and schema.name == 'sensor_msgs/msg/CompressedImage':
                    name = c.topic.split('/camera/')[-1].split('/')[0]
                    result['cameras'][name] = dict(topic=c.topic, count=count, rate_hz=count/max(duration,1e-9))
        for parent in [p.parent, p.parent.parent]:
            side = parent / 'episode_meta.json'
            if side.exists():
                side = json.loads(side.read_text())
                m = side.get('metas', {})
                result.update(source_eligible=m.get('training_eligible'),
                              source_reason=m.get('training_ineligibility_reason', ''),
                              source_instruction=side.get('instruction', ''),
                              head_camera_mode=m.get('head_camera_mode', 'unknown'),
                              target_pipeline=m.get('target_training_pipeline', ''))
                if m.get('wrist_cameras'):
                    result['expected_cameras']=['head','left_wrist','right_wrist']
                break
    except Exception as e:
        result['error'] = str(e)
    return result


def assess(ref, info, cache, rules):
    issues = []
    def add(code, level, label, detail):
        issues.append(dict(code=code, level=level, label=label, detail=detail))
    if info.get('error'):
        add('unreadable', 'error', '文件不可读取', info['error'])
    if ref.outcome is False:
        add('operator_failure', 'review', '采集端标记失败', ref.failure_reason or '请复核是否有可用片段')
    elif ref.outcome is None:
        add('outcome_unknown', 'review', '任务结果未知', '没有找到采集端成功 / 失败标记')
    if info.get('source_eligible') is False:
        add('source_ineligible', 'warning', '原转换流程尚未支持训练', info['source_reason'])
    instruction = info.get('source_instruction', '')
    if not instruction or 'validate v1.4' in instruction.lower():
        add('task_instruction', 'review', '需要补充具体任务指令', '当前指令为空或仅描述系统验证；请描述动作目标，如“拿起桌上的物体并放入容器”。')
    if info['duration_s'] < rules['min_duration_s']:
        add('short', 'warning', '记录时间过短', f"{info['duration_s']:.2f} 秒")
    topics = {t['topic']:t for t in info['topics']}
    for topic, label, minimum in [(T_LOWSTATE,'身体状态',rules['state_min_hz']), (T_ACTION,'身体动作',rules['action_min_hz'])]:
        t = topics.get(topic)
        if not t or not t['count']:
            add('missing_'+label, 'error', label+'缺失', topic)
        elif t['rate_hz'] < minimum:
            add('rate_'+label, 'warning', label+'频率偏低', f"{t['rate_hz']:.1f} Hz < {minimum:g} Hz")
    if not info['cameras']:
        add('camera_missing', 'error', '没有图像数据', '未发现 CompressedImage 通道')
    for name in info.get('expected_cameras',[]):
        if name not in info['cameras']:
            add('camera_missing_'+name,'error','预期相机通道缺失',name+' 在采集配置中启用，但未找到消息')
    for name,c in info['cameras'].items():
        if c['rate_hz'] < rules['camera_min_hz']:
            add('fps_'+name,'warning',name+' 帧率偏低',f"{c['rate_hz']:.1f} Hz")
    meta_path = cache / 'meta.json'
    metrics = {}
    grading = None
    rating = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for t in meta['topics']:
            limit = rules['camera_gap_ms'] if t['topic'] in [c['topic'] for c in info['cameras'].values()] else rules['action_gap_ms'] if t['topic']==T_ACTION else None
            if limit and t['max_gap_ms'] > limit:
                add('gap_'+t['topic'], 'warning', '采样间隔过大', f"{t['topic']} · {t['max_gap_ms']:.1f} ms > {limit:g} ms")
        if not meta['episode']['instruction_locked']:
            add('instruction_changes','review','记录期间任务指令发生变化','建议按任务变化拆分片段')
        block = np.memmap(cache/'series.f32',dtype='<f4',mode='r')
        def get(k):
            c=meta['channels'].get(k)
            return block[c['offset']:c['offset']+c['series']*c['samples']].reshape(c['series'],c['samples']) if c else None
        q,a=get('state.q'),get('action.q')
        if q is not None and a is not None:
            diff=np.abs(q-a); vals=diff[np.isfinite(diff)]
            if vals.size:
                metrics['tracking_p99_rad']=round(float(np.percentile(vals,99)),4)
                if metrics['tracking_p99_rad']>rules['tracking_p99_rad']:
                    add('tracking','warning','动作跟踪偏差较大',f"绝对误差 P99 = {metrics['tracking_p99_rad']:.3f} rad；需结合动作速度复核")
        temp=get('state.temp')
        if temp is not None and np.isfinite(temp).any():
            metrics['max_temperature_c']=float(np.nanmax(temp))
            if metrics['max_temperature_c']>rules['temperature_c']:
                add('temperature','warning','电机温度偏高',f"{metrics['max_temperature_c']:.1f} °C")
        with np.load(cache/'native.npz', allow_pickle=False) as native:
            for key in ['state.q','action.q','hand.left.state_q','hand.right.state_q','hand.left.cmd_q','hand.right.cmd_q']:
                if key+'.v' not in native:
                    add('missing_'+key,'error','状态 / 动作通道缺失',key)
                elif not np.isfinite(native[key+'.v']).all():
                    add('nonfinite_'+key,'error','存在无效数值',key+' 包含 NaN / Inf')
            grading = grade_data(info, issues, native=native, meta=meta)
            rating = technical_rating(grading, native, meta, rules)
        stage='complete'
    else:
        stage='summary'
    grading = grading or grade_data(info, issues)
    rating = rating or technical_rating(grading)
    grade = grading['grade'] or ''
    context_codes={'source_ineligible','task_instruction','operator_failure','outcome_unknown','instruction_changes'}
    technical=[i for i in issues if i['code'] not in context_codes]
    technical_status='error' if any(i['level']=='error' for i in technical) else 'pending' if stage!='complete' else 'attention' if technical else 'pass'
    return dict(stage=stage,suggested_grade=grade,grading=grading,issues=issues,metrics=metrics,
                technical=dict(status=technical_status,count=len(technical)),
                score=rating['score'], rating=rating,
                rules=rules)
