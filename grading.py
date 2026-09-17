"""Versioned, evidence-based data grades for the G1 / BrainCo RGB workflow.

Task annotations and operator outcomes never lower a measured data grade.
The grade describes measured numeric/timing quality, not visual task success.
"""
import numpy as np

from exporter import plan_clip

VERSION = 1
LABELS = dict(A='优质', B='良好', C='需修整', D='当前不可用')
POLICY = dict(version=VERSION, profile='G1 · 29 关节 + 双手 12 电机 + 三路 RGB',
              fps=30, coverage_a=.99, coverage_b=.95,
              cameras=['head', 'left_wrist', 'right_wrist'],
              scope='数值与时序质量；图像内容、任务成效和训练适配另行审核')
WIDTHS = {'state.q':29, 'action.q':29, 'hand.left.state_q':6,
          'hand.right.state_q':6, 'hand.left.cmd_q':6, 'hand.right.cmd_q':6}


def coverage_grade(fraction):
    if fraction >= POLICY['coverage_a']: return 'A'
    if fraction >= POLICY['coverage_b']: return 'B'
    return 'C' if fraction > 0 else 'D'


def grade_data(info, issues, *, native=None, meta=None):
    checks = []
    metrics = {}

    def check(key, label, grade, detail, action=''):
        checks.append(dict(key=key, label=label, grade=grade, detail=detail, action=action))

    def result(pending=False):
        grade = None if pending else max(c['grade'] for c in checks)
        reasons = [c for c in checks if c['grade'] != 'A']
        # Report the decisive restriction before lesser reminders.
        reasons.sort(key=lambda c:c['grade'], reverse=True)
        return dict(version=VERSION, grade=grade, label=LABELS.get(grade, '待评估'),
                    state='pending' if pending else 'evaluated', checks=checks,
                    reasons=reasons, metrics=metrics, policy=POLICY,
                    summary=('完成深度质检后评估数据等级' if pending else
                             reasons[0]['detail'] if reasons else '必需通道齐全，已测数值与时序指标达到 A 级要求'))

    missing = [name for name in POLICY['cameras']
               if not info.get('cameras', {}).get(name, {}).get('count')]
    fatal = [i for i in issues if i['code'] == 'unreadable' or
             i['code'].startswith('missing_') or i['code'].startswith('camera_missing')]
    if missing or fatal:
        details = [i['label']+'：'+i['detail'] for i in fatal]
        if missing: details.append('缺少相机消息：'+', '.join(missing))
        check('integrity', '必需通道', 'D', '；'.join(details), '补齐或重新采集缺失通道；当前 G1 工作流无法使用整段数据')
        return result()
    if native is None or meta is None:
        return result(pending=True)

    invalid, unordered, malformed = [], [], []
    for key, width in WIDTHS.items():
        if key+'.t' not in native or key+'.v' not in native:
            malformed.append(key); continue
        t, v = native[key+'.t'], native[key+'.v']
        if t.ndim != 1 or v.ndim != 2 or v.shape != (len(t), width) or not len(t):
            malformed.append(key); continue
        finite_rows = np.isfinite(v).all(axis=1)
        if not finite_rows.any():
            malformed.append(key+'（没有完整有效数值）'); continue
        if not finite_rows.all(): invalid.append(key)
        if not np.issubdtype(t.dtype, np.integer) or np.any(t[1:] < t[:-1]): unordered.append(key)
    for name in POLICY['cameras']:
        key = 'camera.'+name+'.t'
        if name not in meta.get('cameras', {}) or key not in native or native[key].ndim != 1 or not len(native[key]):
            malformed.append(key); continue
        t = native[key]
        if not np.issubdtype(t.dtype, np.integer) or np.any(t[1:] < t[:-1]): unordered.append(key)
    if malformed:
        check('integrity', '必需通道', 'D', '通道为空、缺失或维度不匹配：'+', '.join(malformed),
              '核对原始采集与解析结果；当前 41 维状态 / 动作要求未满足')
        return result()
    check('integrity', '必需通道', 'A', '身体 29 关节、双手各 6 电机的状态 / 动作与三路相机齐全')
    metrics.update(nonfinite_channels=invalid, unordered_channels=unordered)
    check('numeric', '数值完整性', 'C' if invalid else 'A',
          '存在 NaN / Inf：'+', '.join(invalid) if invalid else '六组状态 / 动作通道均为有限数值',
          '定位无效时段并裁剪，处理后重新评估' if invalid else '')
    if unordered:
        check('timestamps', '时间戳顺序', 'C', '时间戳格式或顺序异常：'+', '.join(unordered),
              '修正时间索引后重新评估；当前不提供可能失真的覆盖率')
        return result()

    try:
        duration = (meta['episode']['end_ns']-meta['episode']['start_ns'])/1e9
        plan, _ = plan_clip(native, meta, 0, duration, POLICY['fps'])
        valid = plan['valid']; n = len(valid); count = int(valid.sum())
        fraction = count/n if n else 0
        # Consecutive valid samples only; never bridge invalid intervals.
        edges = np.diff(np.r_[False, valid, False].astype(np.int8))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        runs = []
        for a, b in zip(starts, ends):
            start = float(plan['timestamp_s'][a])
            end = min(duration, float(plan['timestamp_s'][b-1])+1/POLICY['fps'])
            runs.append(dict(start=start, end=end, duration_s=end-start))
        runs.sort(key=lambda r:r['duration_s'], reverse=True)
        metrics.update(samples=n, aligned_samples=count, coverage_fraction=fraction,
                       longest_valid_s=runs[0]['duration_s'] if runs else 0,
                       valid_intervals=runs[:5],
                       invalid_by_channel={k[6:]:int((~v).sum()) for k,v in plan.items()
                                           if k.startswith('valid.') and (k[6:] in WIDTHS or k.startswith('valid.camera.'))})
        band = coverage_grade(fraction)
        check('coverage', '有效数据覆盖', band,
              f'{count} / {n} 个时刻具备完整数值与图像时间覆盖（{fraction:.2%}）',
              '' if band == 'A' else '裁剪无效区间后复核保留片段' if count else '没有可对齐样本，需要补录或修正源数据')
    except (ValueError, KeyError) as e:
        check('coverage', '有效数据覆盖', 'D', str(e), '核对原始数据与时间索引后重新解析')

    continuity = [i for i in issues if i['code'] == 'short' or i['code'].startswith(('rate_', 'fps_', 'gap_'))]
    check('continuity', '采样连续性', 'B' if continuity else 'A',
          '；'.join(i['detail'] for i in continuity) if continuity else '时长、关键流频率和间隔均在当前规则范围内',
          '回放间隔异常处，确认是否需要裁剪或补录' if continuity else '')
    motion = [i for i in issues if i['code'] in ('tracking', 'temperature')]
    check('motion', '运动与温度提醒', 'B' if motion else 'A',
          '；'.join(i['detail'] for i in motion) if motion else '跟踪误差与电机温度未触发当前提醒线',
          '结合动作速度与接触过程复核；提醒不等同任务失败' if motion else '')
    return result()
