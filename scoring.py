"""Explainable G1 technical rating on a ten-point scale.

The weights are an explicit pilot policy. Missing dimensions never receive a
clean score. Mandatory grade gates remain separate and cannot be averaged away.
"""
import numpy as np
from exporter import plan_clip
from grading import WIDTHS, POLICY

VERSION = 3
MAX_SCORE = 10
WEIGHTS = dict(coverage=.40, timing=.15, numeric=.15, tracking=.30)


def technical_rating(grading, native=None, meta=None, rules=None):
    out = dict(version=VERSION, method_version=2, max_score=MAX_SCORE, score=None, state='pending', dimensions=[],
               weights=WEIGHTS, grade_gate=grading['grade'],
               scope='技术评分 · 试行；未评估图像内容、任务成功或训练收益')
    if grading['grade'] == 'D':
        out.update(state='blocked', reason=grading['summary'])
        return out
    if native is None or meta is None or rules is None:
        return out
    def add(key, label, score, detail, **evidence):
        out['dimensions'].append(dict(key=key, label=label, score=round(float(score), 3),
                                      weight=WEIGHTS[key], detail=detail, evidence=evidence))
    try:
        if grading['metrics'].get('unordered_channels'):
            raise ValueError('时间戳异常，评分待复核')
        duration = (meta['episode']['end_ns']-meta['episode']['start_ns'])/1e9
        plan, _ = plan_clip(native, meta, 0, duration, 30)
        coverage = float(plan['valid'].mean())
        add('coverage', '同步覆盖', coverage*MAX_SCORE,
            '30 Hz 网格上三路相机与 41 维状态、动作同时有效的比例；不含 JPEG 解码检查。',
            valid=int(plan['valid'].sum()), total=len(plan['valid']))
        finite = [float(np.isfinite(native[k+'.v']).all(axis=1).mean()) for k in WIDTHS]
        add('numeric', '数值完整', np.mean(finite)*MAX_SCORE,
            '六组原始状态和动作通道中完整有限数值的行占比，各组等权。', fractions=dict(zip(WIDTHS, finite)))
        timing = []
        streams = [('state.q', rules['state_min_hz'], .1), ('action.q', rules['action_min_hz'], rules['action_gap_ms']/1000)]
        streams += [('camera.'+name, rules['camera_min_hz'], rules['camera_gap_ms']/1000) for name in POLICY['cameras']]
        for key, minimum_hz, gap in streams:
            t = native[key+'.t']; dt = np.diff(t.astype(np.int64))/1e9
            if not len(dt): raise ValueError('时间样本不足')
            # Count compliance and wall-time support together: one long gap
            # cannot hide among thousands of high-rate samples.
            supported = float(dt[(dt > 0) & (dt <= gap)].sum())/duration
            ordered = float((dt > 0).mean())
            rate = min(1., len(t)/duration/minimum_hz)
            value = min(1., supported, ordered, rate)
            timing.append(dict(channel=key, score=value*MAX_SCORE, min_hz=minimum_hz, max_gap_s=gap))
        add('timing', '采样连续', min(t['score'] for t in timing),
            '身体状态、动作和三路相机中最弱一路：频率达标率、正常间隔覆盖时长、正向时间间隔比例取最小值。', channels=timing)
        groups = [('左腿', 'state.q', 'action.q', 0, 6, rules['tracking_p99_rad']),
                  ('右腿', 'state.q', 'action.q', 6, 12, rules['tracking_p99_rad']),
                  ('腰部', 'state.q', 'action.q', 12, 15, rules['tracking_p99_rad']),
                  ('左臂', 'state.q', 'action.q', 15, 22, rules['tracking_p99_rad']),
                  ('右臂', 'state.q', 'action.q', 22, 29, rules['tracking_p99_rad']),
                  ('左手', 'hand.left.state_q', 'hand.left.cmd_q', 0, 6, rules['hand_error']),
                  ('右手', 'hand.right.state_q', 'hand.right.cmd_q', 0, 6, rules['hand_error'])]
        tracking = []
        for label, state, action, start, stop, threshold in groups:
            valid = plan['valid.'+state] & plan['valid.'+action]
            if not valid.any(): raise ValueError('缺少可比较的状态和动作')
            delta = np.abs(plan[state][valid, start:stop]-plan[action][valid, start:stop])
            ratio = float((delta <= threshold).all(axis=1).mean())
            tracking.append(dict(label=label, fraction=ratio, threshold=threshold,
                                 compared_samples=int(valid.sum()), p99=float(np.percentile(delta,99))))
        add('tracking', '跟踪一致', np.mean([g['fraction'] for g in tracking])*MAX_SCORE,
            '七个部位各关节均未超过提醒线的时刻占比，部位等权。动作取过去最近指令，未补偿控制延迟；接触和快速运动需人工判断。', groups=tracking)
        out.update(state='measured', score=round(sum(d['score']*d['weight'] for d in out['dimensions']), 2))
    except (KeyError, ValueError, IndexError, TypeError) as e:
        out.update(state='unavailable', reason=str(e))
    return out
