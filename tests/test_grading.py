import numpy as np
import pytest

from grading import grade_data, coverage_grade, WIDTHS, POLICY


@pytest.fixture
def complete_data():
    t0 = 1_789_000_000_123_456_789
    t = t0 + np.arange(0, 10_000_000_000, 20_000_000, dtype=np.int64)
    native = {}
    for key, width in WIDTHS.items():
        native[key+'.t'] = t.copy()
        native[key+'.v'] = np.zeros((len(t), width), dtype=np.float32)
    cameras = {name:dict(count=300, topic='/camera/'+name) for name in POLICY['cameras']}
    for name in cameras:
        native['camera.'+name+'.t'] = t0 + np.rint(np.arange(300)*1e9/30).astype(np.int64)
    info = dict(cameras=cameras)
    meta = dict(episode=dict(start_ns=t0, end_ns=t0+10_000_000_000), cameras=cameras)
    return info, native, meta


def grade(data, issues=None):
    info, native, meta = data
    return grade_data(info, issues or [], native=native, meta=meta)


def test_complete_data_is_a_and_annotation_status_does_not_lower_it(complete_data):
    issues = [dict(code=code, label='审核待办', detail='待人工确认', level='review')
              for code in ['source_ineligible','task_instruction','operator_failure','outcome_unknown','instruction_changes']]
    a = grade(complete_data)
    assert a['grade'] == 'A' and a['metrics']['coverage_fraction'] == 1
    assert grade(complete_data, issues) == a


def test_unanalysed_is_pending_not_poor_quality(complete_data):
    result = grade_data(complete_data[0], [])
    assert result['grade'] is None and result['state'] == 'pending'
    assert result['metrics'] == {}


@pytest.mark.parametrize('ratio,expected',[(1,'A'),(.99,'A'),(.9899,'B'),(.95,'B'),(.9499,'C'),(.001,'C'),(0,'D')])
def test_exact_coverage_boundaries(ratio, expected):
    assert coverage_grade(ratio) == expected


def test_missing_wrist_is_d_for_current_three_camera_workflow(complete_data):
    info, _, _ = complete_data
    del info['cameras']['left_wrist']
    result = grade_data(info, [])
    assert result['grade'] == 'D' and 'left_wrist' in result['summary']


def test_small_coverage_loss_is_b_but_long_loss_is_c(complete_data):
    info, native, meta = complete_data
    camera = 'camera.head.t'
    native[camera] = native[camera][9:]  # 3% of aligned instants unavailable.
    result = grade(complete_data)
    assert result['grade'] == 'B' and result['metrics']['coverage_fraction'] == .97
    native[camera] = native[camera][51:]
    result = grade(complete_data)
    assert result['grade'] == 'C' and result['metrics']['coverage_fraction'] == .8


def test_local_nan_can_be_salvaged_but_all_invalid_is_d(complete_data):
    native = complete_data[1]
    native['state.q.v'][100, 0] = np.nan
    result = grade(complete_data)
    assert result['grade'] == 'C'
    assert result['metrics']['aligned_samples'] > 0
    native['state.q.v'][:, 0] = np.nan
    assert grade(complete_data)['grade'] == 'D'


def test_no_common_time_coverage_is_d(complete_data):
    complete_data[1]['camera.head.t'] += 20_000_000_000
    result = grade(complete_data)
    assert result['grade'] == 'D' and result['metrics']['aligned_samples'] == 0


def test_wrong_joint_width_is_d(complete_data):
    complete_data[1]['action.q.v'] = complete_data[1]['action.q.v'][:, :28]
    assert grade(complete_data)['grade'] == 'D'


def test_unsorted_camera_does_not_report_misleading_coverage(complete_data):
    complete_data[1]['camera.head.t'][[10, 11]] = complete_data[1]['camera.head.t'][[11, 10]]
    result = grade(complete_data)
    assert result['grade'] == 'C'
    assert 'coverage_fraction' not in result['metrics']


def test_motion_reminder_is_b_not_a_task_failure(complete_data):
    issues = [dict(code='tracking', level='warning', label='跟踪偏差', detail='P99 超过提醒线')]
    result = grade(complete_data, issues)
    assert result['grade'] == 'B' and result['metrics']['coverage_fraction'] == 1
    assert result['reasons'][0]['key'] == 'motion'


def test_longest_valid_interval_never_bridges_a_gap(complete_data):
    native = complete_data[1]
    native['state.q.v'][200:250] = np.nan
    result = grade(complete_data)
    assert result['metrics']['longest_valid_s'] <= 5.01
    assert all(r['end'] <= 4+1e-8 or r['start'] >= 5 for r in result['metrics']['valid_intervals'])
