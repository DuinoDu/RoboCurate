"""Native-timestamp wrist origins in the URDF pelvis frame, without odometry.

This is visualization data, not an export or a task-success metric. Split on
invalid rows BEFORE decimation, and never repair timestamp order silently.
"""
import numpy as np

VERSION = 2
MAX_GAP_S = .1


def rotation(axis, angles):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError('URDF joint axis must be finite and nonzero')
    x, y, z = axis / norm
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    outer = np.outer(axis / norm, axis / norm)
    angles = np.asarray(angles)
    return (np.cos(angles)[..., None, None] * (np.eye(3)-outer)
            + outer + np.sin(angles)[..., None, None]*cross)


def chain_for(desc, link):
    parents = {j['child']: j for j in desc['joints']}
    chain, visited = [], set()
    while link != desc['root']:
        if link in visited or link not in parents:
            raise ValueError('URDF link is not connected to the root')
        visited.add(link)
        joint = parents[link]
        chain.append(joint)
        link = joint['parent']
    return chain[::-1]


def wrist_positions(desc, names, values, link):
    """Vectorized FK. Only ancestors of the requested link affect validity."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError('Joint order/width does not match the recording')
    columns = {name + '_joint': i for i, name in enumerate(names)}
    n = len(values)
    matrix = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    position = np.zeros((n, 3))
    valid = np.ones(n, dtype=bool)
    for joint in chain_for(desc, link):
        position += np.einsum('nij,j->ni', matrix, joint['xyz'])
        r, p, y = joint['rpy']
        origin = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
        matrix = matrix @ origin
        if joint['type'] == 'fixed':
            continue
        if joint['name'] not in columns:
            raise ValueError('Unrecorded ancestor joint: ' + joint['name'])
        q = values[:, columns[joint['name']]]
        valid &= np.isfinite(q)
        q = np.where(np.isfinite(q), q, 0)
        if joint['type'] in ('revolute', 'continuous'):
            matrix = matrix @ rotation(joint['axis'], q)
        elif joint['type'] == 'prismatic':
            axis = np.asarray(joint['axis'], dtype=float)
            axis /= np.linalg.norm(axis)
            position += np.einsum('nij,nj->ni', matrix, q[:, None]*axis)
        else:
            raise ValueError('Unsupported joint type: '+joint['type'])
    position[~valid] = np.nan
    return position, valid


def segmented_track(times, positions, valid, t0, *, display_hz=30):
    times = np.asarray(times)
    if times.ndim != 1 or times.dtype.kind not in 'iu' or len(times) != len(valid):
        raise ValueError('Expected matching native integer timestamps')
    if np.any(np.diff(times.astype(np.int64)) < 0):
        raise ValueError('时间戳倒序，轨迹未绘制；请复核数据时间索引')
    seconds = (times.astype(np.int64)-int(t0))/1e9
    dt = np.diff(seconds)
    # A duplicate timestamp starts a new segment; no zero-duration path step.
    joins = valid[1:] & valid[:-1] & (dt > 0) & (dt <= MAX_GAP_S)
    segments, start = [], None
    travel, moving_s, observed_s = 0., 0., 0.
    if joins.any():
        length = np.linalg.norm(np.diff(positions, axis=0)[joins], axis=1)
        step = dt[joins]
        travel = float(length.sum())
        observed_s = float(step.sum())
        moving_s = float(step[length/step >= .01].sum())
    for i in range(len(times)+1):
        if start is not None and (i == len(times) or not joins[i-1]):
            candidates = np.arange(start, i)
            keep = [start]
            for j in candidates[1:]:
                if seconds[j]-seconds[keep[-1]] >= 1/display_hz:
                    keep.append(int(j))
            if keep[-1] != i-1: keep.append(i-1)
            segments.append(dict(times_s=seconds[keep].tolist(), positions=positions[keep].round(6).tolist()))
            start = None
        if i < len(times) and valid[i] and start is None: start = i
    return dict(segments=segments, native_samples=len(times),
                points=sum(len(s['times_s']) for s in segments),
                invalid_samples=int((~valid).sum()), gap_count=int((dt > MAX_GAP_S).sum()),
                motion=dict(travel_m=travel, observed_s=observed_s,
                            active_fraction=moving_s/observed_s if observed_s else None,
                            active_threshold_m_s=.01, advisory_only=True))


def build_trajectories(native, meta, desc):
    tracks, unavailable = [], []
    for source in ['state.q', 'action.q']:
        for part, side in [(p, s) for p in ['wrist', 'foot'] for s in ['left', 'right']]:
            link = side+('_wrist_yaw_link' if part=='wrist' else '_ankle_roll_link')
            try:
                if source+'.t' not in native or source+'.v' not in native:
                    raise ValueError('缺少原始关节通道')
                positions, valid = wrist_positions(desc, meta['joints']['names'], native[source+'.v'], link)
                track = segmented_track(native[source+'.t'], positions, valid, meta['timeline']['t0_ns'])
                tracks.append(dict(source=source, side=side, part=part, link=link, **track))
            except ValueError as e:
                unavailable.append(dict(source=source, side=side, part=part, reason=str(e)))
    poses = dict(source='state.q', names=meta['joints']['names'], segments=[], reason='')
    if 'state.q.t' in native and 'state.q.v' in native:
        try:
            q = native['state.q.v']
            if q.ndim != 2 or q.shape[1] != len(poses['names']):
                raise ValueError('关节维度不匹配')
            sampled = segmented_track(native['state.q.t'], q, np.isfinite(q).all(axis=1), meta['timeline']['t0_ns'])
            poses['segments'] = [dict(times_s=s['times_s'], q=s['positions']) for s in sampled['segments']]
        except ValueError as e:
            poses['reason'] = str(e)
    else: poses['reason'] = '缺少原始实测关节数据'
    return dict(version=VERSION, frame='pelvis', imu_applied=False, max_gap_s=MAX_GAP_S, poses=poses,
                display_hz=30, tracks=tracks, unavailable=unavailable,
                note='骨盆坐标系中的腕部与踝部关节原点；未标定 TCP 或接触点，不含基座位移或 IMU 旋转。原始采样计算，轨迹仅作显示抽稀。')
