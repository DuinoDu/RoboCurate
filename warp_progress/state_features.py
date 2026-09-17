"""Causally align measured G1 joint/hand states; no action commands or outcome flags."""
from pathlib import Path
import numpy as np

STATE_KEYS = ('state.q', 'hand.left.state_q', 'hand.right.state_q')


def native_fingerprint(path):
    if path is None or not Path(path).is_file():return None
    stat=Path(path).stat()
    return dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def state_contract(meta):
    channels = []
    for key in STATE_KEYS:
        row = meta.get('channels', {}).get(key)
        if row is None:
            raise ValueError('缺少训练所需的身体或手部状态描述')
        channels.append(dict(key=key, names=row['names'], unit=row['unit']))
    if [len(c['names']) for c in channels] != [29,6,6]:
        raise ValueError('状态模型要求 29 个身体关节和双手各 6 个通道')
    return dict(version=1, channels=channels, dimension=41, alignment='latest_at_or_before',max_age_ns=100_000_000)


def sample_state(native, timestamps, contract):
    times = np.asarray(timestamps)
    if times.ndim != 1 or times.dtype.kind not in 'iu' or np.any(times[1:]<times[:-1]):
        raise ValueError('状态采样需要有序的整数时间戳')
    if (tuple(c['key'] for c in contract['channels']) != STATE_KEYS or
        [len(c['names']) for c in contract['channels']] != [29,6,6]):
        raise ValueError('状态特征约定维数或通道顺序错误')
    data = np.full((len(times),41),np.nan,np.float32)
    valid = np.ones(len(times),bool)
    source_times = np.full((len(times),3),-1,np.int64)
    offset=0
    for j,ch in enumerate(contract['channels']):
        key=ch['key'];width=len(ch['names'])
        if key+'.t' not in native or key+'.v' not in native:
            valid[:]=False;offset+=width;continue
        t=np.asarray(native[key+'.t']);v=np.asarray(native[key+'.v'])
        if t.dtype.kind not in 'iu' or t.ndim!=1 or v.shape!=(len(t),width) or np.any(t[1:]<t[:-1]):
            raise ValueError('原始状态的形状或时间顺序不匹配')
        if len(t):
            ix=np.searchsorted(t,times,side='right')-1
            safe=np.maximum(ix,0)
            ok=(ix>=0)&(times-t[safe]<=contract['max_age_ns'])&np.isfinite(v[safe]).all(1)
            data[ok,offset:offset+width]=v[safe[ok]]
            source_times[ok,j]=t[safe[ok]]
            valid &= ok
        else:valid[:]=False
        offset+=width
    if offset!=41:
        raise ValueError('状态特征约定维数错误')
    data[~valid]=np.nan
    return dict(state=data,state_valid=valid,state_source_timestamp_ns=source_times)


def read_state(path,timestamps,meta,contract):
    missing=False
    for trained in contract['channels']:
        current=meta.get('channels',{}).get(trained['key'])
        if current is None:
            missing=True
        elif current.get('names')!=trained['names'] or current.get('unit')!=trained['unit']:
            raise ValueError('当前关节状态名称或单位与训练约定不匹配')
    if missing or path is None or not Path(path).is_file():
        return dict(state=np.full((len(timestamps),41),np.nan,np.float32),
                    state_valid=np.zeros(len(timestamps),bool),state_source_timestamp_ns=np.full((len(timestamps),3),-1,np.int64))
    with np.load(path,allow_pickle=False) as native:
        return sample_state(native,timestamps,contract)
