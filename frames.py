"""Exact camera frame indexes and bounded batch reads for local playback."""
from bisect import bisect_left
from collections import OrderedDict, defaultdict
import json
import struct
import threading
import numpy as np
from mcap.reader import make_reader
from server import _jpeg_payload
from storage import Conflict


class FrameStore:
    def __init__(self, library, max_bytes=64*1024*1024):
        self.library=library
        self.indexes={}
        self.index_lock=threading.Lock()
        self.cache=OrderedDict()
        self.cache_lock=threading.Lock()
        self.cache_bytes=0
        self.max_bytes=max_bytes

    def index(self, ep):
        lib=self.library;ref=lib.ref(ep);fp=lib.fingerprint(ref)
        token=f"{fp['size']}:{fp['mtime_ns']}"
        with self.index_lock:
            hit=self.indexes.get(ep)
            if hit and hit['token']==token:return hit
            info=lib.info(ref)
            if info.get('error'):raise ValueError(info['error'])
            cameras={name:dict(topic=cam['topic'],times=[]) for name,cam in info['cameras'].items()}
            cache=lib.cache_dir(ref)
            if lib.state(ep)['state']=='ready':
                with np.load(cache/'native.npz',allow_pickle=False) as native:
                    for name,cam in cameras.items():cam['times']=native['camera.'+name+'.t'].tolist()
            else:
                by_topic={c['topic']:c for c in cameras.values()}
                if by_topic:
                    with ref.path.open('rb') as f:
                        for _,channel,message in make_reader(f).iter_messages(topics=list(by_topic)):
                            by_topic[channel.topic]['times'].append(int(message.log_time))
                for cam in cameras.values():cam['times'].sort()
            if lib.fingerprint(ref)!=fp:raise Conflict('源文件已变化，请重新载入记录')
            result=dict(token=token,start_ns=info['start_ns'],duration_s=info['duration_s'],cameras=cameras)
            self.indexes[ep]=result
            return result

    def public_index(self,ep):
        d=self.index(ep)
        return dict(token=d['token'],duration_s=d['duration_s'],cameras={name:dict(count=len(c['times']),times_s=[(t-d['start_ns'])/1e9 for t in c['times']]) for name,c in d['cameras'].items()})

    def batch(self,ep,cam,start,count,token=None):
        index=self.index(ep)
        if token is not None and token!=index['token']:raise Conflict('源文件已变化，请重新载入记录')
        if cam not in index['cameras']:raise KeyError('相机不存在')
        camera=index['cameras'][cam];times=camera['times']
        if not 1<=count<=24 or not 0<=start<len(times):raise ValueError('帧范围无效')
        ids=list(range(start,min(start+count,len(times))))
        payloads={};missing=[]
        with self.cache_lock:
            for i in ids:
                key=(ep,cam,index['token'],i)
                if key in self.cache:
                    payloads[i]=self.cache[key];self.cache.move_to_end(key)
                else:missing.append(i)
        ref=self.library.ref(ep)
        if missing:
            wanted=set(missing);seen=defaultdict(int)
            with ref.path.open('rb') as f:
                for _,_,msg in make_reader(f).iter_messages(topics=[camera['topic']],start_time=times[missing[0]],end_time=times[missing[-1]]+1):
                    # Preserve distinct messages with identical log timestamps.
                    i=bisect_left(times,msg.log_time)+seen[msg.log_time];seen[msg.log_time]+=1
                    if i in wanted:payloads[i]=_jpeg_payload(msg.data) or b''
            for i in missing:payloads.setdefault(i,b'')
            if self.library.fingerprint(ref)!={'size':int(index['token'].split(':')[0]),'mtime_ns':int(index['token'].split(':')[1])}:
                raise Conflict('源文件已变化，请重新载入记录')
            with self.cache_lock:
                for i in missing:
                    key=(ep,cam,index['token'],i)
                    old=self.cache.pop(key,None)
                    if old is not None:self.cache_bytes-=len(old)
                    self.cache[key]=payloads[i];self.cache_bytes+=len(payloads[i])
                while self.cache_bytes>self.max_bytes and self.cache:
                    _,old=self.cache.popitem(last=False);self.cache_bytes-=len(old)
        header=dict(token=index['token'],frames=[dict(index=i,time_s=(times[i]-index['start_ns'])/1e9,size=len(payloads[i])) for i in ids])
        return header,[payloads[i] for i in ids]

    def encoded_batch(self,*args,**kwargs):
        header,images=self.batch(*args,**kwargs)
        raw=json.dumps(header,separators=(',',':')).encode()
        return struct.pack('<I',len(raw))+raw+b''.join(images)
