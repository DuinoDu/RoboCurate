"""Local WARP job orchestration, stale-cache checks and export alignment.

This module deliberately depends only on the normal application environment.
Checkpoint loading, DINO extraction and scoring run in an isolated subprocess.
"""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid
import numpy as np
from warp_progress.core import VERSION, UPSTREAM, json_signature, align_velocity, chunk_weights
from warp_progress.features import atomic_json
from warp_progress.state_features import STATE_KEYS, native_fingerprint

APP = Path(__file__).resolve().parent


def validate_policy(value):
    if value is None or value == {}:
        return None
    if not isinstance(value, dict) or set(value) - {'model_id','mode','threshold','horizon'}:
        raise ValueError('进展权重配置无效')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', str(value.get('model_id', ''))):
        raise ValueError('请选择进展模型')
    horizon = value.get('horizon', 30)
    threshold = float(value.get('threshold', 1.))
    mode = value.get('mode', 'continuous')
    if type(horizon) is not int or not 1 <= horizon <= 10000 or not math.isfinite(threshold) or not 0 <= threshold <= 100:
        raise ValueError('动作块长度或速度阈值无效')
    if mode not in ('continuous', 'binary'):
        raise ValueError('权重模式无效')
    return dict(model_id=value['model_id'], mode=mode, threshold=threshold, horizon=horizon)


def stage_evidence(result, path):
    """Read display evidence from the existing score; never rerun or alter it."""
    n=len(result.get('times_s',[]));classes=len(result.get('stage_names',[]))
    evidence=dict(available=False,source='saved_model_output',cameras=['head','right_wrist'])
    with np.load(path,allow_pickle=False) as arrays:
        if 'probability' not in arrays:return evidence
        p=arrays['probability']
        if p.shape!=(n,classes) or classes<1:raise ValueError('阶段概率缓存与时间轴不匹配')
        valid=np.asarray(result['stage'])>=0
        if (not np.isfinite(p[valid]).all() or np.any(p[valid]<0) or np.any(p[valid]>1) or
            not np.allclose(p[valid].sum(-1),1.,atol=1e-5)):
            raise ValueError('阶段概率缓存无效')
        if not np.allclose(p[valid].max(-1),np.asarray(result['confidence'],float)[valid],atol=2e-6):
            raise ValueError('阶段概率与评分把握度不一致')
        evidence.update(available=True,probability=[[float(v) for v in row] if ok else None for row,ok in zip(p,valid)])
        for key,shape in [('camera_valid',(n,2)),('state_valid',(n,))]:
            if key in arrays:
                a=arrays[key]
                if a.shape!=shape or a.dtype!=np.dtype(bool):raise ValueError('阶段输入掩码与时间轴不匹配')
                evidence[key]=a.tolist()
    return evidence


class ProgressService:
    def __init__(self, library):
        self.library = library
        self.root = library.workspace / 'warp'
        self.model_root = self.root / 'models'
        self.jobs_root = self.root / 'jobs'
        self.results_root = self.root / 'scores'
        for p in (self.model_root, self.jobs_root, self.results_root):
            p.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='warp')
        self.lock = threading.RLock()
        self.active = {}
        self.processes = {}
        self._lease = (self.root/'jobs.lock').open('a')
        try:
            fcntl.flock(self._lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.owns_jobs = True
        except BlockingIOError:
            self.owns_jobs = False
        # A server restart cannot leave an old job displaying 'running'.
        for path in self.jobs_root.glob('*.json'):
            try:
                job = json.loads(path.read_text())
                if self.owns_jobs and job.get('state') in ('queued', 'running'):
                    job.update(state='interrupted', message='服务已重启，请重新提交评分', finished_at=time.time())
                    atomic_json(path, job)
            except (ValueError, OSError):
                pass

    @property
    def python(self):
        return self.library.workspace / 'warp-env' / 'bin' / 'python'

    def _profiles(self):
        path = self.model_root / 'profiles.json'
        if not path.exists():
            return []
        rows = json.loads(path.read_text())['models']
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('模型配置 ID 重复')
        return rows

    def profile(self, model_id):
        row = next((r for r in self._profiles() if r['id'] == model_id), None)
        if row is None:
            raise ValueError('未安装所选进展模型，请运行模型准备脚本')
        row = dict(row)
        for key in ('checkpoint', 'backbone_dir'):
            path = (self.model_root / row[key]).resolve()
            if not path.is_relative_to(self.model_root.resolve()):
                raise ValueError('模型路径必须位于本工作区的 warp/models 目录')
            row[key] = str(path)
        required = [Path(row['checkpoint']), Path(row['backbone_dir'])/'config.json',
                    Path(row['backbone_dir'])/'model.safetensors']
        missing = [p.name for p in required if not p.is_file()]
        if not {'config.json','model.safetensors'} <= set(row.get('backbone_sha256',{})):
            missing.append('backbone-provenance.json')
        if not self.python.is_file():
            missing.append('warp-env/bin/python')
        row['ready'] = not missing
        row['missing'] = missing
        # Include artifact fingerprints and the complete preprocessing contract.
        row['artifact_fingerprints'] = {str(p): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                                        for p in required if p.is_file()}
        row['signature'] = json_signature({**row, 'pipeline_version': VERSION, 'upstream': UPSTREAM})
        return row

    def models(self):
        models = []
        for item in self._profiles():
            p = self.profile(item['id'])
            public = {k:p[k] for k in ('id','label','ready','missing','validation_status','domain_note','task_scope','camera','view','crop','signature')}
            public.update(kind=p.get('kind','warp_velocity'), cameras=p.get('cameras',[p['camera']]),
                          supports_weights=p.get('kind') != 'stage_progress',
                          stage_version=p.get('stage_version',1),modalities=p.get('modalities',['vision']))
            models.append(public)
        status_path=self.root/'workflow/status.json'
        workflow=json.loads(status_path.read_text()) if status_path.is_file() else None
        return dict(models=models, environment_ready=self.python.is_file(), offline=True, workflow=workflow)

    def _request(self, ep, model_id):
        ref = self.library.ref(ep)
        if self.library.state(ep)['state'] != 'ready':
            raise ValueError('请先完成记录解析，再计算任务进展')
        profile = self.profile(model_id)
        source = self.library.fingerprint(ref)
        meta = json.loads((self.library.cache_dir(ref)/'meta.json').read_text())
        fusion = profile.get('kind')=='stage_progress' and profile.get('stage_version') in (2,3,4)
        if fusion and not any(c in meta['cameras'] for c in profile.get('cameras',[])):
            raise ValueError('多模态阶段模型至少需要一个训练所用的相机')
        if not fusion and profile['camera'] not in meta['cameras']:
            raise ValueError('记录没有模型所需的相机')
        if not fusion and any(camera not in meta['cameras'] for camera in profile.get('cameras', [])):
            raise ValueError('记录缺少阶段模型所需的相机')
        identity = dict(version=VERSION, source=source, path=str(ref.path.resolve()),
                        start_ns=meta['episode']['start_ns'], end_ns=meta['episode']['end_ns'],
                        model=profile['signature'])
        state_request={}
        if fusion:
            native=self.library.cache_dir(ref)/'native.npz'
            state_request=dict(native_path=str(native.resolve()),native_fingerprint=native_fingerprint(native))
            identity.update(stage_pipeline_version=profile['stage_version'],**state_request,
                state_schema={k:meta.get('channels',{}).get(k) for k in STATE_KEYS},
                camera_schema={k:meta['cameras'].get(k) for k in profile['cameras']})
        signature = json_signature(identity)
        return dict(episode_id=ep, source=str(ref.path.resolve()), source_fingerprint=source,
                    meta=meta, profile=profile, model_signature=profile['signature'], signature=signature,
                    output=str(self.results_root/(signature+'.json')), feature_root=str(self.root/'features'),
                    device='cpu', threads=4, feature_batch_size=8, score_batch_size=16,**state_request)

    @staticmethod
    def public_job(job):
        return {k:v for k,v in job.items() if k != 'request'}

    def job(self, job_id):
        if not re.fullmatch(r'[0-9a-f]{32}', str(job_id)):
            raise ValueError('评分任务 ID 无效')
        return json.loads((self.jobs_root/(job_id+'.json')).read_text())

    def current(self, ep, model_id):
        request = self._request(ep, model_id)
        path = Path(request['output'])
        if path.exists() and path.with_suffix('.npz').is_file():
            result = json.loads(path.read_text())
            if result['signature'] != request['signature']:
                raise ValueError('进展缓存签名不匹配')
            if result.get('kind')=='stage_progress':
                result['evidence']=stage_evidence(result,path.with_suffix('.npz'))
            return dict(state='ready', result=result)
        with self.lock:
            job_id = self.active.get(request['signature'])
        if job_id:
            job = self.job(job_id)
            return dict(state=job['state'], job=self.public_job(job))
        # Show the latest failed/interrupted attempt, including after a restart.
        latest = None
        for file in self.jobs_root.glob('*.json'):
            job = json.loads(file.read_text())
            if job.get('signature') == request['signature'] and (latest is None or job['created_at'] > latest['created_at']):
                latest = job
        if latest and latest['state'] != 'complete':
            return dict(state=latest['state'], job=self.public_job(latest))
        return dict(state='absent', message='尚无当前源文件和模型对应的评分')

    def start(self, ep, model_id):
        if not self.owns_jobs:
            raise ValueError('另一个 RoboCurate 进程正在管理本工作区的评分任务，请在该进程中提交')
        request = self._request(ep, model_id)
        if not request['profile']['ready']:
            raise ValueError('模型尚未就绪：' + '、'.join(request['profile']['missing']))
        with self.lock:
            current = self.current(ep, model_id)
            if current['state'] in ('ready', 'queued', 'running'):
                return current
            job_id = uuid.uuid4().hex
            job = dict(id=job_id, signature=request['signature'], episode_id=ep, model_id=model_id,
                       state='queued', message='等待评分', progress=0, created_at=time.time(), request=request)
            path = self.jobs_root/(job_id+'.json')
            atomic_json(path, job)
            self.active[request['signature']] = job_id
            self.pool.submit(self._run, job_id)
            return dict(state='queued', job=self.public_job(job))

    def _run(self, job_id):
        path = self.jobs_root/(job_id+'.json')
        job = self.job(job_id)
        try:
            with self.lock:
                if self.job(job_id)['state'] == 'cancelled':
                    return
                env = {k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
                env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
                with path.with_suffix('.log').open('w') as log:
                    process = subprocess.Popen([str(self.python), '-m', 'warp_progress.worker', str(path)],
                                               cwd=APP, env=env, stdout=log, stderr=subprocess.STDOUT)
                self.processes[job_id] = process
            code = process.wait()
            with self.lock:
                result = self.job(job_id)
                if result['state'] not in ('complete','cancelled','failed'):
                    result.update(state='failed', message=f'评分进程中断（退出码 {code}），详情见任务日志', finished_at=time.time())
                    atomic_json(path, result)
        except Exception as exc:
            result = self.job(job_id)
            if result['state'] != 'cancelled':
                result.update(state='failed', message=str(exc), finished_at=time.time())
                atomic_json(path, result)
        finally:
            with self.lock:
                self.processes.pop(job_id, None)
                if self.active.get(job['signature']) == job_id:
                    self.active.pop(job['signature'], None)

    def cancel(self, job_id):
        if not self.owns_jobs:
            raise ValueError('请在管理该工作区评分任务的 RoboCurate 进程中取消')
        with self.lock:
            job = self.job(job_id)
            if job['state'] in ('queued', 'running'):
                process = self.processes.get(job_id)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=5)
                job = self.job(job_id)
                job.update(state='cancelled', message='评分已取消，可重新提交', finished_at=time.time())
                atomic_json(self.jobs_root/(job_id+'.json'), job)
            return self.public_job(job)

    def close(self):
        if self._lease.closed:return
        with self.lock:
            ids = list(self.active.values())
        for job_id in ids:
            self.cancel(job_id)
        self.pool.shutdown(wait=True, cancel_futures=True)
        if not self._lease.closed:
            if self.owns_jobs:fcntl.flock(self._lease,fcntl.LOCK_UN)
            self._lease.close()
        self.owns_jobs = False

    def for_export(self, ep, policy):
        if self.profile(policy['model_id']).get('kind') == 'stage_progress':
            raise ValueError('阶段模型目前用于任务复核，阶段进度不能作为 WARP 速度权重导出')
        current = self.current(ep, policy['model_id'])
        if current['state'] != 'ready':
            raise ValueError('所选进展模型没有当前有效评分，请先完成评分')
        result = current['result']
        path = self.results_root/(result['signature']+'.npz')
        with np.load(path, allow_pickle=False) as a:
            arrays = {k:a[k] for k in ('timestamp_ns','velocity','valid')}
        return result, arrays

    @staticmethod
    def apply_to_clip(result, arrays, data, policy, fps):
        if result.get('kind') == 'stage_progress':
            raise ValueError('阶段进度与 WARP 速度权重的单位不同')
        config = result['warp_config']
        age = round(config['feature_stride'] * 1e9 / config['fps']) + 1
        velocity, valid = align_velocity(arrays['timestamp_ns'], arrays['velocity'], arrays['valid'], data['timestamp_ns'], age)
        valid &= data['valid']
        weights, eligible = chunk_weights(velocity, valid, policy['horizon'], policy['threshold'], policy['mode'])
        data.update({'warp.velocity':velocity, 'warp.valid':valid, 'warp.eligible':eligible, 'warp.weight':weights})
        return dict(**policy, horizon_s=policy['horizon']/fps, tail_padding=False,
                    signature=result['signature'], checkpoint_sha256=result['checkpoint_sha256'],
                    feature_contract=result['feature_contract'], validation_status=result['validation_status'],
                    domain_note=result['domain_note'], covered_samples=int(valid.sum()),
                    eligible_anchors=int(eligible.sum()), retained_anchors=int((weights > 0).sum()),
                    candidate_anchors=len(weights), weight_sum=float(weights.sum()),
                    gate='v[t+H-1] > threshold; all H frames valid; no clip boundary crossing')
