#!/usr/bin/env python3
"""Bounded, resumable real-video validation after the user's model-access request.

It checks an existing access grant; it never submits/accepts license forms.
Model scoring is offline after the official files are downloaded.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import urllib.request
APP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP))
from scripts.warp_prepare_models import prepare
from scripts.warp_reproduce import reproduce
from warp_progress.features import atomic_json
from warp_progress.model import file_sha256


def resume(args):
    workspace=Path(args.workspace).resolve()
    research=Path(args.research).resolve()
    status_path=workspace/'warp/workflow/status.json'
    status=dict(state='starting',started_at=time.time(),stages={},limitations=[
        'Public checkpoint inference reproduction; no full policy-training or robot rollout experiment',
        'G1 scores from sim-trained head are transfer trials, not validated task labels'])
    def save(state,message):
        status.update(state=state,message=message,updated_at=time.time());atomic_json(status_path,status)
        print(message,flush=True)
    deadline=time.monotonic()+args.wait_minutes*60
    try:
        while True:
            try:
                save('preparing_models','检查已申请的 DINOv3 访问权限与本地模型')
                prepare(workspace)
                status['stages']['models']='complete'
                break
            except SystemExit as exc:
                if exc.code==3:
                    save('access_rejected','模型作者已拒绝当前账号的访问申请，已停止重试；需要已授权账号或已授权本地权重')
                    return status
                if exc.code!=2:raise
                if time.monotonic()>=deadline:
                    save('awaiting_access','模型访问仍未获批；本轮等待已结束，获批后可重新执行此脚本')
                    return status
                save('awaiting_access','等待已提交的模型访问申请获批，60 秒后再次检查')
                time.sleep(min(60,max(0,deadline-time.monotonic())))
        public_args=argparse.Namespace(workspace=str(workspace),
            video=str(research/'sources/public_top_camera_file_00007.mp4'),
            video_sha256='14ceccf471beb2b97ab4cd04dc6b5a0d986659afa17bc374323c7ad8d963fe4d',
            episode_metadata=str(research/'sources/public_episodes.parquet'),episode_index=2316,
            output=str(research/'public_inference'),threads=4,device='cpu',feature_batch=8,score_batch=16,threshold=1.)
        save('public_validation','在公开第 2316 段视频上对照作者推理代码')
        report=reproduce(public_args)
        status['stages']['public_inference']=dict(state='complete',report=str(research/'public_inference/report.json'),
                                                  max_reference_velocity_error=report['max_reference_velocity_error'])
        # A single job owner prevents CLI and GUI from interrupting one another.
        from studio import StudioLibrary
        lib=StudioLibrary(workspace)
        try:
            for root in lib.store.setting('roots',[]):
                lib.add_root(Path(root))
            records=[lib.ref(ep) for ep in lib.order if lib.ref(ep).episode_id in args.episodes]
            if len(records)!=len(set(args.episodes)):
                raise ValueError('部分指定的 G1 记录未出现在当前工作区')
            remote=False
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if not lib.progress_service.owns_jobs:
                with opener.open(args.server+'/api/health',timeout=5) as response:health=json.load(response)
                if health.get('workspace')!=str(workspace):raise ValueError('本机服务使用了另一个工作区')
                remote=True
            def api(route,body=None):
                request=urllib.request.Request(args.server+route,
                    data=json.dumps(body).encode() if body is not None else None,
                    headers={'Content-Type':'application/json'})
                with opener.open(request,timeout=15) as response:return json.load(response)
            for ref in records:
                save('local_scoring','计算本地 '+ref.episode_id+' 的迁移试算曲线')
                model_id='paper-sim-sss15-head-left'
                if remote:api('/api/progress/jobs',dict(ep=ref.id,model_id=model_id))
                else:lib.progress_service.start(ref.id,model_id)
                while True:
                    current=api('/api/progress?ep='+ref.id+'&model_id='+model_id) if remote else lib.progress_service.current(ref.id,model_id)
                    if current['state']=='ready':break
                    if current['state'] not in ('queued','running'):
                        raise RuntimeError(current.get('job',{}).get('message',current['state']))
                    time.sleep(3)
                result=current['result']
                status['stages'][ref.episode_id]=dict(state='complete',episode_id=ref.id,signature=result['signature'],
                    coverage=result['summary']['coverage'],validation_status=result['validation_status'])
                atomic_json(status_path,status)
        finally:
            lib.progress_service.close();lib.pool.shutdown();lib.export_pool.shutdown()
        save('complete','公开推理对照和指定 G1 记录评分已完成；G1 任务效果仍需人工与策略实验验证')
        return status
    except BaseException as exc:
        save('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',str(exc))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',default=str(APP/'workspace'))
    p.add_argument('--research',default=str(APP.parent/'research/warp_rm_20260916'))
    p.add_argument('--wait-minutes',type=float,default=0)
    p.add_argument('--server',default='http://127.0.0.1:8421')
    p.add_argument('--episodes',nargs='+',default=['episode_000124','episode_000125','episode_000133','episode_000141'])
    args=p.parse_args()
    if not 0<=args.wait_minutes<=240:p.error('--wait-minutes must be between 0 and 240')
    resume(args)
