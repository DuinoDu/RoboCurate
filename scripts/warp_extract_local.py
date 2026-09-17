#!/usr/bin/env python3
"""Build a task-specific WARP training manifest from saved, reviewed MCAP episodes.

Requires explicit train/validation episode IDs and a shared saved task
instruction. It never invents successful task labels or changes reviews.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
APP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP))
from studio import StudioLibrary
from exporter import concrete_instruction
from warp_progress.features import BACKBONE_ID,BACKBONES,DinoEncoder,extract_mcap,atomic_json,installed_backbone_profile
from warp_progress.core import WarpConfig
from warp_progress.model import load_model,file_sha256


def extract(args):
    workspace=Path(args.workspace).resolve();lib=StudioLibrary(workspace)
    try:
        for root in lib.store.setting('roots',[]):lib.add_root(Path(root))
        selected=set(args.train+args.validation)
        if set(args.train)&set(args.validation):raise ValueError('Train and validation episodes overlap')
        rows=[];blocked=[]
        for ep in lib.order:
            ref=lib.ref(ep)
            if ref.episode_id not in selected:continue
            record=lib.record(ref);review=record['review'];reasons=[]
            if review['grade'] not in ('A','B'):reasons.append('尚未保存可用于训练的人工审核结论')
            if review.get('source_fingerprint')!=lib.fingerprint(ref):reasons.append('审核来源缺失或过期')
            if not concrete_instruction(review.get('instruction')):reasons.append('缺少具体任务指令')
            if review.get('instruction','').strip()!=args.task.strip():reasons.append('保存的任务指令与本次训练任务不同')
            if record['outcome'] is not True:reasons.append('没有成功示范的结局标记')
            if record['cache']!='ready':reasons.append('记录尚未解析')
            if review.get('segments'):reasons.append('本入口训练完整成功记录；有裁剪的记录请先明确完整任务边界')
            if reasons:blocked.append(dict(episode=ref.episode_id,reasons=reasons))
            else:rows.append((ref,record,'validation' if ref.episode_id in args.validation else 'train'))
        missing=selected-{lib.ref(ep).episode_id for ep in lib.order}
        blocked.extend(dict(episode=ep,reasons=['当前工作区未找到']) for ep in sorted(missing))
        plan=dict(task=args.task,train=args.train,validation=args.validation,blocked=blocked,
                  split_policy='Explicit whole MCAP episodes; single-session validation is not independent-session evidence',
                  success_source='operator outcome plus saved human acceptance; no model-inferred success labels')
        output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(output.with_suffix('.plan.json'),plan)
        if args.plan_only:return plan
        if blocked:raise ValueError('训练输入尚未满足条件，详见 '+str(output.with_suffix('.plan.json')))
        if args.backbone:
            profile=installed_backbone_profile(workspace/'warp/models',args.backbone,args.camera,args.view,args.crop)
            config=WarpConfig(feature_stride=args.feature_stride,source_standard_stride=args.source_standard_stride)
        else:
            profile=lib.progress_service.profile(args.model_id)
            model,config,_=load_model(profile['checkpoint'],args.device,profile['checkpoint_sha256']);del model
        encoder=DinoEncoder(profile['backbone_dir'],args.device,profile.get('backbone_sha256'),profile.get('backbone_id',BACKBONE_ID))
        manifest=[];hash_splits={}
        for ref,record,split in rows:
            meta=json.loads((lib.cache_dir(ref)/'meta.json').read_text())
            arrays,descriptor,path=extract_mcap(ref.path,meta,config,profile,workspace/'warp/features',encoder)
            digest=file_sha256(ref.path)
            if digest in hash_splits and hash_splits[digest]!=split:raise ValueError('相同源文件内容跨越了训练与验证集')
            hash_splits[digest]=split
            descriptor['source_sha256']=digest;descriptor['source_hash_kind']='original MCAP bytes';atomic_json(path.with_suffix('.json'),descriptor)
            manifest.append(dict(id=ref.episode_id,split=split,source_sha256=digest,features=str(path),task=args.task,
                                 selection='saved A/B human review + operator success + explicit task and split',
                                 review_revision=record['review']['revision']))
        result=dict(episodes=manifest,warp_config=asdict(config),split_policy=plan['split_policy'])
        atomic_json(output,result);return result
    finally:
        lib.progress_service.close();lib.pool.shutdown();lib.export_pool.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',default=str(APP/'workspace'));p.add_argument('--output',required=True)
    p.add_argument('--model-id',default='paper-sim-sss15-head-left');p.add_argument('--device',default='cpu')
    p.add_argument('--backbone',choices=list(BACKBONES),help='Extract features for a fresh head without requiring an existing head')
    p.add_argument('--camera',choices=['head','left_wrist','right_wrist'],default='head')
    p.add_argument('--view',choices=['full','left','right'],default='left');p.add_argument('--crop',choices=['squash','center'],default='squash')
    p.add_argument('--feature-stride',type=int,default=3);p.add_argument('--source-standard-stride',type=int,default=45)
    p.add_argument('--task',required=True);p.add_argument('--train',nargs='+',required=True)
    p.add_argument('--validation',nargs='+',required=True);p.add_argument('--plan-only',action='store_true')
    result=extract(p.parse_args());print(json.dumps(result,ensure_ascii=False,indent=2))
