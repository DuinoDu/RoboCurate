#!/usr/bin/env python3
"""Register a locally trained, provenance-bearing WARP model for the review UI."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
APP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP))
from warp_progress.features import BACKBONES,atomic_json
from warp_progress.model import load_model


def register(args):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',args.id):raise ValueError('Invalid model ID')
    root=Path(args.workspace).resolve()/'warp/models'
    model,config,checkpoint=load_model(args.checkpoint)
    del model
    audit=checkpoint.get('provenance',{})
    contract=audit.get('feature_contract',{})
    spec=BACKBONES.get(contract.get('backbone'))
    if not spec or contract.get('dimension')!=spec['dimension'] or not audit.get('task'):
        raise ValueError('Checkpoint must carry supported real DINO feature and task provenance')
    if contract.get('feature_stride')!=config.feature_stride or contract.get('fps')!=config.fps:
        raise ValueError('Checkpoint time scale differs from its feature provenance')
    backbone=json.loads((root/spec['provenance']).read_text())
    if backbone['repo']!=contract['backbone'] or contract.get('backbone_revision')!=backbone['revision'] or contract.get('backbone_sha256')!=backbone['sha256']:
        raise ValueError('Installed backbone differs from the model training backbone')
    transfer_camera=getattr(args,'transfer_camera',None)
    if not transfer_camera and (contract.get('camera') not in ('head','left_wrist','right_wrist') or contract.get('view') not in ('full','left','right')):
        raise ValueError('This UI registration entry point requires a local MCAP camera contract')
    if not transfer_camera and (contract.get('decoder')!='Pillow-RGB' or contract.get('crop') not in ('squash','center')):
        raise ValueError('Use the same local image decoder and geometry for training and scoring')
    target=root/(args.id+'-'+checkpoint['sha256'][:12]+'.pt')
    if Path(args.checkpoint).resolve()!=target.resolve():shutil.copyfile(args.checkpoint,target)
    profile=dict(id=args.id,label=args.label,checkpoint=target.name,checkpoint_sha256=checkpoint['sha256'],
        backbone_id=contract['backbone'],backbone_dir=spec['directory'],backbone_revision=backbone['revision'],backbone_sha256=backbone['sha256'],
        camera=transfer_camera or contract['camera'],view=args.transfer_view if transfer_camera else contract['view'],crop=contract['crop'],task_scope=audit['task'],
        validation_status='local_unvalidated',domain_note='本地任务模型：'+audit['task']+'。当前仅完成时间扭曲验证，机器人任务效果尚未验证。',
        training_manifest_sha256=audit['manifest_sha256'],training_step=checkpoint['step'])
    if transfer_camera:
        profile.update(validation_status='transfer_unvalidated',training_feature_contract=contract,
            domain_note='公开任务小样本模型：'+audit['task']+'。当前为 G1 视角迁移试算，尚未验证本地任务判别效果；候选需结合视频复核。')
    path=root/'profiles.json';rows=json.loads(path.read_text())['models'] if path.exists() else []
    atomic_json(path,dict(version=1,models=[profile,*[p for p in rows if p['id']!=args.id]]))
    print(json.dumps(dict(registered=args.id,checkpoint_sha256=checkpoint['sha256'],validation_status=profile['validation_status']),ensure_ascii=False))
    return profile


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',default=str(APP/'workspace'));p.add_argument('--checkpoint',required=True)
    p.add_argument('--id',required=True);p.add_argument('--label',required=True)
    p.add_argument('--transfer-camera',choices=['head','left_wrist','right_wrist'],help='Explicitly register a different camera/domain as unvalidated transfer')
    p.add_argument('--transfer-view',choices=['full','left','right'],default='left')
    register(p.parse_args())
