"""Install the validation-selected stage model alongside existing WARP profiles."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parent
APP=ROOT.parents[1]
sys.path.insert(0,str(APP))
from warp_progress.features import atomic_json,installed_backbone_profile,OPEN_BACKBONE_ID
from warp_progress.stages import load_stage_checkpoint

def main():
    source=ROOT/'stage_training/selected.pt'
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    _,saved=load_stage_checkpoint(source,sha)
    root=APP/'workspace/warp/models'
    checkpoint=root/('g1-stage-v1-'+sha[:12]+'.pt')
    shutil.copy2(source,checkpoint)
    profile=installed_backbone_profile(root,OPEN_BACKBONE_ID)
    profile.update(id='g1-stage-v1',kind='stage_progress',label='G1 香蕉搬运 · 双视角阶段模型（研究版）',
        checkpoint=checkpoint.name,checkpoint_sha256=sha,backbone_dir='dinov2-small',
        task_id=saved['task_id'],cameras=saved['cameras'],
        validation_status='same_session_provisional_annotations',
        task_scope='G1：左桌抓取香蕉模型，搬到右桌，放入篮子。仅适用相同录制设置。',
        domain_note='针对香蕉搬运训练；5 段留出录像的研究标注上阶段识别准确率 90.0%。独立人工验收与跨场景能力尚未验证；最终入篮需查看视频确认。',
        architecture=saved['config'],annotation_sha256=saved['annotation_sha256'],
        protocol_sha256=saved['protocol_sha256'],test_report_sha256=saved['test_report_sha256'])
    path=root/'profiles.json';old=json.loads(path.read_text())
    if any(p['id']==profile['id'] for p in old['models']):
        raise ValueError('Stage model already registered; do not silently replace an existing model')
    (ROOT/'profiles.before-stage.json').write_text(path.read_text())
    old['models'].append(profile)
    atomic_json(path,old)
    atomic_json(ROOT/'installed_model.json',profile)
    print(json.dumps(profile,ensure_ascii=False))

if __name__=='__main__':main()
