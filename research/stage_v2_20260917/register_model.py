"""Register only the development-selected candidate which passed the frozen gate."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;APP=ROOT.parents[1]
sys.path.insert(0,str(APP))
from warp_progress.features import atomic_json,installed_backbone_profile,OPEN_BACKBONE_ID
from warp_progress.stages_v2 import load_fusion_checkpoint

def main():
    selection=json.loads((ROOT/'training/selection.json').read_text())
    if not selection['eligible']:raise ValueError('Candidate did not pass the frozen promotion gate')
    source=ROOT/'training/selected.pt';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    _,saved=load_fusion_checkpoint(source,digest)
    assert saved['winner']==selection['winner'] and saved['promotion_eligible']
    assert saved['test_report_sha256']==hashlib.sha256((ROOT/'training/report.json').read_bytes()).hexdigest()
    root=APP/'workspace/warp/models';path=root/'profiles.json';old=json.loads(path.read_text())
    if any(p['id']=='g1-stage-v2' for p in old['models']):raise ValueError('Model is already registered; do not silently replace it')
    checkpoint=root/('g1-stage-v2-'+digest[:12]+'.pt');shutil.copy2(source,checkpoint)
    profile=installed_backbone_profile(root,OPEN_BACKBONE_ID)
    profile.update(id='g1-stage-v2',kind='stage_progress',stage_version=2,
        label='G1 香蕉搬运 · 多模态时序模型 v2（研究版）',
        checkpoint=checkpoint.name,checkpoint_sha256=digest,backbone_dir='dinov2-small',
        task_id=saved['task_id'],cameras=saved['cameras'],state_contract=saved['state_contract'],
        modalities=['vision','measured_joint_state'],validation_status='same_session_development_cv_provisional_annotations',
        task_scope='G1：左桌抓取香蕉模型，搬到右桌，放入篮子。仅适用相同录制设置。',
        domain_note=f"融合双视角、身体与手部状态及近期历史；现有录像五折开发验证宏 F1 {selection['macro_f1']*100:.1f}%。独立新录像和跨场景能力尚未验证，最终入篮需查看视频确认。",
        architecture=saved['config'],annotation_sha256=saved['annotation_sha256'],
        protocol_sha256=saved['protocol_sha256'],test_report_sha256=saved['test_report_sha256'])
    (ROOT/'profiles.before-v2.json').write_text(path.read_text())
    old['models'].append(profile);atomic_json(path,old);atomic_json(ROOT/'installed_model.json',profile)
    print(json.dumps(profile,ensure_ascii=False))

if __name__=='__main__':main()
