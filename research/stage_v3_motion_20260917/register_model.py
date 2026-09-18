"""Promote only a frozen-gate winner, chosen by development metrics across both studies."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;APP=ROOT.parents[1]
sys.path.insert(0,str(APP))
from warp_progress.features import atomic_json,installed_backbone_profile,OPEN_BACKBONE_ID
from warp_progress.stages_v3 import load_stage_v3
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    decision=json.loads((ROOT/'promotion_protocol.json').read_text())
    candidates=[]
    for item in decision['studies']:
        folder=ROOT.parent/item['directory'];assert sha(folder/'protocol.json')==item['protocol_sha256']
        s=json.loads((folder/'training/selection.json').read_text())
        if s['eligible']:
            cv=json.loads((folder/'training/cross_validation.json').read_text())[s['winner']]['overall']
            candidates.append((cv['macro_f1'],-cv['nll'],folder,s))
    if not candidates:
        atomic_json(ROOT/'promotion.json',dict(promoted=False,reason='No classifier passed its frozen development gate; existing v2 retained.'))
        print('No eligible classifier; v2 retained');return
    _,_,folder,selection=max(candidates,key=lambda x:x[:2])
    source=folder/'training/selected.pt';digest=sha(source);_,saved=load_stage_v3(source,digest)
    assert saved['promotion_eligible'] and saved['test_report_sha256']==sha(folder/'training/report.json')
    root=APP/'workspace/warp/models';path=root/'profiles.json';profiles=json.loads(path.read_text())
    if any(p['id']=='g1-stage-v3' for p in profiles['models']):raise ValueError('Model already registered; no silent replacement')
    checkpoint=root/('g1-stage-v3-'+digest[:12]+'.pt');shutil.copy2(source,checkpoint)
    label='运动感知' if saved['config']['architecture']=='motion' else '长短期时序'
    profile=installed_backbone_profile(root,OPEN_BACKBONE_ID)
    profile.update(id='g1-stage-v3',kind='stage_progress',stage_version=3,
        label=f'G1 香蕉搬运 · {label}模型 v3（研究版）',checkpoint=checkpoint.name,checkpoint_sha256=digest,backbone_dir='dinov2-small',
        task_id=saved['task_id'],cameras=saved['cameras'],state_contract=saved['state_contract'],modalities=['vision','measured_joint_state'],
        validation_status='iterated_same_session_development_cv_provisional_annotations',
        task_scope='G1：左桌抓取香蕉模型，搬到右桌，放入篮子。仅适用相同录制设置。',
        domain_note=f"{label}网络；现有录像五折开发验证宏 F1 {selection['macro_f1']*100:.1f}%。反复使用同场景开发数据，独立新录像和跨场景能力尚未验证；最终入篮需人工确认。",
        architecture=saved['config'],annotation_sha256=saved['annotation_sha256'],protocol_sha256=saved['protocol_sha256'],test_report_sha256=saved['test_report_sha256'])
    (ROOT/'profiles.before-v3.json').write_text(path.read_text())
    profiles['models'].append(profile);atomic_json(path,profiles);atomic_json(ROOT/'installed_model.json',profile)
    atomic_json(ROOT/'promotion.json',dict(promoted=True,study=folder.name,winner=selection['winner'],model_id=profile['id'],checkpoint_sha256=digest,
        selection_basis=decision['selection'],source_checkpoint=str(source)))
    print(json.dumps(dict(promoted=True,study=folder.name,winner=selection['winner'],model_id=profile['id']),ensure_ascii=False))

if __name__=='__main__':main()
