"""Attach a development-selected review safeguard before the first v3 release.

Keep the classifier checkpoint, first registration and all experiments intact.
The added rule changes review flags only; learned weights and calibration remain
bit-identical. Refuse to replace an already-scored model or repeat this operation.
"""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;APP=ROOT.parents[1]
sys.path.insert(0,str(APP))
from warp_progress.features import atomic_json
from warp_progress.stages_v3 import load_stage_v3
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (ROOT/'finalization.json').exists(),'Already finalized; no silent replacement'
    guard=ROOT/'review_guard';selection=json.loads((guard/'selection.json').read_text())
    report=json.loads((guard/'report.json').read_text())
    assert selection['enabled'] and report['classifier_weights_and_predictions_unchanged']
    assert selection['protocol_sha256']==sha(guard/'protocol.json')
    metrics=selection['development'];before=metrics['False'];after=metrics['True']
    assert after['unflagged_mistakes']<before['unflagged_mistakes']
    assert after['review_fraction']-before['review_fraction']<=.15
    for p in (APP/'workspace/warp/scores').glob('*.json'):
        assert json.loads(p.read_text()).get('model_id')!='g1-stage-v3','Refusing to change an already-scored release'
    path=APP/'workspace/warp/models/profiles.json';profiles=json.loads(path.read_text())
    candidates=[p for p in profiles['models'] if p['id']=='g1-stage-v3'];assert len(candidates)==1
    profile=candidates[0];original=json.loads((ROOT/'installed_model.json').read_text());assert profile==original
    source=path.parent/profile['checkpoint'];_,saved=load_stage_v3(source,profile['checkpoint_sha256'])
    assert profile['checkpoint_sha256']==sha(ROOT/'training/selected.pt')
    updated=dict(saved,review_guard=True,review_guard_report_sha256=sha(guard/'report.json'),
        review_guard_protocol_sha256=sha(guard/'protocol.json'),classifier_source_sha256=sha(source))
    final=ROOT/'training/release.pt';assert not final.exists();torch.save(updated,final)
    digest=sha(final);_,check=load_stage_v3(final,digest)
    assert check['temperature']==saved['temperature'] and check['config']==saved['config']
    for old,new in zip(saved['states'],check['states']):
        assert old.keys()==new.keys()
        for key in old:assert torch.equal(old[key],new[key]),key
    checkpoint=path.parent/('g1-stage-v3-'+digest[:12]+'.pt');assert not checkpoint.exists()
    checkpoint.write_bytes(final.read_bytes())
    atomic_json(ROOT/'installed_model.before-review-guard.json',original)
    atomic_json(ROOT/'profiles.before-finalization.json',profiles)
    profile.update(checkpoint=checkpoint.name,checkpoint_sha256=digest,review_guard_enabled=True,
        review_guard_report_sha256=sha(guard/'report.json'),classifier_source_sha256=sha(source))
    atomic_json(path,profiles);atomic_json(ROOT/'installed_model.json',profile)
    result=dict(finalized=True,model_id=profile['id'],classifier_checkpoint_sha256=sha(source),
        release_checkpoint_sha256=digest,review_guard_report_sha256=sha(guard/'report.json'),
        learned_tensors_bit_identical=True,calibration_unchanged=True,stage_predictions_unchanged=True,
        prior_checkpoint_preserved=source.name,scope='Pre-release review flags only; before first production score')
    atomic_json(ROOT/'finalization.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
