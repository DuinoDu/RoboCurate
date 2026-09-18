"""Audit all frozen gates and preserve the deployed classifier when none pass."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;APP=ROOT.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    studies=[]
    for folder in [ROOT,ROOT.parent/'stage_v4_decoupled_20260917']:
        p=read(folder/'protocol.json');frozen=read(folder/'freeze.json')['protocol_sha256']
        assert sha(folder/'protocol.json')==frozen
        report=read(folder/'training/cross_validation.json');base=report['v3_reference']['overall'];gates={}
        for name,r in report.items():
            if name=='v3_reference':continue
            s=r['overall'];b=base['bracket_diagnostic'];d=s['bracket_diagnostic']
            stable=s['macro_f1']>=base['macro_f1']-1e-10 and s['nll']<=base['nll']+.02
            improved=s['macro_f1']>=base['macro_f1']+.005 or s['nll']<=base['nll']-.01 or (d['matched']>=b['matched']+2 and d['extra_switches']<=b['extra_switches'])
            gates[name]=bool(stable and improved)
        expected=read(folder/'training/selection.json')['eligibility']
        assert all(expected[n]==v for n,v in gates.items())
        if folder!=ROOT:
            proxy=read(folder/'training/progress_proxy_audit.json');gates['detached_progress'] &= proxy['eligible']
        assert not any(gates.values()),'An eligible model requires separate integration and validation, not automatic installation.'
        studies.append(dict(directory=folder.name,protocol_sha256=frozen,cv_sha256=sha(folder/'training/cross_validation.json'),eligibility=gates))
    old=read(ROOT.parent/'stage_v3_motion_20260917/installed_model.json')
    profiles=read(APP/'workspace/warp/models/profiles.json')['models']
    current=next(p for p in profiles if p['id']==old['id'])
    assert current==old and sha(APP/'workspace/warp/models'/old['checkpoint'])==old['checkpoint_sha256']
    assert not any(p.get('stage_version')==4 for p in profiles)
    write(ROOT/'active_model.json',current)
    write(ROOT/'promotion.json',dict(promoted=False,active_model=current['id'],checkpoint_sha256=current['checkpoint_sha256'],
        prior_profile_unchanged=True,studies=studies,
        reason='No new candidate passed its frozen development gate. New training objectives remain research tools; deployed stage classification remains v3.'))
    print('Audited 4 candidates: no promotion; installed v3 profile and checkpoint unchanged')
if __name__=='__main__':main()
