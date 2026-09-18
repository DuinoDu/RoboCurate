"""Package-relative verification of the retained model and experiment integrity."""
import contextlib,hashlib,importlib.util,io,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;APP=ROOT.parents[1];OLD=ROOT.parent/'stage_v3_motion_20260917'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    decision=json.loads((ROOT/'promotion.json').read_text());assert not decision['promoted']
    for s in decision['studies']:
        path=ROOT.parent/s['directory']
        assert sha(path/'protocol.json')==s['protocol_sha256']
        assert sha(path/'training/cross_validation.json')==s['cv_sha256']
        assert not any(s['eligibility'].values())
    profile=json.loads((ROOT/'active_model.json').read_text())
    installed=json.loads((APP/'workspace/warp/models/profiles.json').read_text())['models']
    assert profile==next(p for p in installed if p['id']==decision['active_model'])
    assert not any(p.get('stage_version')==4 for p in installed)
    spec=importlib.util.spec_from_file_location('v3_package_verifier',OLD/'verify_delivery.py')
    verify=importlib.util.module_from_spec(spec);spec.loader.exec_module(verify)
    output=io.StringIO()
    with contextlib.redirect_stdout(output):verify.main()
    active=json.loads(output.getvalue());assert active['passed']
    spec=importlib.util.spec_from_file_location('v4_experiment_verifier',ROOT/'verify_experiments.py')
    experiments=importlib.util.module_from_spec(spec);spec.loader.exec_module(experiments)
    checked=experiments.main();assert checked['passed']
    print(json.dumps(dict(passed=True,promoted=False,active_model=decision['active_model'],active_model_verification=active,
        experiment_verification=checked),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
