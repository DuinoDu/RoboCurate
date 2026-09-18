"""Recheck the unchanged v3 on all 18 recordings after research integration."""
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'stage_v3_motion_20260917'
spec=importlib.util.spec_from_file_location('v3_runtime_verifier',OLD/'verify_runtime.py')
verify=importlib.util.module_from_spec(spec);spec.loader.exec_module(verify)
if __name__=='__main__':
    decision=json.loads((ROOT/'promotion.json').read_text());assert not decision['promoted']
    # Redirect only the final output; all reference metadata remains immutable.
    original=verify.atomic_json
    def save(path,value):
        assert path.name=='runtime_verification.json'
        original(ROOT/path.name,dict(value,active_model=decision['active_model'],checkpoint_sha256=decision['checkpoint_sha256']))
    verify.atomic_json=save;verify.main()
