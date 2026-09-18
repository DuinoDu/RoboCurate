"""Check the predeclared review policy on unchanged deployed v2 weights.

This can improve review tooling if no new classifier qualifies; it never relaxes
the model promotion gate or changes v2 weights/temperature/stage predictions.
"""
import importlib.util,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;FIRST=ROOT.parent/'stage_v3_20260917';V2=ROOT.parent/'stage_v2_20260917';APP=ROOT.parents[1]
spec=importlib.util.spec_from_file_location('v3_review_engine',FIRST/'train.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
from warp_progress.stages_v2 import load_fusion_checkpoint
from warp_progress.features import atomic_json

def main():
    torch.set_num_threads(2);p,_,_,rows,_=engine.load_data()
    profile=json.loads((V2/'installed_model.json').read_text())
    models,saved=load_fusion_checkpoint(APP/'workspace/warp/models'/profile['checkpoint'],profile['checkpoint_sha256'])
    audit=engine.policy_audit('v2_reference',rows,p,FIRST/'training',saved['temperature'])
    # Only after the development decision is fixed, inspect previously-seen regression.
    result=dict(source_model_id=profile['id'],source_checkpoint_sha256=profile['checkpoint_sha256'],temperature_unchanged=saved['temperature'],
        classifier_weights_unchanged=True,review_policy=audit,eligible=audit['adopt_boundary'])
    atomic_json(ROOT/'review_fallback_selection.json',result)
    old,_,_=engine.evaluate(models,rows,p['historical_regression'],saved['priors'],saved['temperature'],False)
    new,curves,_=engine.evaluate(models,rows,p['historical_regression'],saved['priors'],saved['temperature'],audit['adopt_boundary'])
    result.update(historical_before=old,historical_after=new)
    atomic_json(ROOT/'review_fallback_report.json',result);atomic_json(ROOT/'review_fallback_curves.json',curves)
    print(json.dumps(dict(development=audit,historical_unflagged_before=old['unflagged_mistakes'],historical_unflagged_after=new['unflagged_mistakes'])))

if __name__=='__main__':main()
