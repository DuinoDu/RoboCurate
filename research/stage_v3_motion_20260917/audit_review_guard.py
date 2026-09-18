"""Fixed post-selection safeguard: calibration must not erase raw uncertainty.

No model or temperature changes; no threshold search. Decision uses development
OOF only, and is an explicitly subsequent experiment on already-used data.
"""
import importlib.util,hashlib,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;FIRST=ROOT.parent/'stage_v3_20260917'
spec=importlib.util.spec_from_file_location('guard_engine',FIRST/'train.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine);engine.ROOT=ROOT
from warp_progress.stages_v3 import inference_v3,load_stage_v3
from warp_progress.features import atomic_json

def run(models,rows,ids,prior,temp,boundary,guard):
    n=reviewed=errors=unflagged=0;curves={}
    for i in ids:
        r=rows[i];out=inference_v3(models,r['features'],r['camera_valid'],r['state'],r['state_valid'],prior,temp,boundary,guard)
        ix=r['points_ix'];y=r['points_y'];keep=(y>=0)&r['camera_valid'].all(1)[ix]&r['state_valid'][ix]
        ix=ix[keep];y=y[keep];wrong=out['stage'][ix]!=y
        n+=len(y);errors+=int(wrong.sum());reviewed+=int(out['review'][ix].sum());unflagged+=int((wrong&~out['review'][ix]).sum())
        curves[str(i)]=dict(stage=out['stage'].tolist(),review=out['review'].tolist(),review_reasons=out['review_reasons'].tolist())
    return dict(n=n,errors=errors,reviewed=reviewed,review_fraction=reviewed/max(1,n),unflagged_mistakes=unflagged),curves

def main():
    torch.set_num_threads(2);out=ROOT/'review_guard';out.mkdir(exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Guard experiment already run')
    p,_,_,rows,_=engine.load_data();selection=json.loads((ROOT/'training/selection.json').read_text())
    name=selection['winner'];temp=selection['temperature'];boundary=selection['review_policy']['adopt_boundary']
    protocol=dict(method='Review if either raw T=1 or calibrated ensemble requests review; predictions and threshold 0.8 unchanged.',
        gate='Fewer development OOF unflagged mistakes and added reviewed fraction <= 0.15; otherwise leave disabled.',
        chronology='Post-classifier-selection iteration prompted by calibration/review conflict; not an independent test.',
        classifier_selection_sha256=hashlib.sha256((ROOT/'training/selection.json').read_bytes()).hexdigest())
    atomic_json(out/'protocol.json',protocol)
    scores={}
    for guard in [False,True]:
        totals=dict(n=0,errors=0,reviewed=0,unflagged_mistakes=0)
        for f,held in enumerate(p['folds']):
            models,saved=engine.load_fold(name,f,ROOT/'training')
            score,_=run(models,rows,held,saved['priors'],temp,boundary,guard)
            for k in totals:totals[k]+=score[k]
        totals['review_fraction']=totals['reviewed']/totals['n'];scores[str(guard)]=totals
    before,after=scores['False'],scores['True']
    enabled=after['unflagged_mistakes']<before['unflagged_mistakes'] and after['review_fraction']-before['review_fraction']<=.15
    decision=dict(enabled=enabled,development=scores,protocol_sha256=hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest())
    atomic_json(out/'selection.json',decision)
    checkpoint=ROOT/'training/selected.pt';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    models,saved=load_stage_v3(checkpoint,digest)
    old,_=run(models,rows,p['historical_regression'],saved['priors'],temp,boundary,False)
    new,curves=run(models,rows,p['historical_regression'],saved['priors'],temp,boundary,enabled)
    assert old['errors']==new['errors']
    decision.update(historical_before=old,historical_after=new,classifier_weights_and_predictions_unchanged=True)
    atomic_json(out/'report.json',decision);atomic_json(out/'regression_curves.json',curves)
    print(json.dumps(decision,ensure_ascii=False))

if __name__=='__main__':main()
