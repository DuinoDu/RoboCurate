"""Descriptive post-selection diagnostics; never changes weights or selection."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from train_cv import metrics,atomic_json

def main():
    cv=json.loads((ROOT/'training/cross_validation.json').read_text())
    selection=json.loads((ROOT/'training/selection.json').read_text());name=selection['winner']
    with np.load(ROOT/'training'/f'{name}-oof.npz') as z:w={k:z[k] for k in z.files}
    with np.load(ROOT/'training/baseline_mlp-oof.npz') as z:b={k:z[k] for k in z.files}
    assert np.array_equal(w['y'],b['y']) and np.array_equal(w['episode'],b['episode'])
    rng=np.random.default_rng(720);eps=np.unique(w['episode']);diff=[]
    for _ in range(5000):
        ix=np.concatenate([np.flatnonzero(w['episode']==e) for e in rng.choice(eps,len(eps),replace=True)])
        diff.append(metrics(w['y'][ix],w['p'][ix])['macro_f1']-metrics(b['y'][ix],b['p'][ix])['macro_f1'])
    mistakes=[];n=0;reviewed=0
    for ep,c in json.loads((ROOT/'training/regression_curves.json').read_text()).items():
        for j,label in zip(c['annotation_indices'],c['annotation_stages']):
            n+=1;reviewed+=c['review'][j]
            if c['stage'][j]!=label:
                mistakes.append(dict(episode=ep,time_s=c['times_s'][j],annotation=label,predicted=c['stage'][j],confidence=c['confidence'][j],review=c['review'][j]))
    result=dict(recording_bootstrap=dict(draws=5000,seed=720,
        macro_f1_difference=selection['macro_f1']-selection['baseline_mlp_f1'],percentile_95_interval=np.quantile(diff,[.025,.975]).tolist(),
        limitation='Descriptive development CV interval, same-session provisional labels; candidate selection makes this optimistic. Not independent proof.'),
        cross_validation={k:v['overall'] for k,v in cv.items()},
        historical_regression=dict(n=n,reviewed=reviewed,mistakes=mistakes,unflagged_mistakes=sum(not x['review'] for x in mistakes)))
    atomic_json(ROOT/'diagnostics.json',result);print(json.dumps(result['recording_bootstrap']))

if __name__=='__main__':main()
