"""Correct calibration of the first running experiment, without retraining or reselection.

The initial process already loaded the old calibration code. All fold fits and
selection at T=1 are unchanged. Fix member/ensemble ordering using development
OOF predictions only, then recompute derived regression artifacts at that T.
Raw first-pass reports are retained. This is a measurement correction, not a
second hyperparameter search on historical regression recordings.
"""
import hashlib,json,shutil
from pathlib import Path
import torch
from train_cv import ROOT,load_data,make_model,calibrate_oof,evaluate,sha,atomic_json

def state_digest(states):
    h=hashlib.sha256()
    for state in states:
        for k,v in sorted(state.items()):h.update(k.encode());h.update(v.numpy().tobytes())
    return h.hexdigest()

def main():
    torch.set_num_threads(2)
    out=ROOT/'training';audit=out/'calibration_correction.json'
    if audit.exists():raise ValueError('Calibration correction already completed')
    protocol,_,_,rows,_=load_data()
    selection=json.loads((out/'selection.json').read_text())
    report=json.loads((out/'report.json').read_text())
    saved=torch.load(out/'selected.pt',map_location='cpu',weights_only=True)
    original=state_digest(saved['states']);name=selection['winner']
    cal=calibrate_oof(name,protocol,rows,out)
    chosen=min(cal,key=lambda r:r['nll'])['temperature']
    raw=out/'before_calibration_correction';raw.mkdir(exist_ok=False)
    for filename in ['selection.json','report.json','regression_curves.json','stress_curves.json','selected.pt']:
        shutil.copy2(out/filename,raw/filename)
    old_temperature=selection['temperature']
    selection.update(temperature=chosen,calibration=cal,calibration_method='Each member probability temperature-scaled before ensemble mean; development OOF only')
    atomic_json(out/'selection.json',selection)
    models=[]
    for state in saved['states']:
        model=make_model(name);model.load_state_dict(state);models.append(model.eval())
    prior=saved['priors']
    regression,curves,_=evaluate(models,rows,protocol['historical_regression'],prior,chosen)
    controls={drop:evaluate(models,rows,protocol['historical_regression'],prior,chosen,drop)[0]
              for drop in ['head','wrist','state','all_cameras']}
    stress={}
    for i in protocol['stress']:stress.update(evaluate(models,rows,[i],prior,chosen)[1])
    report.update(selection=selection,historical_regression=regression,controlled_missing_inputs=controls,
                  calibration_note='Calibration order corrected using development predictions only; raw first pass archived; winner and all learned weights unchanged.')
    atomic_json(out/'report.json',report);atomic_json(out/'regression_curves.json',curves);atomic_json(out/'stress_curves.json',stress)
    saved.update(temperature=chosen,test_report_sha256=sha(out/'report.json'))
    torch.save(saved,out/'selected.pt')
    assert original==state_digest(saved['states'])
    atomic_json(audit,dict(reason=report['calibration_note'],old_temperature=old_temperature,
        temperature=chosen,winner=name,learned_states_sha256=original,weights_unchanged=True,protocol_sha256=sha(ROOT/'protocol.json')))
    print(json.dumps(dict(winner=name,temperature=chosen,macro_f1=regression['macro_f1']),ensure_ascii=False))

if __name__=='__main__':main()
