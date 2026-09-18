"""Correct comparison support without retraining, model reselection, or changing labels."""
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from train_stages import ROOT, read_study, evaluate, predict, metrics, digest, StageNet, atomic_json

def main():
    torch.set_num_threads(2)
    protocol,_,_,rows,priors=read_study()
    out=ROOT/'stage_training'
    raw=out/'test_report.raw.json'
    if raw.exists():
        raise ValueError('Support audit already applied')
    shutil.copy2(out/'test_report.json',raw)
    shutil.copy2(out/'test_curves.json',out/'test_curves.raw.json')
    report=json.loads(raw.read_text())
    candidates={}
    for name in protocol['candidates']:
        saved=torch.load(out/(name+'.pt'),weights_only=True)
        candidates[name]=[]
        for state in saved['states']:
            m=StageNet(**saved['config']);m.load_state_dict(state);candidates[name].append(m.eval())
        report['results'][name],_=evaluate(candidates[name],rows,protocol['test'],priors,saved['config']['views'],common_support=True)
    winner=report['selection']['winner']; models=candidates[winner]
    views=models[0].config['views'];temperature=report['selection']['temperature']
    curves=json.loads((out/'test_curves.json').read_text())
    for control in [None,'reverse','shuffle','constant']:
        key=control or 'selected_calibrated'
        report['results'][key],curves[key]=evaluate(models,rows,protocol['test'],priors,views,temperature,control,common_support=True)
    relative=np.mean([np.array(rows[i]['bounds'][1:])/rows[i]['times'][-1] for i in protocol['train']],0)
    yt,pt=[],[]
    for i in protocol['test']:
        r=rows[i];y=r['points_y'];ok=(y>=0)&r['camera_valid'].all(1)[r['points_ix']];ix=r['points_ix'][ok]
        yp=np.searchsorted(relative,r['times'][ix]/r['times'][-1],side='right')
        yt.extend(y[ok]);pt.extend(np.eye(5)[yp]*.96+.008)
    report['results']['clock_prior']=metrics(np.asarray(yt),np.asarray(pt))
    # Cluster bootstrap: recordings, not highly correlated video frames, are resampled.
    samples=[]; rng=np.random.default_rng(1709)
    m_selected=np.array([report['results']['selected_calibrated']['per_episode'][str(i)]['confusion'] for i in protocol['test']])
    m_head=np.array([report['results']['head_mlp']['per_episode'][str(i)]['confusion'] for i in protocol['test']])
    def f1(c): return float(np.mean(2*np.diag(c)/np.maximum(1,c.sum(0)+c.sum(1))))
    for _ in range(5000):
        ix=rng.integers(0,len(protocol['test']),len(protocol['test']))
        samples.append(f1(m_selected[ix].sum(0))-f1(m_head[ix].sum(0)))
    report['comparison_support']=dict(known_points=75,common_valid_points=70,
        rule='Identical known annotation points with valid head and right-wrist images. Five initial points omitted because no earlier camera image exists. Shuffle additionally has one invalid shuffled frame.',
        revision='Initial report compared available support (head 71, dual 70, clock 75). This audit fixes support only; all weights, validation selection, labels, temperature and thresholds are unchanged.',
        selected_minus_head_macro_f1_ci95=np.quantile(samples,[.025,.975]).tolist(),
        bootstrap_unit='whole episode', bootstrap_replicates=5000,
        caveat='Five same-session episodes and provisional annotations; CI does not account for annotation or session bias.')
    atomic_json(out/'test_report.json',report);atomic_json(out/'test_curves.json',curves)
    saved=torch.load(out/'selected.pt',weights_only=True)
    saved['test_report_sha256']=digest(out/'test_report.json')
    torch.save(saved,out/'selected.pt')
    print(json.dumps({k:{x:v.get(x) for x in ['n','macro_f1','accuracy']} for k,v in report['results'].items()},indent=2))

if __name__=='__main__':main()
