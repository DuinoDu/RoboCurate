"""SARM2-inspired gradient isolation using the frozen v4 experiment engine."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;FIRST=ROOT.parent/'stage_v4_20260917'
spec=importlib.util.spec_from_file_location('v4_engine',FIRST/'train.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
from warp_progress.stages_v4 import RefinedStageNet
from warp_progress.features import atomic_json

class DetachedProgressNet(RefinedStageNet):
    def __init__(self,**kwargs):
        super().__init__(**dict(kwargs,detach_progress=True))

def audit_proxy():
    p,_,_,rows,_=engine.load_data()
    curves=json.loads((ROOT/'training/oof_curves.json').read_text());scores={}
    for name,outputs in curves.items():
        errors=[]
        for held in p['folds']:
            ids=[i for i in p['development'] if i not in held]
            prior=np.asarray(engine.v2.priors_for(rows,ids),np.float32);offset=np.r_[0.,np.cumsum(prior)[:-1]]
            for ep in held:
                row=rows[ep];out=outputs[str(ep)];ix=np.asarray(out['annotation_indices']);y=np.asarray(out['annotation_stages'])
                target=offset[y]+prior[y]*row['within'][ix]
                predicted=np.asarray(out['progress'],float)[ix];errors.extend(predicted-target)
        scores[name]=dict(n=len(errors),mae=float(np.abs(errors).mean()),mse=float(np.square(errors).mean()))
    eligible=scores['detached_progress']['mae']<=scores['v3_reference']['mae']+.01
    atomic_json(ROOT/'training/progress_proxy_audit.json',dict(scores=scores,eligible=bool(eligible),
        gate='New MAE <= v3 MAE +0.01 at T=1 on the same OOF sparse points',
        limitation='Inherited stage-time interpolation proxy, not independent measured task progress or success.'))

def main():
    engine.ROOT=ROOT;engine.RefinedStageNet=DetachedProgressNet
    engine.main();audit_proxy()
if __name__=='__main__':main()
