"""Five-fold development experiment. Historical test is regression, never a new blind test."""
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent;PREV=ROOT.parent/'model_upgrade_20260917';APP=ROOT.parents[1]
sys.path[:0]=[str(APP),str(PREV)]
from train_stages import read_study,metrics
from warp_progress.stages import StageNet,context_indices,infer_stage_models
from warp_progress.stages_v2 import FusionStageNet,infer_fusion
from warp_progress.features import atomic_json

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_data():
    frozen=json.loads((ROOT/'freeze.json').read_text())
    assert sha(ROOT/'protocol.json')==frozen['protocol_sha256']
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert sha(PREV/'annotations.v1.json')==protocol['annotation_sha256']
    assert sha(PREV/'features_manifest.json')==protocol['source_feature_manifest_sha256']
    _,annotations,manifest,rows,_=read_study()
    states=json.loads((ROOT/'state_manifest.json').read_text())
    for record in states['episodes']:
        p=Path(record['path'])
        if not p.exists():p=ROOT/'states'/p.name
        assert sha(p)==record['sha256']
        with np.load(p,allow_pickle=False) as z:
            row=rows[int(record['episode'][-6:])]
            row['state']=z['state'];row['state_valid']=z['state_valid']
            row['timestamp_ns']=z['timestamp_ns']
    assert sorted(sum(protocol['folds'],[]))==sorted(protocol['development'])
    assert not set(protocol['development'])&set(protocol['historical_regression']+protocol['stress'])
    return protocol,annotations,manifest,rows,states['contract']

def priors_for(rows,ids):
    return np.r_[np.mean([np.diff(rows[i]['bounds'])/rows[i]['bounds'][-1] for i in ids],0),0.].astype(np.float32)

def make_model(name):
    if name.startswith('baseline'):
        return StageNet(views=2,temporal=name=='baseline_temporal')
    return FusionStageNet(temporal=name!='fusion_mlp',stage_tokens='tokens' in name)

def forward(model,x,cv,q,qv):
    return model(x,cv,q,qv) if isinstance(model,FusionStageNet) else model(x,cv.all(-1))

def infer(models,row,priors,temperature=1.,drop=None):
    x=row['features'].copy();cv=row['camera_valid'].copy();q=row['state'].copy();qv=row['state_valid'].copy()
    if drop in ['head','wrist']:
        j=0 if drop=='head' else 1;cv[:,j]=False;x[:,j]=np.nan
    if drop=='state':qv[:]=False;q[:]=np.nan
    if drop=='all_cameras':cv[:]=False;x[:]=np.nan
    if isinstance(models[0],FusionStageNet):return infer_fusion(models,x,cv,q,qv,priors,temperature)
    return infer_stage_models(models,x,cv.all(1),priors,temperature)

def evaluate(models,rows,ids,priors,temperature=1.,drop=None):
    ys=[];ps=[];episodes=[];reviews=[];total=0;curves={};by_episode={};switches=[]
    for i in ids:
        row=rows[i];out=infer(models,row,priors,temperature,drop)
        ix=row['points_ix'];y=row['points_y']
        common=(y>=0)&row['camera_valid'].all(1)[ix]&row['state_valid'][ix]
        total+=int(common.sum());ok=common&out['valid'][ix]
        y=y[ok];ix=ix[ok];p=out['probability'][ix]
        ys.extend(y);ps.extend(p);episodes.extend([i]*len(y));reviews.extend(out['review'][ix])
        if len(y):by_episode[str(i)]=metrics(y,p)
        stages=out['stage'];contiguous=out['valid'][1:]&out['valid'][:-1]
        changes=int(((stages[1:]!=stages[:-1])&contiguous).sum());switches.append(changes)
        curves[str(i)]=dict(times_s=row['times'].tolist(),stage=stages.tolist(),
            probability=[[float(x) if np.isfinite(x) else None for x in p] for p in out['probability']],
            confidence=[float(v) if np.isfinite(v) else None for v in out['confidence']],
            progress=[float(v) if np.isfinite(v) else None for v in out['progress']],review=out['review'].tolist(),
            annotation_indices=ix.tolist(),annotation_stages=y.tolist(),switches=changes)
    y=np.array(ys,dtype=int);p=np.array(ps,dtype=np.float32).reshape(-1,5)
    result=metrics(y,p) if len(y) else dict(n=0,accuracy=None,macro_f1=None,nll=None)
    result.update(eligible_annotations=total,coverage=len(y)/max(1,total),per_episode=by_episode,
        review_fraction=float(np.mean(reviews)) if reviews else None,mean_switches=float(np.mean(switches)))
    return result,curves,dict(y=y,p=p,episode=np.array(episodes,dtype=int))

def build_arrays(rows,ids):
    output=[[] for _ in range(10)]
    for i in ids:
        r=rows[i];keep=(r['y']>=0)&r['camera_valid'].all(1)&r['state_valid']
        for offset,stride in [(0,1),(4,2)]:
            ix=np.maximum(0,np.arange(len(keep))[:,None]-stride*np.arange(7,-1,-1))
            for j,k in enumerate(['features','camera_valid','state','state_valid']):output[offset+j].append(r[k][ix[keep]])
        output[8].append(r['y'][keep]);output[9].append(r['within'][keep])
    return [torch.from_numpy(np.concatenate(a)) for a in output]

def train_one(name,seed,rows,ids,protocol):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=make_model(name)
    vision=np.concatenate([rows[i]['features'][rows[i]['camera_valid'].all(1)] for i in ids])
    model.mean.copy_(torch.from_numpy(vision.mean(0)));model.scale.copy_(torch.from_numpy(np.maximum(.1,vision.std(0))))
    if isinstance(model,FusionStageNet):
        q=np.concatenate([rows[i]['state'][rows[i]['state_valid']] for i in ids])
        model.state_mean.copy_(torch.from_numpy(q.mean(0)));model.state_scale.copy_(torch.from_numpy(np.maximum(.05,q.std(0))))
    arrays=build_arrays(rows,ids);y,w=arrays[-2:]
    by_class=[torch.where(y==k)[0].numpy() for k in range(5)]
    assert all(len(c)>0 for c in by_class)
    opt=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
    snapshots=[];losses=[]
    for step in range(1,protocol['steps']+1):
        model.train()
        ix=np.array([rng.choice(by_class[k]) for k in rng.integers(0,5,protocol['batch_size'])])
        x,cv,q,qv=[a[ix].clone() for a in arrays[:4]]
        if name.endswith('_aug'):
            wide=torch.from_numpy(rng.random(len(ix))<.5)
            for a,b in zip([x,cv,q,qv],arrays[4:8]):a[wide]=b[ix][wide]
            missing=rng.random(len(ix))<.2;which=rng.integers(0,2,len(ix))
            for j in range(2):cv[torch.from_numpy(missing&(which==j)),:,j]=False
            qv[torch.from_numpy(rng.random(len(ix))<.15)]=False
        logits,local=forward(model,x,cv,q,qv)
        loss=F.cross_entropy(logits,y[ix])+.5*F.mse_loss(local[torch.arange(len(ix)),y[ix]],w[ix])
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%50==0:losses.append(dict(step=step,loss=float(loss.detach())))
        if step in [200,250,300]:snapshots.append(copy.deepcopy(model.state_dict()))
    average={k:torch.stack([s[k] for s in snapshots]).mean(0) for k in snapshots[0]}
    model.load_state_dict(average);model.eval()
    return model,dict(seed=seed,train_episodes=ids,train_samples=len(y),parameters=sum(p.numel() for p in model.parameters()),loss=losses)

def temperature_prob(p,t):
    log=np.log(np.maximum(np.finfo(np.float32).tiny,p))/t;log-=log.max(1,keepdims=True)
    e=np.exp(log);return e/e.sum(1,keepdims=True)

def calibrate_oof(name,protocol,rows,out):
    """Temperature scales each member before ensemble averaging, as in runtime."""
    ys=[];members=[]
    for f,held in enumerate(protocol['folds']):
        saved=torch.load(out/f'{name}-fold{f}.pt',map_location='cpu',weights_only=True)
        member=[]
        for state in saved['states']:
            model=make_model(name);model.load_state_dict(state);model.eval()
            _,_,a=evaluate([model],rows,held,saved['priors'])
            member.append(a['p'])
        ys.append(a['y']);members.append(np.stack(member))
    y=np.concatenate(ys);p=np.concatenate(members,axis=1)
    return [dict(temperature=float(t),nll=metrics(y,np.mean([temperature_prob(m,t) for m in p],axis=0))['nll'])
            for t in np.linspace(.5,3,26)]

def main():
    torch.set_num_threads(2)
    protocol,annotations,manifest,rows,contract=load_data();started=time.monotonic()
    out=ROOT/'training';out.mkdir(exist_ok=True)
    if (out/'selection.json').exists():raise ValueError('Selection already finalized. Do not tune using historical regression.')
    reports={};oof={};all_curves={}
    for name in protocol['candidates']:
        folds=[];ys=[];ps=[];es=[];candidate_curves={}
        for f,held in enumerate(protocol['folds']):
            train=[i for i in protocol['development'] if i not in held];prior=priors_for(rows,train)
            models=[];runs=[]
            for seed in protocol['seeds']:
                model,run=train_one(name,seed+1000*f,rows,train,protocol);models.append(model);runs.append(run)
            score,curves,arrays=evaluate(models,rows,held,prior)
            folds.append(dict(fold=f,held_out=held,training=train,metrics=score,runs=runs,priors=prior.tolist()))
            ys.append(arrays['y']);ps.append(arrays['p']);es.append(arrays['episode']);candidate_curves.update(curves)
            torch.save(dict(config=models[0].config,states=[m.state_dict() for m in models],priors=prior.tolist()),out/f'{name}-fold{f}.pt')
            print(name,'fold',f,'F1',round(score['macro_f1'],4),'accuracy',round(score['accuracy'],4),flush=True)
        y=np.concatenate(ys);p=np.concatenate(ps);e=np.concatenate(es)
        overall=metrics(y,p)
        overall['mean_switches']=float(np.mean([c['switches'] for c in candidate_curves.values()]))
        reports[name]=dict(overall=overall,folds=folds)
        oof[name]=dict(y=y,p=p,episode=e);all_curves[name]=candidate_curves
        np.savez_compressed(out/(name+'-oof.npz'),**oof[name])
        atomic_json(out/'cross_validation.json',reports)
    winner=max(reports,key=lambda n:(reports[n]['overall']['macro_f1'],-reports[n]['overall']['nll']))
    wf1=reports[winner]['overall']['macro_f1'];base=reports['baseline_mlp']['overall']['macro_f1'];temporal=reports['baseline_temporal']['overall']['macro_f1']
    eligible=winner.startswith('fusion') and wf1>=base+.02 and wf1>=temporal-.01
    cal=calibrate_oof(winner,protocol,rows,out)
    temperature=min(cal,key=lambda r:r['nll'])['temperature']
    selection=dict(winner=winner,eligible=bool(eligible),basis='Five-fold development out-of-fold results; no independent new test',
        macro_f1=wf1,baseline_mlp_f1=base,baseline_temporal_f1=temporal,temperature=temperature,
        calibration=cal,protocol_sha256=sha(ROOT/'protocol.json'),elapsed_cv_s=time.monotonic()-started)
    atomic_json(out/'selection.json',selection);atomic_json(out/'oof_curves.json',all_curves)
    models=[];runs=[];prior=priors_for(rows,protocol['development'])
    for seed in protocol['seeds']:
        model,run=train_one(winner,seed,rows,protocol['development'],protocol);models.append(model);runs.append(run)
    # Historical regression is opened only after development selection is fixed.
    regression,curves,_=evaluate(models,rows,protocol['historical_regression'],prior,temperature)
    controls={}
    for drop in ['head','wrist','state','all_cameras']:
        controls[drop],_,_=evaluate(models,rows,protocol['historical_regression'],prior,temperature,drop)
    stress={}
    for i in protocol['stress']:
        _,c,_=evaluate(models,rows,[i],prior,temperature);stress.update(c)
    report=dict(selection=selection,historical_regression=regression,controlled_missing_inputs=controls,
        final_training=runs,elapsed_s=time.monotonic()-started,
        limitation='Development CV and previously seen regression clips, provisional labels, one session. No independent generalization or final success claim.')
    atomic_json(out/'report.json',report);atomic_json(out/'regression_curves.json',curves);atomic_json(out/'stress_curves.json',stress)
    checkpoint=dict(kind='stage_progress',version=2 if winner.startswith('fusion') else 1,config=models[0].config,
        states=[m.state_dict() for m in models],priors=prior.tolist(),temperature=temperature,
        stage_names=annotations['stage_names'],cameras=['head','right_wrist'],state_contract=contract,
        feature_contracts=[manifest['episodes'][0]['cameras'][c]['contract'] for c in ['head','right_wrist']],
        task_id='g1_banana_left_table_to_right_basket',winner=winner,annotation_status=annotations['status'],
        annotation_sha256=protocol['annotation_sha256'],protocol_sha256=sha(ROOT/'protocol.json'),
        test_report_sha256=sha(out/'report.json'),promotion_eligible=bool(eligible))
    torch.save(checkpoint,out/'selected.pt')
    print('WINNER',winner,'eligible',eligible,'OOF F1',wf1,'HISTORICAL',regression,flush=True)

if __name__=='__main__':main()
