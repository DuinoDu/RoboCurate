"""Frozen comparisons of longer context and boundary-ambiguous supervision."""
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent;PREV=ROOT.parent/'stage_v2_20260917';APP=ROOT.parents[1]
sys.path[:0]=[str(APP),str(PREV)]
import train_cv as v2
from warp_progress.stages_v2 import FusionStageNet
from warp_progress.stages_v3 import MultiScaleStageNet,MotionStageNet,inference_v3
from warp_progress.stages import context_indices
from warp_progress.features import atomic_json
metrics=v2.metrics
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_data():
    p=json.loads((ROOT/'protocol.json').read_text())
    assert sha(ROOT/'protocol.json')==json.loads((ROOT/'freeze.json').read_text())['protocol_sha256']
    assert sha(PREV/'state_manifest.json')==p['state_manifest_sha256']
    old,a,m,rows,contract=v2.load_data()
    for k in ['folds','development','historical_regression','stress','seeds','annotation_sha256']:
        assert p[k]==old[k]
    for f,digest in p['reference_fold_sha256'].items():assert sha(PREV/'training'/f'fusion_tokens-fold{f}.pt')==digest
    return p,a,m,rows,contract

def make_model(name):
    if name.startswith('motion'):return MotionStageNet()
    return MultiScaleStageNet() if name.startswith('multiscale') else FusionStageNet(stage_tokens=True)

def edge_distances(row,mean,scale,qmean,qscale):
    x=(row['features']-mean)/scale;q=(row['state']-qmean)/qscale
    valid=row['camera_valid'].all(1)&row['state_valid']
    d=(np.square(np.diff(x,axis=0)).mean((1,2))*2+np.square(np.diff(q,axis=0)).mean(1))/3
    d[~(valid[:-1]&valid[1:])]=np.nan
    return d

def random_walk_targets(row,distances,edge_scale):
    """Exact hitting probabilities for a chain with clamped known-stage anchors.

    Between two anchors, cumulative edge resistance gives the absorption
    probability. Only previously unknown, valid nodes inside anchors are used.
    """
    y=row['y'];valid=row['camera_valid'].all(1)&row['state_valid']
    target=np.zeros((len(y),5),np.float32);weak=np.zeros(len(y),bool)
    anchors=np.flatnonzero((y>=0)&valid)
    resistance=np.exp(np.clip(distances/max(1e-8,edge_scale),0,3))
    for a,b in zip(anchors[:-1],anchors[1:]):
        if b<=a+1 or not valid[a:b+1].all():continue
        r=resistance[a:b]
        if not np.isfinite(r).all():continue
        ids=np.arange(a+1,b);fraction=np.cumsum(r)[:-1]/r.sum()
        target[ids,y[a]]+=1-fraction;target[ids,y[b]]+=fraction
        weak[ids]=True
    return target,weak

def arrays_for(rows,ids,context,model):
    vision=np.concatenate([rows[i]['features'][rows[i]['camera_valid'].all(1)] for i in ids])
    q=np.concatenate([rows[i]['state'][rows[i]['state_valid']] for i in ids])
    mean=vision.mean(0);scale=np.maximum(.1,vision.std(0));qm=q.mean(0);qs=np.maximum(.05,q.std(0))
    model.mean.copy_(torch.from_numpy(mean));model.scale.copy_(torch.from_numpy(scale))
    model.state_mean.copy_(torch.from_numpy(qm));model.state_scale.copy_(torch.from_numpy(qs))
    distances={i:edge_distances(rows[i],mean,scale,qm,qs) for i in ids}
    all_edges=np.concatenate(list(distances.values()));good=all_edges[np.isfinite(all_edges)&(all_edges>0)]
    edge_scale=float(np.median(good)) if len(good) else 1.
    strong=[[] for _ in range(6)];weak=[[] for _ in range(5)]
    for i in ids:
        r=rows[i];ix=context_indices(len(r['y']),context)
        keep=(r['y']>=0)&r['camera_valid'].all(1)&r['state_valid']
        target,wkeep=random_walk_targets(r,distances[i],edge_scale)
        for j,k in enumerate(['features','camera_valid','state','state_valid']):
            strong[j].append(r[k][ix[keep]]);weak[j].append(r[k][ix[wkeep]])
        strong[4].append(r['y'][keep]);strong[5].append(r['within'][keep]);weak[4].append(target[wkeep])
    return [torch.from_numpy(np.concatenate(a)) for a in strong],[torch.from_numpy(np.concatenate(a)) for a in weak],edge_scale

def train_one(name,seed,rows,ids,p):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=make_model(name)
    strong,weak,edge_scale=arrays_for(rows,ids,model.config['context'],model)
    y,w=strong[-2:];classes=[torch.where(y==k)[0].numpy() for k in range(5)]
    assert all(len(c) for c in classes)
    opt=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    snapshots=[];losses=[];uses_weak=name.endswith('_rw')
    if uses_weak and not len(weak[-1]):raise ValueError('No weak transition samples')
    for step in range(1,p['steps']+1):
        model.train();ix=np.array([rng.choice(classes[k]) for k in rng.integers(0,5,p['batch_size'])])
        inputs=[a[ix] for a in strong[:4]];n=len(ix)
        if uses_weak:
            wi=rng.integers(0,len(weak[-1]),p['weak_batch_size'])
            inputs=[torch.cat([a,b[wi]]) for a,b in zip(inputs,weak[:4])]
        logits,local=model(*inputs)
        loss=F.cross_entropy(logits[:n],y[ix])+.5*F.mse_loss(local[:n][torch.arange(n),y[ix]],w[ix])
        if uses_weak:loss+=p['weak_weight']*-(weak[-1][wi]*logits[n:].log_softmax(-1)).sum(-1).mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%50==0:losses.append(dict(step=step,loss=float(loss.detach())))
        if step in [200,250,300]:snapshots.append(copy.deepcopy(model.state_dict()))
    model.load_state_dict({k:torch.stack([s[k] for s in snapshots]).mean(0) for k in snapshots[0]})
    return model.eval(),dict(seed=seed,train_episodes=ids,strong_samples=len(y),weak_samples=len(weak[-1]) if uses_weak else 0,
        edge_scale=edge_scale,parameters=sum(x.numel() for x in model.parameters()),loss=losses)

class HistoryShuffled(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model;self.config=model.config
        n=model.config['context'];self.order=np.r_[np.random.default_rng(811).permutation(n-1),n-1]
    def forward(self,x,cv,q,qv):return self.model(x[:,self.order],cv[:,self.order],q[:,self.order],qv[:,self.order])

def infer(models,row,prior,temp=1.,boundary=False,control=None):
    x=row['features'].copy();cv=row['camera_valid'].copy();q=row['state'].copy();qv=row['state_valid'].copy()
    if control in ['head','wrist']:
        j=0 if control=='head' else 1;cv[:,j]=False;x[:,j]=np.nan
    if control=='state':qv[:]=False;q[:]=np.nan
    if control=='all_cameras':cv[:]=False;x[:]=np.nan
    if control=='history_shuffle':models=[HistoryShuffled(m).eval() for m in models]
    return inference_v3(models,x,cv,q,qv,prior,temp,boundary)

def evaluate(models,rows,ids,prior,temp=1.,boundary=False,control=None):
    ys=[];ps=[];eps=[];reviews=[];curves={};eligible=0;per_episode={}
    for i in ids:
        row=rows[i]
        if control=='mid_start':
            cut=len(row['features'])//2;keep=row['points_ix']>=cut
            row={**row,**{k:row[k][cut:] for k in ['features','camera_valid','state','state_valid','times','timestamp_ns']},
                 'points_ix':row['points_ix'][keep]-cut,'points_y':row['points_y'][keep]}
        out=infer(models,row,prior,temp,boundary,control)
        ix=row['points_ix'];y=row['points_y'];common=(y>=0)&row['camera_valid'].all(1)[ix]&row['state_valid'][ix]
        eligible+=int(common.sum());ok=common&out['valid'][ix];ix=ix[ok];y=y[ok];prob=out['probability'][ix]
        ys.extend(y);ps.extend(prob);eps.extend([i]*len(y));reviews.extend(out['review'][ix])
        if len(y):per_episode[str(i)]=metrics(y,prob)
        serial=lambda a:[float(v) if np.isfinite(v) else None for v in a]
        curves[str(i)]=dict(times_s=row['times'].tolist(),stage=out['stage'].tolist(),confidence=serial(out['confidence']),
            probability=[serial(a) for a in out['probability']],progress=serial(out['progress']),review=out['review'].tolist(),
            review_reasons=out['review_reasons'].tolist(),boundary_review=out['boundary_review'].tolist(),degraded=out['degraded'].tolist(),
            annotation_indices=ix.tolist(),annotation_stages=y.tolist(),
            switches=int(((out['stage'][1:]!=out['stage'][:-1])&out['valid'][1:]&out['valid'][:-1]).sum()))
    y=np.asarray(ys,int);prob=np.asarray(ps,np.float32).reshape(-1,5);review=np.asarray(reviews,bool)
    score=metrics(y,prob) if len(y) else dict(n=0,macro_f1=None,accuracy=None,nll=None)
    score.update(eligible_annotations=eligible,coverage=len(y)/max(1,eligible),per_episode=per_episode,
        review_fraction=float(review.mean()) if len(y) else None,
        unflagged_mistakes=int(((prob.argmax(-1)!=y)&~review).sum()) if len(y) else 0)
    return score,curves,dict(y=y,p=prob,episode=np.asarray(eps),review=review)

def load_fold(name,f,out):
    path=PREV/'training'/f'fusion_tokens-fold{f}.pt' if name=='v2_reference' else out/f'{name}-fold{f}.pt'
    saved=torch.load(path,map_location='cpu',weights_only=True);models=[]
    for state in saved['states']:
        model=make_model(name);model.load_state_dict(state);models.append(model.eval())
    return models,saved

def calibration(name,rows,p,out):
    members=[];ys=[]
    for f,held in enumerate(p['folds']):
        models,saved=load_fold(name,f,out);probs=[]
        for m in models:
            _,_,a=evaluate([m],rows,held,saved['priors']);probs.append(a['p'])
        members.append(np.stack(probs));ys.append(a['y'])
    mp=np.concatenate(members,axis=1);y=np.concatenate(ys)
    table=[dict(temperature=float(t),nll=metrics(y,np.mean([v2.temperature_prob(a,t) for a in mp],0))['nll']) for t in np.linspace(.5,3,26)]
    return min(table,key=lambda r:r['nll'])['temperature'],table

def policy_audit(name,rows,p,out,temp):
    result={}
    for boundary in [False,True]:
        ys=[];ps=[];review=[]
        for f,held in enumerate(p['folds']):
            models,saved=load_fold(name,f,out);_,_,a=evaluate(models,rows,held,saved['priors'],temp,boundary)
            ys.extend(a['y']);ps.extend(a['p']);review.extend(a['review'])
        y=np.asarray(ys);prob=np.asarray(ps);review=np.asarray(review)
        result[str(boundary)]=dict(n=len(y),review_fraction=float(review.mean()),
            unflagged_mistakes=int(((prob.argmax(-1)!=y)&~review).sum()),reviewed=int(review.sum()))
    a=result['False'];b=result['True']
    result['adopt_boundary']=b['unflagged_mistakes']<a['unflagged_mistakes'] and b['review_fraction']-a['review_fraction']<=.15
    return result

def main():
    torch.set_num_threads(2);p,annotations,manifest,rows,contract=load_data();start=time.monotonic()
    out=ROOT/'training';out.mkdir(exist_ok=True)
    if (out/'selection.json').exists():raise ValueError('Selection already fixed; no regression-driven retuning')
    reports={};all_curves={}
    for name in p['candidates']:
        folds=[];ys=[];ps=[];episodes=[];review=[];curves={}
        for f,held in enumerate(p['folds']):
            train=[i for i in p['development'] if i not in held];prior=v2.priors_for(rows,train);runs=[]
            if name=='v2_reference':models,saved=load_fold(name,f,out)
            else:
                models=[]
                for seed in p['seeds']:
                    model,run=train_one(name,seed+1000*f,rows,train,p);models.append(model);runs.append(run)
                torch.save(dict(config=models[0].config,states=[m.state_dict() for m in models],priors=prior.tolist()),out/f'{name}-fold{f}.pt')
            score,c,a=evaluate(models,rows,held,prior);curves.update(c)
            folds.append(dict(fold=f,held_out=held,training=train,runs=runs,metrics=score,priors=prior.tolist()))
            ys.extend(a['y']);ps.extend(a['p']);episodes.extend(a['episode']);review.extend(a['review'])
            print(name,f,'F1',round(score['macro_f1'],4),'accuracy',round(score['accuracy'],4),flush=True)
        y=np.asarray(ys);prob=np.asarray(ps);rv=np.asarray(review)
        overall=metrics(y,prob);overall.update(review_fraction=float(rv.mean()),unflagged_mistakes=int(((prob.argmax(-1)!=y)&~rv).sum()))
        reports[name]=dict(overall=overall,folds=folds);all_curves[name]=curves
        np.savez_compressed(out/f'{name}-oof.npz',y=y,p=prob,episode=np.asarray(episodes),review=rv)
        atomic_json(out/'cross_validation.json',reports)
    winner=max(reports,key=lambda n:(reports[n]['overall']['macro_f1'],-reports[n]['overall']['nll']))
    baseline=reports['v2_reference']['overall'];chosen=reports[winner]['overall']
    eligible=chosen['macro_f1']>=baseline['macro_f1']+.01 and chosen['nll']<=baseline['nll']+.03
    temp,table=calibration(winner,rows,p,out);policy=policy_audit(winner,rows,p,out,temp)
    selection=dict(winner=winner,eligible=bool(eligible),temperature=temp,calibration=table,review_policy=policy,
        macro_f1=chosen['macro_f1'],reference_macro_f1=baseline['macro_f1'],protocol_sha256=sha(ROOT/'protocol.json'))
    atomic_json(out/'selection.json',selection);atomic_json(out/'oof_curves.json',all_curves)
    models=[];runs=[];prior=v2.priors_for(rows,p['development'])
    for seed in p['seeds']:
        model,run=train_one(winner,seed,rows,p['development'],p);models.append(model);runs.append(run)
    boundary=policy['adopt_boundary']
    regression,curves,_=evaluate(models,rows,p['historical_regression'],prior,temp,boundary)
    controls={mode:evaluate(models,rows,p['historical_regression'],prior,temp,boundary,mode)[0]
        for mode in ['head','wrist','state','all_cameras','mid_start','history_shuffle']}
    _,stress,_=evaluate(models,rows,p['stress'],prior,temp,boundary)
    report=dict(selection=selection,historical_regression=regression,controls=controls,final_training=runs,
        elapsed_s=time.monotonic()-start,limitation=p['limitation'])
    atomic_json(out/'report.json',report);atomic_json(out/'regression_curves.json',curves);atomic_json(out/'stress_curves.json',stress)
    architecture='motion' if winner.startswith('motion') else 'multiscale' if winner.startswith('multiscale') else 'v2'
    config=dict(models[0].config,architecture=architecture)
    saved=dict(kind='stage_progress',version=3,config=config,states=[m.state_dict() for m in models],priors=prior.tolist(),temperature=temp,
        boundary_review=boundary,stage_names=annotations['stage_names'],cameras=['head','right_wrist'],state_contract=contract,
        feature_contracts=[manifest['episodes'][0]['cameras'][c]['contract'] for c in ['head','right_wrist']],
        task_id='g1_banana_left_table_to_right_basket',winner=winner,annotation_status=annotations['status'],
        annotation_sha256=p['annotation_sha256'],protocol_sha256=sha(ROOT/'protocol.json'),
        test_report_sha256=sha(out/'report.json'),promotion_eligible=bool(eligible))
    torch.save(saved,out/'selected.pt')
    print('WINNER',winner,'eligible',eligible,'OOF',chosen,'REGRESSION',regression,flush=True)

if __name__=='__main__':main()
