"""Preregistered contrastive and interval-boundary study on unchanged G1 inputs."""
import copy,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent;V3=ROOT.parent/'stage_v3_motion_20260917';V2=ROOT.parent/'stage_v2_20260917';APP=ROOT.parents[1]
sys.path[:0]=[str(APP),str(V2)]
import train_cv as v2
spec=importlib.util.spec_from_file_location('v3_shared_training',ROOT.parent/'stage_v3_20260917/train.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
from warp_progress.stages import context_indices
from warp_progress.stages_v2 import infer_fusion
from warp_progress.stages_v3 import MotionStageNet,inference_v3
from warp_progress.stages_v4 import RefinedStageNet,cross_recording_contrastive,boundary_bag_loss,segment_cdf_loss,inference_v4
from warp_progress.features import atomic_json
metrics=v2.metrics
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_data():
    p=json.loads((ROOT/'protocol.json').read_text())
    assert sha(ROOT/'protocol.json')==json.loads((ROOT/'freeze.json').read_text())['protocol_sha256']
    old,a,m,rows,contract=v2.load_data()
    for key in ['folds','development','historical_regression','stress','seeds','annotation_sha256','source_feature_manifest_sha256']:
        assert p[key]==old[key]
    assert sha(V2/'state_manifest.json')==p['state_manifest_sha256']
    for fold,digest in p['reference_fold_sha256'].items():assert sha(V3/'training'/f'motion_rw-fold{fold}.pt')==digest
    return p,a,m,rows,contract


def supervision_regions(row):
    y=row['y'];valid=row['camera_valid'].all(1)&row['state_valid'];known=np.flatnonzero((y>=0)&valid)
    bags=[];stable=[];segments=[]
    for a,b in zip(known[:-1],known[1:]):
        if y[a]!=y[b] and valid[a:b+1].all():bags.append(np.arange(a+1,b+1))
    for t in range(1,len(y)):
        if valid[t-1:t+1].all() and y[t]>=0 and y[t]==y[t-1]:stable.append(t)
    start=0
    while start<len(y):
        if not valid[start] or y[start]<0:start+=1;continue
        end=start+1
        while end<len(y) and valid[end] and y[end]==y[start]:end+=1
        # Leave a grid sample on each side outside the CDF interior.
        if end-start>=4:segments.append((np.arange(start+1,end-1),int(y[start])))
        start=end
    return bags,np.asarray(stable,int),segments


def extra_arrays(rows,ids,context):
    full=[[] for _ in range(4)];recordings=[];bags=[];stable=[];segments=[];offset=0
    for ep in ids:
        row=rows[ep];ix=context_indices(len(row['y']),context)
        for j,key in enumerate(['features','camera_valid','state','state_valid']):full[j].append(row[key][ix])
        keep=(row['y']>=0)&row['camera_valid'].all(1)&row['state_valid']
        recordings.extend([ep]*int(keep.sum()))
        b,s,g=supervision_regions(row);bags.extend([x+offset for x in b]);stable.extend((s+offset).tolist())
        segments.extend([(x+offset,y) for x,y in g]);offset+=len(row['y'])
    return [torch.from_numpy(np.concatenate(a)) for a in full],torch.tensor(recordings),bags,np.asarray(stable),segments


def train_one(name,seed,rows,ids,p):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);extra_rng=np.random.default_rng(seed+9181)
    model=RefinedStageNet(contrastive=name in ['contrastive','combined'],boundary=name in ['boundary_structure','combined'])
    strong,weak,edge_scale=shared.arrays_for(rows,ids,8,model)
    full,recordings,bags,stable,segments=extra_arrays(rows,ids,8)
    assert len(recordings)==len(strong[-2]) and bags and len(stable) and segments
    y,within=strong[-2:];classes=[torch.where(y==k)[0].numpy() for k in range(5)];assert all(len(c) for c in classes)
    opt=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);snapshots=[];losses=[];total_examples=0
    for step in range(1,p['steps']+1):
        model.train();ix=np.asarray([rng.choice(classes[k]) for k in rng.integers(0,5,p['batch_size'])]);n=len(ix)
        wi=rng.integers(0,len(weak[-1]),p['weak_batch_size']);inputs=[torch.cat([a[ix],b[wi]]) for a,b in zip(strong[:4],weak[:4])]
        selected=[];bag_slices=[];segment_slices=[];offset=n+len(wi);negative_slice=None
        if model.boundary_head is not None:
            cfg=p['structure']
            for b in extra_rng.integers(0,len(bags),cfg['positive_bags']):
                points=bags[b];selected.extend(points);bag_slices.append(slice(offset,offset+len(points)));offset+=len(points)
            points=extra_rng.choice(stable,cfg['stable_samples']);selected.extend(points)
            negative_slice=slice(offset,offset+len(points));offset+=len(points)
            if step>cfg['warmup_steps']:
                for g in extra_rng.integers(0,len(segments),cfg['segments']):
                    points,label=segments[g]
                    if len(points)>cfg['max_segment_samples']:
                        start=extra_rng.integers(0,len(points)-cfg['max_segment_samples']+1);points=points[start:start+cfg['max_segment_samples']]
                    selected.extend(points);segment_slices.append((slice(offset,offset+len(points)),label));offset+=len(points)
            inputs=[torch.cat([a,b[np.asarray(selected,int)]]) for a,b in zip(inputs,full)]
        logits,local,z,boundary=model(*inputs,return_aux=True)
        hard=F.cross_entropy(logits[:n],y[ix])+.5*F.mse_loss(local[:n][torch.arange(n),y[ix]],within[ix])
        soft=-(weak[-1][wi]*logits[n:n+len(wi)].log_softmax(-1)).sum(-1).mean()
        loss=hard+p['weak_weight']*soft;parts={}
        if model.projection is not None:
            con=cross_recording_contrastive(z[:n],y[ix],recordings[ix],p['contrastive']['temperature'])
            loss+=p['contrastive']['weight']*con;parts['contrastive']=float(con.detach())
        if model.boundary_head is not None:
            bl=boundary_bag_loss([boundary[s] for s in bag_slices],boundary[negative_slice])
            loss+=p['structure']['boundary_weight']*bl;parts['boundary']=float(bl.detach())
            if segment_slices:
                seg=torch.stack([segment_cdf_loss(logits[s].softmax(-1)[:,k]) for s,k in segment_slices]).mean()
                loss+=p['structure']['segment_weight']*seg;parts['segment_cdf']=float(seg.detach())
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total_examples+=len(logits)
        if step%50==0:losses.append(dict(step=step,loss=float(loss.detach()),**parts))
        if step in [200,250,300]:snapshots.append(copy.deepcopy(model.state_dict()))
    model.load_state_dict({k:torch.stack([s[k] for s in snapshots]).mean(0) for k in snapshots[0]})
    return model.eval(),dict(seed=seed,train_episodes=ids,strong_samples=len(y),weak_samples=len(weak[-1]),boundary_brackets=len(bags),
        stable_boundary_samples=len(stable),constant_stage_interiors=len(segments),edge_scale=edge_scale,processed_examples=total_examples,
        parameters=sum(x.numel() for x in model.parameters()),loss=losses)


def infer(models,row,priors,temp=1.,review_boundary=False):
    args=(models,row['features'],row['camera_valid'],row['state'],row['state_valid'],priors,temp)
    if isinstance(models[0],RefinedStageNet):return inference_v4(*args,review_boundary)
    out=inference_v3(*args,False,True)
    out.update(boundary_probability=np.full(len(row['features']),np.nan,np.float32),learned_boundary_review=np.zeros(len(row['features']),bool))
    return out


def bracket_metrics(row,out):
    ix=row['points_ix'];y=row['points_y'];ok=(y>=0)&row['camera_valid'].all(1)[ix]&row['state_valid'][ix]
    ix=ix[ok];y=y[ok]
    if len(ix)<2:return dict(brackets=0,matched=0,extra_switches=0)
    stage=out['stage'];valid=out['valid'];changes=np.flatnonzero(valid[1:]&valid[:-1]&(stage[1:]!=stage[:-1]))+1
    changes=changes[(changes>ix[0])&(changes<=ix[-1])];used=set();total=0;matched=0
    for a,b,la,lb in zip(ix[:-1],ix[1:],y[:-1],y[1:]):
        if la==lb or b<=a or not valid[a:b+1].all():continue
        total+=1
        candidates=[int(t) for t in changes if a<t<=b and stage[t-1]==la and stage[t]==lb and t not in used]
        if candidates:used.add(candidates[0]);matched+=1
    return dict(brackets=total,matched=matched,extra_switches=len(changes)-len(used))


def evaluate(models,rows,ids,priors,temp=1.,review_boundary=False,control=None):
    ys=[];ps=[];episodes=[];reviews=[];curves={};diag=dict(brackets=0,matched=0,extra_switches=0);eligible=0
    for ep in ids:
        original=rows[ep];row={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in original.items()}
        common=(row['points_y']>=0)&row['camera_valid'].all(1)[row['points_ix']]&row['state_valid'][row['points_ix']]
        if control in ['head','wrist']:
            j=0 if control=='head' else 1;row['camera_valid'][:,j]=False;row['features'][:,j]=np.nan
        if control=='state':row['state_valid'][:]=False;row['state'][:]=np.nan
        if control=='all_cameras':row['camera_valid'][:]=False;row['features'][:]=np.nan
        if control=='mid_start':
            cut=len(row['features'])//2;keep=row['points_ix']>=cut;common=common[keep]
            row={**row,**{k:row[k][cut:] for k in ['features','camera_valid','state','state_valid','times','timestamp_ns']},
                 'points_ix':row['points_ix'][keep]-cut,'points_y':row['points_y'][keep]}
        out=infer(models,row,priors,temp,review_boundary);ix=row['points_ix'];ok=common&out['valid'][ix];eligible+=int(common.sum())
        ix=ix[ok];y=row['points_y'][ok];prob=out['probability'][ix]
        ys.extend(y);ps.extend(prob);episodes.extend([ep]*len(y));reviews.extend(out['review'][ix])
        d=bracket_metrics(row,out)
        for k in diag:diag[k]+=d[k]
        serial=lambda a:[float(v) if np.isfinite(v) else None for v in a]
        curves[str(ep)]=dict(times_s=row['times'].tolist(),stage=out['stage'].tolist(),confidence=serial(out['confidence']),
            probability=[serial(a) for a in out['probability']],progress=serial(out['progress']),review=out['review'].tolist(),
            review_reasons=out['review_reasons'].tolist(),boundary_probability=serial(out['boundary_probability']),
            annotation_indices=ix.tolist(),annotation_stages=y.tolist(),bracket_diagnostic=d)
    y=np.asarray(ys,int);prob=np.asarray(ps,np.float32).reshape(-1,5);rv=np.asarray(reviews,bool)
    score=metrics(y,prob) if len(y) else dict(n=0,macro_f1=None,accuracy=None,nll=None)
    score.update(eligible_annotations=eligible,coverage=len(y)/max(1,eligible),reviewed=int(rv.sum()),review_fraction=float(rv.mean()) if len(rv) else None,
        unflagged_mistakes=int(((prob.argmax(-1)!=y)&~rv).sum()) if len(y) else 0,bracket_diagnostic=diag)
    return score,curves,dict(y=y,p=prob,episode=np.asarray(episodes),review=rv)


def load_fold(name,fold,out):
    path=V3/'training'/f'motion_rw-fold{fold}.pt' if name=='v3_reference' else out/f'{name}-fold{fold}.pt'
    saved=torch.load(path,map_location='cpu',weights_only=True);models=[]
    for state in saved['states']:
        if name=='v3_reference':model=MotionStageNet()
        else:
            config=dict(saved['config']);config.pop('use_state',None);model=RefinedStageNet(**config)
        model.load_state_dict(state,strict=True);models.append(model.eval())
    return models,saved


def calibrate(name,rows,p,out):
    members=[];ys=[]
    for f,held in enumerate(p['folds']):
        models,saved=load_fold(name,f,out);fold=[]
        for m in models:
            _,_,a=evaluate([m],rows,held,saved['priors']);fold.append(a['p'])
        members.append(np.stack(fold));ys.append(a['y'])
    mp=np.concatenate(members,axis=1);y=np.concatenate(ys)
    table=[dict(temperature=float(t),**metrics(y,np.mean([v2.temperature_prob(m,t) for m in mp],0))) for t in np.linspace(.5,3.,26)]
    return min(table,key=lambda r:r['nll'])['temperature'],table


def review_audit(name,rows,p,out,temp):
    result={}
    for flag in [False,True]:
        if flag and name not in ['boundary_structure','combined']:continue
        errors=reviewed=unflagged=n=0
        for f,held in enumerate(p['folds']):
            models,saved=load_fold(name,f,out);s,_,_=evaluate(models,rows,held,saved['priors'],temp,flag)
            n+=s['n'];errors+=round(s['n']*(1-s['accuracy']));reviewed+=s['reviewed'];unflagged+=s['unflagged_mistakes']
        result[str(flag)]=dict(n=n,errors=errors,reviewed=reviewed,review_fraction=reviewed/n,unflagged_mistakes=unflagged)
    a=result['False'];b=result.get('True',a)
    result['enabled']=bool(b['unflagged_mistakes']<a['unflagged_mistakes'] and b['review_fraction']-a['review_fraction']<=.10)
    return result


def main():
    torch.set_num_threads(2);p,annotations,manifest,rows,contract=load_data();started=time.monotonic()
    out=ROOT/'training';out.mkdir(exist_ok=True)
    if (out/'selection.json').exists():raise ValueError('Selection frozen; no historical-driven retuning')
    reports={};all_curves={}
    for name in p['candidates']:
        folds=[];ys=[];ps=[];eps=[];reviews=[];curves={};diag=dict(brackets=0,matched=0,extra_switches=0)
        for f,held in enumerate(p['folds']):
            ids=[i for i in p['development'] if i not in held];prior=v2.priors_for(rows,ids);runs=[]
            if name=='v3_reference':models,_=load_fold(name,f,out)
            else:
                models=[]
                for seed in p['seeds']:
                    model,run=train_one(name,seed+1000*f,rows,ids,p);models.append(model);runs.append(run)
                torch.save(dict(config=models[0].config,states=[m.state_dict() for m in models],priors=prior.tolist()),out/f'{name}-fold{f}.pt')
            score,c,a=evaluate(models,rows,held,prior);curves.update(c)
            folds.append(dict(fold=f,training=ids,held_out=held,metrics=score,runs=runs))
            for k in diag:diag[k]+=score['bracket_diagnostic'][k]
            ys.extend(a['y']);ps.extend(a['p']);eps.extend(a['episode']);reviews.extend(a['review'])
            print(name,'fold',f,'F1',round(score['macro_f1'],4),'brackets',score['bracket_diagnostic'],flush=True)
        y=np.asarray(ys);prob=np.asarray(ps);rv=np.asarray(reviews);overall=metrics(y,prob)
        overall.update(bracket_diagnostic=diag,review_fraction=float(rv.mean()),unflagged_mistakes=int(((prob.argmax(-1)!=y)&~rv).sum()))
        reports[name]=dict(overall=overall,folds=folds);all_curves[name]=curves
        np.savez_compressed(out/f'{name}-oof.npz',y=y,p=prob,episode=np.asarray(eps),review=rv)
        atomic_json(out/'cross_validation.json',reports)
    base=reports['v3_reference']['overall'];eligibility={}
    for name,r in reports.items():
        s=r['overall'];d=s['bracket_diagnostic'];b=base['bracket_diagnostic']
        stable=s['macro_f1']>=base['macro_f1']-1e-10 and s['nll']<=base['nll']+.02
        improved=s['macro_f1']>=base['macro_f1']+.005 or s['nll']<=base['nll']-.01 or (d['matched']>=b['matched']+2 and d['extra_switches']<=b['extra_switches'])
        eligibility[name]=bool(name!='v3_reference' and stable and improved)
    eligible=[n for n,e in eligibility.items() if e]
    winner=max(eligible or reports,key=lambda n:(reports[n]['overall']['macro_f1'],-reports[n]['overall']['nll']))
    temp,calibration=calibrate(winner,rows,p,out);policy=review_audit(winner,rows,p,out,temp)
    selection=dict(winner=winner,eligible=bool(eligibility[winner]),eligibility=eligibility,temperature=temp,calibration=calibration,
        review_policy=policy,reference=base,chosen=reports[winner]['overall'],protocol_sha256=sha(ROOT/'protocol.json'))
    atomic_json(out/'selection.json',selection);atomic_json(out/'oof_curves.json',all_curves)
    if winner=='v3_reference':
        atomic_json(out/'report.json',dict(selection=selection,elapsed_s=time.monotonic()-started,limitation=p['limitation'],reason='No new candidate beat v3; reference retained'))
        print('NO PROMOTION',flush=True);return
    models=[];runs=[];prior=v2.priors_for(rows,p['development'])
    for seed in p['seeds']:
        m,run=train_one(winner,seed,rows,p['development'],p);models.append(m);runs.append(run)
    flag=policy['enabled'];reg,curves,_=evaluate(models,rows,p['historical_regression'],prior,temp,flag)
    controls={c:evaluate(models,rows,p['historical_regression'],prior,temp,flag,c)[0] for c in ['head','wrist','state','all_cameras','mid_start']}
    _,stress,_=evaluate(models,rows,p['stress'],prior,temp,flag)
    report=dict(selection=selection,historical_regression=reg,controls=controls,final_training=runs,elapsed_s=time.monotonic()-started,limitation=p['limitation'])
    atomic_json(out/'report.json',report);atomic_json(out/'regression_curves.json',curves);atomic_json(out/'stress_curves.json',stress)
    saved=dict(kind='stage_progress',version=4,config=models[0].config,states=[m.state_dict() for m in models],priors=prior.tolist(),temperature=temp,
        review_guard=True,review_boundary=flag,stage_names=annotations['stage_names'],cameras=['head','right_wrist'],state_contract=contract,
        feature_contracts=[manifest['episodes'][0]['cameras'][c]['contract'] for c in ['head','right_wrist']],
        task_id='g1_banana_left_table_to_right_basket',winner=winner,annotation_status=annotations['status'],annotation_sha256=p['annotation_sha256'],
        protocol_sha256=sha(ROOT/'protocol.json'),test_report_sha256=sha(out/'report.json'),promotion_eligible=bool(eligibility[winner]))
    torch.save(saved,out/'selected.pt')
    print('WINNER',winner,'eligible',eligibility[winner],'F1',reports[winner]['overall']['macro_f1'],'REGRESSION',reg,flush=True)
if __name__=='__main__':main()
