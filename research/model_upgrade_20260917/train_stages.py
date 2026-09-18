"""Frozen-protocol baseline/ablation training; test is accessed only after selection."""
import copy
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parent
APP = ROOT.parents[1]
sys.path.insert(0, str(APP))
from warp_progress.stages import StageNet, infer_stage_models, context_indices
from warp_progress.features import atomic_json

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_study():
    freeze = json.loads((ROOT/'freeze.json').read_text())
    for filename, key in [('annotations.v1.json','annotation_sha256'), ('protocol.v1.json','protocol_sha256')]:
        if digest(ROOT/filename) != freeze[key]:
            raise ValueError('Frozen protocol or labels changed')
    protocol = json.loads((ROOT/'protocol.v1.json').read_text())
    annotations = json.loads((ROOT/'annotations.v1.json').read_text())
    manifest = json.loads((ROOT/'features_manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError('Feature extraction incomplete')
    rows = {}
    for a,m in zip(annotations['episodes'], manifest['episodes']):
        assert a['episode'] == m['episode']
        cams = []
        for name in ['head','right_wrist']:
            cache=Path(m['cameras'][name]['cache'])
            if not cache.is_file():
                cache=APP/'workspace/warp/features'/cache.name
            expected=m['cameras'][name].get('cache_sha256')
            if expected and digest(cache)!=expected:
                raise ValueError('Cached visual feature checksum mismatch')
            with np.load(cache, allow_pickle=False) as f:
                cams.append({k:f[k] for k in f.files})
        assert np.array_equal(cams[0]['timestamp_ns'], cams[1]['timestamp_ns'])
        start_ns=m.get('start_ns')
        if start_ns is None:
            start_ns=json.loads(Path(m['meta']).read_text())['episode']['start_ns']
        t = (cams[0]['timestamp_ns'] - start_ns) / 1e9
        features = np.stack([c['features'] for c in cams], axis=1)
        point_t = np.array([p['time_s'] for p in a['points']])
        point_y = np.array([p['stage'] for p in a['points']])
        point_ix = np.abs(t[:,None]-point_t[None]).argmin(0)
        y = np.full(len(t), -1, dtype=np.int64)
        for j in range(len(point_t)-1):
            if point_y[j] >= 0 and point_y[j] == point_y[j+1]:
                y[(t >= point_t[j]) & (t <= point_t[j+1])] = point_y[j]
        for idx, pt, label in zip(point_ix, point_t, point_y):
            if label >= 0 and abs(t[idx]-pt) <= .26:
                y[idx] = label
        # Label transition bounds are uncertain; midpoint used ONLY to form a progress proxy.
        bounds = [0.]
        for phase in range(1,5):
            before = point_t[(point_y >= 0)&(point_y < phase)]
            after = point_t[point_y == phase]
            bounds.append(float((before[-1]+after[0])/2) if len(before) and len(after) else float(t[-1]))
        bounds = np.maximum.accumulate(bounds)
        within = np.zeros(len(t), np.float32)
        for phase in range(4):
            ok = y == phase
            within[ok] = np.clip((t[ok]-bounds[phase])/max(.1,bounds[phase+1]-bounds[phase]),0,1)
        within[y == 4] = 1.
        rows[int(a['episode'][-6:])] = dict(features=features, camera_valid=np.stack([c['valid'] for c in cams],1),
            times=t, y=y, within=within, bounds=bounds.tolist(), points_y=point_y, points_ix=point_ix,
            source_sha256=m['source_sha256'])
    # Whole recording leakage guard, including duplicate content under different paths.
    groups = [protocol[k] for k in ['train','validation','test','stress']]
    hashes = [[rows[i]['source_sha256'] for i in g] for g in groups]
    flat = sum(hashes,[])
    if len(set(flat)) != len(flat):
        raise ValueError('Source-content overlap across study episodes')
    prior = np.mean([np.diff(rows[i]['bounds'])/rows[i]['bounds'][-1] for i in protocol['train']],0)
    return protocol, annotations, manifest, rows, np.r_[prior,0.].astype(np.float32)

def metrics(y, prob):
    y = np.asarray(y)
    pred = prob.argmax(-1)
    matrix = np.zeros((5,5), int)
    np.add.at(matrix, (y,pred), 1)
    tp = matrix.diagonal()
    recall = tp/np.maximum(1,matrix.sum(1))
    f1 = 2*tp/np.maximum(1,matrix.sum(0)+matrix.sum(1))
    conf = prob.max(-1)
    ece = 0.
    for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        ix = (conf > low)&(conf <= high)
        if ix.any():
            ece += ix.mean()*abs(float(conf[ix].mean())-float((pred[ix]==y[ix]).mean()))
    return dict(n=len(y), macro_f1=float(f1.mean()), balanced_accuracy=float(recall.mean()),
        accuracy=float((pred==y).mean()), nll=float(-np.log(np.maximum(1e-8,prob[np.arange(len(y)),y])).mean()),
        brier=float(np.square(prob-np.eye(5)[y]).sum(1).mean()), ece=float(ece), confusion=matrix.tolist())

def predict(models, row, priors, views, temperature=1., transform=None):
    features = row['features'][:,:views].copy()
    valid = row['camera_valid'][:,:views].all(1)
    if transform == 'constant':
        features[:] = features[np.flatnonzero(valid)[0]]
    if transform == 'shuffle':
        features = features[np.random.default_rng(718).permutation(len(features))]
        # Sparse first-frame gaps retain their validity, and NaNs in shuffled inputs remain gaps.
        valid &= np.isfinite(features).all((1,2))
    if transform == 'reverse':
        features, valid = features[::-1].copy(), valid[::-1].copy()
    out = infer_stage_models(models, features, valid, priors, temperature)
    if transform == 'reverse':
        out = {k:v[::-1].copy() for k,v in out.items()}
    return out

def evaluate(models, rows, episodes, priors, views, temperature=1., transform=None, common_support=False):
    labels, probabilities, by_episode, curves, reviews = [], [], {}, {}, []
    proxy_errors = []
    for i in episodes:
        row = rows[i]
        out = predict(models, row, priors, views, temperature, transform)
        ix, y = row['points_ix'], row['points_y']
        ok = (y >= 0)&out['valid'][ix]
        if common_support:
            ok &= row['camera_valid'].all(1)[ix]
        y, ix = y[ok], ix[ok]
        p = out['probability'][ix]
        labels.extend(y); probabilities.extend(p); reviews.extend(out['review'][ix])
        by_episode[str(i)] = metrics(y,p)
        curves[str(i)] = dict(times_s=row['times'].tolist(), stage=out['stage'].tolist(),
            confidence=[float(v) if np.isfinite(v) else None for v in out['confidence']],
            progress=[float(v) if np.isfinite(v) else None for v in out['progress']],
            review=out['review'].tolist(), annotation_indices=ix.tolist(), annotation_stages=y.tolist())
        offsets = np.r_[0,np.cumsum(priors)[:-1]]
        target = offsets[y] + priors[y]*row['within'][ix]
        proxy_errors.extend(abs(target-out['progress'][ix]))
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    result = metrics(labels, probabilities)
    review = np.asarray(reviews)
    result.update(per_episode=by_episode, review_fraction=float(review.mean()),
        confident_accuracy=float((probabilities.argmax(-1)[~review]==labels[~review]).mean()) if (~review).any() else None,
        stage_progress_proxy_mae=float(np.mean(proxy_errors)))
    return result, curves

def training_arrays(rows, episodes, views, rewind, context):
    xs, vs, ys, ws = [], [], [], []
    for i in episodes:
        r = rows[i]
        valid = r['camera_valid'][:,:views].all(1)
        base = context_indices(len(valid),context)
        variants = [base]
        if rewind:
            # Reverse and forward-then-rewind clips. Targets follow the observed last state.
            reverse = np.minimum(len(valid)-1,np.arange(len(valid))[:,None]+np.arange(context-1,-1,-1))
            boomerang = np.clip(np.arange(len(valid))[:,None]+np.array([0,1,2,3,3,2,1,0])[None],0,len(valid)-1)
            variants += [reverse,boomerang]
        keep = (r['y'] >= 0)&valid
        for indices in variants:
            xs.append(r['features'][indices[keep],:views]); vs.append(valid[indices[keep]])
            ys.append(r['y'][keep]); ws.append(r['within'][keep])
    return tuple(torch.from_numpy(np.concatenate(v)) for v in [xs,vs,ys,ws])

def train_one(name, seed, rows, protocol, priors):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    views = 1 if name == 'head_mlp' else 2
    model = StageNet(views=views, temporal='temporal' in name)
    clean = np.concatenate([rows[i]['features'][rows[i]['camera_valid'][:,:views].all(1),:views] for i in protocol['train']])
    model.mean.copy_(torch.from_numpy(clean.mean(0)))
    model.scale.copy_(torch.from_numpy(np.maximum(clean.std(0),.1)))
    x,valid,y,w = training_arrays(rows, protocol['train'], views, name.endswith('rewind'),8)
    classes = [torch.where(y == k)[0].numpy() for k in range(5)]
    if any(len(v)==0 for v in classes):
        raise ValueError('Missing training phase')
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best, best_key, history = None, None, []
    for step in range(1,protocol['train_steps']+1):
        model.train()
        ix = np.array([rng.choice(classes[k]) for k in rng.integers(0,5,64)])
        logits, within = model(x[ix],valid[ix])
        # Only the annotated stage's sub-progress is supervised; no fake failure outcomes.
        local = within[torch.arange(len(ix)),y[ix]]
        loss = F.cross_entropy(logits,y[ix]) + .5*F.mse_loss(local,w[ix])
        opt.zero_grad();loss.backward();nnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step % protocol['validate_every'] == 0:
            scores,_ = evaluate([model],rows,protocol['validation'],priors,views)
            key = (scores['macro_f1'],-scores['nll'])
            history.append(dict(step=step,loss=float(loss.detach()),validation_f1=scores['macro_f1'],validation_nll=scores['nll']))
            if best_key is None or key > best_key:
                best_key=key;best=copy.deepcopy(model.state_dict());best_step=step
    model.load_state_dict(best)
    scores,_ = evaluate([model],rows,protocol['validation'],priors,views)
    return model.eval(),dict(seed=seed,best_step=best_step,validation=scores,history=history,
        train_samples=len(y),parameters=sum(p.numel() for p in model.parameters()))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'stage_training',help='New output directory; existing test reports are never overwritten')
    args=parser.parse_args()
    torch.set_num_threads(2)
    protocol,annotations,manifest,rows,priors=read_study()
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'test_report.json').exists():
        raise ValueError('Test already evaluated. Create a new explicitly versioned protocol for further studies.')
    candidates={}; reports={}
    started=time.monotonic()
    for name in protocol['candidates']:
        models=[];runs=[]
        for seed in protocol['seeds']:
            model,report=train_one(name,seed,rows,protocol,priors)
            models.append(model);runs.append(report)
            print(name,seed,'validation F1',round(report['validation']['macro_f1'],4),'step',report['best_step'],flush=True)
        candidates[name]=models
        reports[name]=dict(runs=runs,mean_validation_f1=float(np.mean([r['validation']['macro_f1'] for r in runs])),
                           mean_validation_nll=float(np.mean([r['validation']['nll'] for r in runs])))
        torch.save(dict(states=[m.state_dict() for m in models],config=models[0].config),out/(name+'.pt'))
        atomic_json(out/'validation.json',reports)
    winner=max(reports,key=lambda n:(reports[n]['mean_validation_f1'],-reports[n]['mean_validation_nll']))
    models=candidates[winner];views=models[0].config['views']
    # Temperature fit and model selection exclusively on validation, before opening test scores.
    calibration=[]
    for temp in np.linspace(.5,3.,26):
        score,_=evaluate(models,rows,protocol['validation'],priors,views,float(temp))
        calibration.append(dict(temperature=float(temp),nll=score['nll'],ece=score['ece']))
    temperature=min(calibration,key=lambda v:v['nll'])['temperature']
    selection=dict(winner=winner,temperature=temperature,calibration=calibration,priors=priors.tolist(),
                   selection_basis='validation only',frozen_protocol_sha256=digest(ROOT/'protocol.v1.json'))
    atomic_json(out/'selection.json',selection)
    results={};curves={}
    for name,ms in candidates.items():
        results[name],_=evaluate(ms,rows,protocol['test'],priors,ms[0].config['views'],common_support=True)
    for control in [None,'reverse','shuffle','constant']:
        key=control or 'selected_calibrated'
        results[key],curves[key]=evaluate(models,rows,protocol['test'],priors,views,temperature,control,common_support=True)
    # Clock baseline from train-only normalized phase boundaries. No fitting to held-out frames.
    relative=np.mean([np.array(rows[i]['bounds'][1:])/rows[i]['times'][-1] for i in protocol['train']],0)
    yt,pt=[],[]
    for i in protocol['test']:
        row=rows[i];y=row['points_y'];ok=(y>=0)&row['camera_valid'].all(1)[row['points_ix']];ix=row['points_ix'][ok]
        yp=np.searchsorted(relative,row['times'][ix]/row['times'][-1],side='right')
        yt.extend(y[ok]);pt.extend(np.eye(5)[yp]*.96+.008)
    results['clock_prior']=metrics(np.asarray(yt),np.asarray(pt))
    stress={str(i):predict(models,rows[i],priors,views,temperature) for i in protocol['stress']}
    for k,value in stress.items():
        curves['stress_'+k]=dict(times_s=rows[int(k)]['times'].tolist(), stage=value['stage'].tolist(),
            confidence=[float(v) if np.isfinite(v) else None for v in value['confidence']],review=value['review'].tolist(),
            warning='Exploratory task variants/recovery, no success labels or tuning. Outputs may be wrong.')
    report=dict(selection=selection,results=results,elapsed_s=time.monotonic()-started,
        interpretation='Provisional assistant visual annotation benchmark, one session, 5 test recordings; not success accuracy or robot policy performance.',
        sources={r['episode']:r['source_sha256'] for r in manifest['episodes']},
        annotation_sha256=digest(ROOT/'annotations.v1.json'),protocol_sha256=digest(ROOT/'protocol.v1.json'))
    atomic_json(out/'test_report.json',report);atomic_json(out/'test_curves.json',curves)
    camera_names=['head','right_wrist'][:views]
    saved=dict(kind='stage_progress',version=1,config=models[0].config,states=[m.state_dict() for m in models],
        priors=priors.tolist(),temperature=temperature,stage_names=annotations['stage_names'],
        cameras=camera_names,feature_contracts=[manifest['episodes'][0]['cameras'][c]['contract'] for c in camera_names],
        annotation_status=annotations['status'],annotation_sha256=report['annotation_sha256'],
        protocol_sha256=report['protocol_sha256'],test_report_sha256=digest(out/'test_report.json'),
        task_id='g1_banana_left_table_to_right_basket',winner=winner)
    torch.save(saved,out/'selected.pt')
    print('SELECTED',winner,'TEST',results['selected_calibrated'],'CLOCK',results['clock_prior'],flush=True)

if __name__ == '__main__':
    main()
