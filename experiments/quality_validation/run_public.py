"""Reproducible public-data evaluation; no training labels enter QC inputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from quality_evidence import fixed_spacing, motion_facts, value_facts, integrity_summary
from upstream import motion_reference, spacing_reference, score_reference


def auc(labels, scores):
    y=np.asarray(labels);s=np.asarray(scores)
    p=s[y==1];n=s[y==0]
    return float(((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])).mean()) if len(p) and len(n) else None


def rank(values):
    x=np.asarray(values);order=np.argsort(x,kind='stable');result=np.empty(len(x),float)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        result[order[i:j]]=(i+j-1)/2;i=j
    return result


def balanced(y,pred):
    return float(np.mean([np.mean(pred[y==v]==v) for v in (0,1,2)]))


def fit_cuts(x,y):
    unique=np.unique(x);cuts=np.r_[-np.inf,(unique[:-1]+unique[1:])/2,np.inf]
    best=(-1,None,None)
    for i,a in enumerate(cuts):
        for b in cuts[i:]:
            pred=np.where(x<=a,2,np.where(x<=b,1,0));accuracy=balanced(y,pred)
            if accuracy>best[0]:best=(accuracy,float(a),float(b))
    return best


def label_masks(f):
    result={}
    for level in ('better','okay','worse'):
        for operator in (1,2):
            key=f'{level}_operator_{operator}'
            if key in f['mask']:
                for name in f['mask'][key][:]:result[name.decode()]=(level,operator)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=HERE.parents[1]/'workspace/public-quality-data')
    parser.add_argument('--output',type=Path,default=HERE.parents[1]/'workspace/evidence/quality-validation');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((HERE/'protocol.json').read_text());spec=json.loads((HERE/'datasets.json').read_text())
    reference_motion=motion_reference();reference_spacing=spacing_reference();reference_score=score_reference()
    rows=[];faults=defaultdict(Counter);parity_deltas=[]
    for item in spec['files']:
        path=args.data/item['name']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        kind='mh' if '_mh.' in item['name'] else 'paired'
        with h5py.File(path,'r') as f:
            fps=json.loads(f['data'].attrs['env_args'])['env_kwargs']['control_freq']
            masks=label_masks(f)
            for name in sorted(f['data'],key=lambda n:int(n.split('_')[-1])):
                ep=f['data'][name];q=ep['obs/robot0_joint_pos'][:];a=ep['actions'][:];n=len(q)
                time_s=np.arange(n)/fps;time_ns=np.rint(time_s*1e9).astype(np.int64)
                facts=motion_facts(time_ns,q);ref=reference_motion(time_ns,q)
                keys=['mean_velocity','peak_velocity','motionless_fraction','trajectory_change_p95','final_pose_speed','final_pose_unsettled_ratio']
                delta=max(abs(facts[k]-ref[k]) for k in keys if k in facts and k in ref);parity_deltas.append(delta)
                arrays=[q,a,ep['obs/robot0_eef_pos'][:]]
                checks=[dict(severity='PASS' if len(z)==n and np.isfinite(z).all() else 'FAIL') for z in arrays]
                checks.append(dict(severity='PASS' if q.shape[1]==7 and a.shape[1]==7 and int(ep.attrs['num_samples'])==n else 'FAIL'))
                trust=integrity_summary(checks);assert trust['score']==reference_score(checks)
                reward=ep['rewards'][:]
                row=dict(dataset=kind,episode=name,samples=n,fps=fps,duration_s=n/fps,
                         task_success=bool(np.any(reward>0)),label_source='stored sparse reward (evaluation only)',
                         initial_state_sha256=hashlib.sha256(ep['states'][0].tobytes()).hexdigest(),
                         motion=facts,value_facts=value_facts(q),integrity=trust,
                         clock_observation='reconstructed simulation grid; actual timestamp health unmeasured')
                if kind=='mh':row.update(proficiency=masks[name][0],operator=masks[name][1])
                rows.append(row)
                # Corruption labels are generated independently of either implementation.
                early=time_s.copy();early[1:]+=.0002
                late=time_s.copy();late[max(3,n//2):]+=2.2/fps
                combined=early.copy();combined[max(3,n//2):]+=2.2/fps
                for case,t,expected in [('clean',time_s,'PASS'),('minor',early,'WARN'),('late_gap',late,'FAIL'),('minor_then_gap',combined,'FAIL')]:
                    old=reference_spacing(t,fps);old='PASS' if old=='INFO' else old
                    new=fixed_spacing(t,fps)['severity']
                    faults[case]['cases']+=1;faults[case]['upstream_expected']+=old==expected;faults[case]['fused_expected']+=new==expected
                
    mh=[r for r in rows if r['dataset']=='mh'];paired=[r for r in rows if r['dataset']=='paired']
    develop=[r for r in mh if r['operator']==1];test=[r for r in mh if r['operator']==2]
    def feature(r):return r['motion']['trajectory_change_p95']
    numeric={'worse':0,'okay':1,'better':2}
    dev_x=np.array([feature(r) for r in develop]);dev_y=np.array([numeric[r['proficiency']] for r in develop])
    test_x=np.array([feature(r) for r in test]);test_y=np.array([numeric[r['proficiency']] for r in test])
    dev_acc,lo,hi=fit_cuts(dev_x,dev_y);pred=np.where(test_x<=lo,2,np.where(test_x<=hi,1,0))
    confusion=[[int(np.sum((test_y==a)&(pred==b))) for b in (2,1,0)] for a in (2,1,0)]
    mh_result=dict(development_n=len(develop),holdout_n=len(test),cutpoints=[v if np.isfinite(v) else None for v in (lo,hi)],
                   unbounded_cutpoints=[('negative_infinity' if v<0 else 'positive_infinity') if not np.isfinite(v) else None for v in (lo,hi)],
                   development_balanced_accuracy=dev_acc,holdout_balanced_accuracy=balanced(test_y,pred),
                   class_order=['better','okay','worse'],confusion=confusion,
                   holdout_spearman=float(np.corrcoef(rank(-test_x),rank(test_y))[0,1]),
                   caution='Proficiency is an operator-group label; all MH source demos were collected as successful. Only six operators, one per class in holdout.')
    for r,p in zip(test,pred):r['experimental_proficiency_prediction']={v:k for k,v in numeric.items()}[int(p)]
    groups=defaultdict(list)
    for r in paired:groups[r['initial_state_sha256']].append(r)
    wins=[]
    for group in groups.values():
        if len(group)==2 and sum(r['task_success'] for r in group)==1:
            good=next(r for r in group if r['task_success']);bad=next(r for r in group if not r['task_success'])
            wins.append(float(feature(good)<feature(bad))+.5*float(feature(good)==feature(bad)))
    rng=np.random.default_rng(protocol['seed']);boot=np.mean(rng.choice(wins,(2000,len(wins)),replace=True),axis=1)
    paired_result=dict(episodes=len(paired),labels=dict(Counter('success' if r['task_success'] else 'failure' for r in paired)),
                       matched_pairs=len(wins),excluded_from_matched_comparison=len(paired)-2*len(wins),
                       smoother_success_pair_fraction=float(np.mean(wins)),pair_bootstrap_95_ci=np.quantile(boot,[.025,.975]).tolist(),
                       smoother_predicts_success_auc=auc([int(r['task_success']) for r in paired],[-feature(r) for r in paired]),
                       integrity_pass_counts=dict(Counter('success' if r['task_success'] else 'failure' for r in paired if r['integrity']['score']==100)))
    result=dict(protocol=protocol,data_provenance=spec,episodes=len(rows),upstream_motion_parity=dict(episodes=len(parity_deltas),max_abs_difference=max(parity_deltas)),
                synthetic_timing_tests={k:dict(v) for k,v in faults.items()},multi_human=mh_result,paired=paired_result,
                all_integrity_scores=dict(Counter(str(r['integrity']['score']) for r in rows)),
                images='not present in low-dimensional benchmark; not tested',g1_grades='not applicable to Panda data; no claim of A-D accuracy on this benchmark')
    (args.output/'public-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    (args.output/'public-episodes.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('protocol','data_provenance')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
