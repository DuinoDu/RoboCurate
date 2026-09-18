"""Recompute every candidate's OOF predictions using package-relative inputs."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;APP=ROOT.parents[1];V1=ROOT.parent/'model_upgrade_20260917';V2=ROOT.parent/'stage_v2_20260917';V3=ROOT.parent/'stage_v3_motion_20260917'
sys.path[:0]=[str(APP),str(V1)]
from train_stages import metrics
from warp_progress.stages_v3 import MotionStageNet
from warp_progress.stages_v4 import RefinedStageNet
from warp_progress.stages_v2 import infer_fusion
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def local_rows():
    annotations={r['episode']:r for r in read(V1/'annotations.v1.json')['episodes']}
    states={r['episode']:r for r in read(V2/'state_manifest.json')['episodes']};rows={}
    for row in read(V1/'features_manifest.json')['episodes']:
        arrays=[]
        for camera in ['head','right_wrist']:
            info=row['cameras'][camera];p=APP/'workspace/warp/features'/Path(info['cache']).name
            assert sha(p)==info['cache_sha256']
            with np.load(p,allow_pickle=False) as z:arrays.append({k:z[k] for k in z.files})
        s=states[row['episode']];p=V2/'states'/Path(s['path']).name;assert sha(p)==s['sha256']
        with np.load(p,allow_pickle=False) as z:state={k:z[k] for k in z.files}
        assert all(np.array_equal(a['timestamp_ns'],state['timestamp_ns']) for a in arrays)
        times=(state['timestamp_ns']-row['start_ns'])/1e9
        points=annotations[row['episode']]['points'];ix=np.abs(times[:,None]-np.array([p['time_s'] for p in points])).argmin(0)
        rows[int(row['episode'][-6:])]=dict(features=np.stack([a['features'] for a in arrays],1),
            camera_valid=np.stack([a['valid'] for a in arrays],1),state=state['state'],state_valid=state['state_valid'],
            ix=ix,y=np.array([p['stage'] for p in points]))
    return rows

def main():
    torch.set_num_threads(2);rows=local_rows();checked={};maximum=0.;folds_verified=0
    for folder in [ROOT,ROOT.parent/'stage_v4_decoupled_20260917']:
        p=read(folder/'protocol.json');assert sha(folder/'protocol.json')==read(folder/'freeze.json')['protocol_sha256']
        report=read(folder/'training/cross_validation.json')
        for name in p['candidates']:
            cache=folder/'training'/f'{name}-oof.npz'
            if name=='v3_reference' and name in checked:
                with np.load(cache,allow_pickle=False) as a,np.load(ROOT/'training/v3_reference-oof.npz',allow_pickle=False) as b:
                    for key in a.files:np.testing.assert_array_equal(a[key],b[key])
                continue
            ys=[];ps=[];reviews=[];episodes=[]
            for f,held in enumerate(p['folds']):
                path=V3/'training'/f'motion_rw-fold{f}.pt' if name=='v3_reference' else folder/'training'/f'{name}-fold{f}.pt'
                if name=='v3_reference':assert sha(path)==p['reference_fold_sha256'][str(f)]
                saved=torch.load(path,map_location='cpu',weights_only=True);models=[]
                for weights in saved['states']:
                    if name=='v3_reference':m=MotionStageNet()
                    else:
                        cfg=dict(saved['config']);cfg.pop('use_state',None);m=RefinedStageNet(**cfg)
                    m.load_state_dict(weights,strict=True);models.append(m.eval())
                for ep in held:
                    row=rows[ep];out=infer_fusion(models,row['features'],row['camera_valid'],row['state'],row['state_valid'],saved['priors'],1.)
                    ix=row['ix'];y=row['y'];keep=(y>=0)&row['camera_valid'].all(1)[ix]&row['state_valid'][ix]
                    ys.extend(y[keep]);ps.extend(out['probability'][ix[keep]]);reviews.extend(out['review'][ix[keep]]);episodes.extend([ep]*int(keep.sum()))
                folds_verified+=1
            ps=np.asarray(ps);ys=np.asarray(ys)
            with np.load(cache,allow_pickle=False) as saved_oof:
                np.testing.assert_array_equal(ys,saved_oof['y']);np.testing.assert_array_equal(reviews,saved_oof['review'])
                np.testing.assert_array_equal(episodes,saved_oof['episode']);np.testing.assert_allclose(ps,saved_oof['p'],atol=2e-6,rtol=0)
                maximum=max(maximum,float(np.max(abs(ps-saved_oof['p']))))
            score=metrics(ys,ps)
            for k in ['macro_f1','accuracy','nll']:assert np.isclose(score[k],report[name]['overall'][k],rtol=0,atol=2e-6)
            checked[name]=dict(n=len(ys),macro_f1=score['macro_f1'],nll=score['nll'])
    result=dict(passed=True,candidates=checked,fold_checkpoints=folds_verified,member_networks=3*folds_verified,
        max_float_abs_error=maximum,original_mcap_or_native_cache_used=False,all_inputs_package_relative=True)
    return result
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False,indent=2))
