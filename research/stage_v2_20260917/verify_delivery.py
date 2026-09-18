"""Verify unpacked v2 inference using packaged feature/state arrays, without original MCAP."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;APP=ROOT.parents[1];PREV=ROOT.parent/'model_upgrade_20260917'
sys.path.insert(0,str(APP))
from warp_progress.stages_v2 import load_fusion_checkpoint,infer_fusion

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(2)
    profile=json.loads((ROOT/'installed_model.json').read_text())
    models,saved=load_fusion_checkpoint(APP/'workspace/warp/models'/profile['checkpoint'],profile['checkpoint_sha256'])
    manifest=json.loads((PREV/'features_manifest.json').read_text())
    states={r['episode']:r for r in json.loads((ROOT/'state_manifest.json').read_text())['episodes']}
    production={r['name']:r for r in json.loads((ROOT/'local_stage_report.json').read_text())['episodes']}
    maximum=0.
    for row in manifest['episodes']:
        arrays=[]
        for camera in saved['cameras']:
            info=row['cameras'][camera];path=APP/'workspace/warp/features'/Path(info['cache']).name
            assert sha(path)==info['cache_sha256']
            with np.load(path,allow_pickle=False) as z:arrays.append({k:z[k] for k in z.files})
        s=states[row['episode']];state_path=ROOT/'states'/Path(s['path']).name
        assert sha(state_path)==s['sha256']
        with np.load(state_path,allow_pickle=False) as z:state={k:z[k] for k in z.files}
        assert all(np.array_equal(a['timestamp_ns'],state['timestamp_ns']) for a in arrays)
        out=infer_fusion(models,np.stack([a['features'] for a in arrays],1),np.stack([a['valid'] for a in arrays],1),
            state['state'],state['state_valid'],saved['priors'],saved['temperature'])
        record=production[row['episode']]
        with np.load(APP/'workspace/warp/scores'/(record['signature']+'.npz'),allow_pickle=False) as z:
            for key in ['stage','review','valid','degraded']:np.testing.assert_array_equal(z[key],out[key])
            for key in ['probability','progress','confidence']:
                np.testing.assert_allclose(z[key],out[key],atol=2e-6,rtol=0,equal_nan=True)
                maximum=max(maximum,float(np.nanmax(abs(z[key]-out[key]))))
    report=dict(passed=True,recordings=len(manifest['episodes']),checkpoint_sha256=profile['checkpoint_sha256'],
        maximum_float_error=maximum,original_mcap_or_native_cache_used=False)
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
