"""Verify released v3 with package-relative inputs only, without original MCAP."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;APP=ROOT.parents[1]
V1=ROOT.parent/'model_upgrade_20260917';V2=ROOT.parent/'stage_v2_20260917'
sys.path.insert(0,str(APP))
from warp_progress.stages_v3 import load_stage_v3,inference_v3
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(2)
    profile=json.loads((ROOT/'installed_model.json').read_text())
    models,saved=load_stage_v3(APP/'workspace/warp/models'/profile['checkpoint'],profile['checkpoint_sha256'])
    assert saved['review_guard_report_sha256']==sha(ROOT/'review_guard/report.json')
    manifest=json.loads((V1/'features_manifest.json').read_text())
    states={r['episode']:r for r in json.loads((V2/'state_manifest.json').read_text())['episodes']}
    production={r['name']:r for r in json.loads((ROOT/'local_stage_report.json').read_text())['episodes']}
    assert len(production)==len(manifest['episodes'])==18
    maximum=0.
    for row in manifest['episodes']:
        arrays=[]
        for camera in saved['cameras']:
            info=row['cameras'][camera];path=APP/'workspace/warp/features'/Path(info['cache']).name
            assert sha(path)==info['cache_sha256']
            with np.load(path,allow_pickle=False) as z:arrays.append({k:z[k] for k in z.files})
        s=states[row['episode']];path=V2/'states'/Path(s['path']).name
        assert sha(path)==s['sha256']
        with np.load(path,allow_pickle=False) as z:state={k:z[k] for k in z.files}
        assert all(np.array_equal(a['timestamp_ns'],state['timestamp_ns']) for a in arrays)
        out=inference_v3(models,np.stack([a['features'] for a in arrays],1),np.stack([a['valid'] for a in arrays],1),
            state['state'],state['state_valid'],saved['priors'],saved['temperature'],saved['boundary_review'],saved['review_guard'])
        record=production[row['episode']];score=APP/'workspace/warp/scores'/(record['signature']+'.json')
        result=json.loads(score.read_text());assert result['checkpoint_sha256']==profile['checkpoint_sha256']
        assert result['review_guard_enabled'] and result['summary']['terminal_verdict']=='unverified'
        with np.load(score.with_suffix('.npz'),allow_pickle=False) as z:
            for key in ['stage','review','valid','degraded','review_reasons','guard_review','boundary_review']:
                np.testing.assert_array_equal(z[key],out[key])
            for key in ['probability','progress','confidence']:
                np.testing.assert_allclose(z[key],out[key],atol=2e-6,rtol=0,equal_nan=True)
                maximum=max(maximum,float(np.nanmax(abs(z[key]-out[key]))))
    print(json.dumps(dict(passed=True,recordings=18,checkpoint_sha256=profile['checkpoint_sha256'],
        maximum_float_error=maximum,original_mcap_or_native_cache_used=False,inputs='Package-relative saved visual and state features only'),ensure_ascii=False))
if __name__=='__main__':main()
