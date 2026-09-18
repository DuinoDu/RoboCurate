"""Verify every production v3 score against independent saved-input inference."""
import hashlib,json,sqlite3,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;APP=ROOT.parents[1];V2=ROOT.parent/'stage_v2_20260917';V1=ROOT.parent/'model_upgrade_20260917'
sys.path[:0]=[str(APP),str(V2)]
from train_cv import load_data
from warp_progress.stages_v3 import load_stage_v3,inference_v3
from warp_progress.features import atomic_json
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(2);_,_,_,rows,_=load_data()
    profile=json.loads((ROOT/'installed_model.json').read_text())
    models,saved=load_stage_v3(APP/'workspace/warp/models'/profile['checkpoint'],profile['checkpoint_sha256'])
    local=json.loads((ROOT/'local_stage_report.json').read_text());assert local['state']=='complete' and len(local['episodes'])==18
    checks=[];maximum=0.
    for record in local['episodes']:
        row=rows[int(record['name'][-6:])]
        result=json.loads((APP/'workspace/warp/scores'/(record['signature']+'.json')).read_text())
        assert result['stage_version']==3 and result['summary']['terminal_verdict']=='unverified'
        expected=inference_v3(models,row['features'],row['camera_valid'],row['state'],row['state_valid'],
            saved['priors'],saved['temperature'],saved['boundary_review'],saved.get('review_guard',False))
        assert result['stage']==expected['stage'].tolist() and result['needs_review']==expected['review'].tolist()
        assert result['review_reasons']==expected['review_reasons'].tolist()
        assert result['review_guard_enabled']==saved.get('review_guard',False)
        assert result['review_guard_report_sha256']==saved.get('review_guard_report_sha256')
        np.testing.assert_array_equal(result['times_s'],row['times'])
        with np.load(APP/'workspace/warp/scores'/(record['signature']+'.npz'),allow_pickle=False) as scored:
            np.testing.assert_array_equal(scored['state_valid'],row['state_valid'])
            np.testing.assert_array_equal(scored['boundary_review'],expected['boundary_review'])
            np.testing.assert_array_equal(scored['guard_review'],expected['guard_review'])
            for k in ['probability','confidence','progress']:
                np.testing.assert_allclose(scored[k],expected[k],rtol=0,atol=2e-6,equal_nan=True)
                maximum=max(maximum,float(np.nanmax(abs(scored[k]-expected[k]))))
        checks.append(dict(episode=record['name'],stage_and_review_exact=True,signature=record['signature']))
    for row in json.loads((V1/'annotations.v1.json').read_text())['episodes']:
        stat=Path(row['source']).stat();assert dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)==row['source_fingerprint']
    for row in json.loads((V2/'state_manifest.json').read_text())['episodes']:assert sha(row['native_path'])==row['native_sha256']
    db=sqlite3.connect(f'file:{APP}/workspace/reviews.sqlite3?mode=ro',uri=True)
    review=db.execute('SELECT id,body FROM reviews ORDER BY id').fetchall();audit=db.execute('SELECT count(*) FROM audit').fetchone()[0];db.close()
    actual=dict(review_rows=len(review),audit_rows=audit,review_sha256=hashlib.sha256(json.dumps(review,sort_keys=True).encode()).hexdigest())
    assert actual==json.loads((ROOT/'review_baseline.json').read_text())
    report=dict(passed=True,scored_recordings=18,all_recording_parity=checks,max_float_abs_error=maximum,
        source_fingerprints_unchanged=18,state_cache_sha256_unchanged=18,review_state_unchanged=True,automatic_success_labels_written=0)
    atomic_json(ROOT/'runtime_verification.json',report);print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
