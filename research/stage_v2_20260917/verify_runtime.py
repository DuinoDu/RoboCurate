"""Verify actual worker output, state alignment, source integrity and read-only reviews."""
import hashlib,json,sqlite3,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;APP=ROOT.parents[1];PREV=ROOT.parent/'model_upgrade_20260917'
sys.path[:0]=[str(APP),str(ROOT)]
from train_cv import load_data,infer
from warp_progress.stages_v2 import load_fusion_checkpoint
from warp_progress.features import atomic_json

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    import torch
    torch.set_num_threads(2)
    _,_,_,rows,_=load_data()
    local=json.loads((ROOT/'local_stage_report.json').read_text());assert local['state']=='complete' and len(local['episodes'])==18
    profile=json.loads((ROOT/'installed_model.json').read_text())
    models,saved=load_fusion_checkpoint(APP/'workspace/warp/models'/profile['checkpoint'],profile['checkpoint_sha256'])
    checks=[];maximum=0.
    for record in local['episodes']:
        episode=int(record['name'][-6:]);row=rows[episode]
        result=json.loads((APP/'workspace/warp/scores'/(record['signature']+'.json')).read_text())
        assert result['kind']=='stage_progress' and result['stage_version']==2 and result['summary']['terminal_verdict']=='unverified'
        expected=infer(models,row,saved['priors'],saved['temperature'])
        assert result['stage']==expected['stage'].tolist() and result['needs_review']==expected['review'].tolist()
        assert result['degraded']==expected['degraded'].tolist()
        np.testing.assert_array_equal(result['times_s'],row['times'])
        with np.load(APP/'workspace/warp/scores'/(record['signature']+'.npz'),allow_pickle=False) as scored:
            np.testing.assert_array_equal(scored['timestamp_ns'],row['timestamp_ns'])
            np.testing.assert_array_equal(scored['state_valid'],row['state_valid'])
            assert (scored['state_source_timestamp_ns'][scored['state_valid']]<=scored['timestamp_ns'][scored['state_valid'],None]).all()
            for k in ['probability','confidence','progress']:
                np.testing.assert_allclose(scored[k],expected[k],rtol=0,atol=2e-6,equal_nan=True)
                maximum=max(maximum,float(np.nanmax(abs(scored[k]-expected[k]))))
        checks.append(dict(episode=record['name'],signature=record['signature'],stage_review_and_masks_exact=True))
    annotations=json.loads((PREV/'annotations.v1.json').read_text())
    for row in annotations['episodes']:
        stat=Path(row['source']).stat()
        assert dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)==row['source_fingerprint']
    states=json.loads((ROOT/'state_manifest.json').read_text())
    for row in states['episodes']:assert sha(row['native_path'])==row['native_sha256']
    db=sqlite3.connect(f'file:{APP}/workspace/reviews.sqlite3?mode=ro',uri=True)
    review=db.execute('SELECT id,body FROM reviews ORDER BY id').fetchall();audit=db.execute('SELECT count(*) FROM audit').fetchone()[0];db.close()
    baseline=json.loads((ROOT/'review_baseline.json').read_text())
    assert dict(review_rows=len(review),audit_rows=audit,review_sha256=hashlib.sha256(json.dumps(review,sort_keys=True).encode()).hexdigest())==baseline
    report=dict(passed=True,scored_recordings=18,all_recording_parity=checks,max_float_abs_error=maximum,
        original_source_fingerprints_unchanged=18,native_cache_sha256_unchanged=18,review_state_unchanged=True,
        automatic_success_labels_written=0,review_rows=len(review),audit_rows=audit)
    atomic_json(ROOT/'runtime_verification.json',report)
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
