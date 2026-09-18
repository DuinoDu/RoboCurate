"""Read-only audit of display evidence, score immutability and protected sources."""
import hashlib,json,sqlite3,sys
from pathlib import Path
import numpy as np

STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[1];APP=ROOT
sys.path.insert(0,str(APP))
from progress_service import stage_evidence

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    baseline=json.loads((STUDY/'baseline.json').read_text())
    score_root=APP/'workspace/warp/scores'
    scores={p.name:sha(p) for p in score_root.iterdir() if p.suffix in {'.json','.npz'}}
    assert scores==baseline['scores'],'Saved model scores changed'
    profiles=APP/'workspace/warp/models/profiles.json'
    assert sha(profiles)==baseline['profiles_sha256']
    models=json.loads(profiles.read_text())['models']
    for model in models:
        assert sha(profiles.parent/model['checkpoint'])==model['checkpoint_sha256']
    records=json.loads((STUDY.parent/'stage_v3_motion_20260917/local_stage_report.json').read_text())['episodes']
    assert len(records)==18
    checks=[]
    for row in records:
        path=score_root/(row['signature']+'.json');result=json.loads(path.read_text())
        assert result['model_id']=='g1-stage-v3'
        assert result['summary']['terminal_verdict']=='unverified'
        evidence=stage_evidence(result,path.with_suffix('.npz'))
        assert evidence['available']
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as saved:
            valid=np.asarray(result['stage'])>=0
            for i,probability in enumerate(evidence['probability']):
                if valid[i]:np.testing.assert_array_equal(probability,saved['probability'][i])
                else:assert probability is None
            for key in ['camera_valid','state_valid']:
                np.testing.assert_array_equal(evidence[key],saved[key])
            np.testing.assert_array_equal(result['stage'],saved['stage'])
            np.testing.assert_array_equal(result['needs_review'],saved['review'])
            np.testing.assert_array_equal(result['review_reasons'],saved['review_reasons'])
            np.testing.assert_allclose(np.asarray(result['confidence'],float),saved['confidence'],rtol=0,atol=2e-6,equal_nan=True)
            checks.append(dict(episode=row['name'],signature=row['signature'],samples=len(valid),
                               valid_samples=int(valid.sum()),display_evidence_exact=True))
    protected=['static/workspace.css','static/viewer/robot3d.js','static/solid-canvas.js',
               'scripts/build_preview_meshes.py','static/viewer/g1-preview.json','static/viewer/trails.js']
    assert all(sha(APP/p)==baseline['source'][p] for p in protected)
    with sqlite3.connect(f'file:{APP}/workspace/reviews.sqlite3?mode=ro',uri=True) as db:
        reviews=db.execute('SELECT id,body FROM reviews ORDER BY id').fetchall()
        audit=db.execute('SELECT count(*) FROM audit').fetchone()[0]
    assert len(reviews)==baseline['review_rows'] and audit==baseline['audit_rows']
    assert hashlib.sha256(json.dumps(reviews,sort_keys=True).encode()).hexdigest()==baseline['review_sha256']
    report=dict(passed=True,active_stage_model='g1-stage-v3',recordings=checks,
        unchanged_score_files=len(scores),model_profiles_unchanged=True,checkpoint_hashes_verified=len(models),
        protected_sources=protected,review_state_unchanged=True,automatic_success_labels_written=0,
        scope='Display/API evidence verification only; model training and accuracy are unchanged.')
    (STUDY/'runtime_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='recordings'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
