"""Check production inference against frozen test predictions and original source fingerprints."""
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
ROOT=Path(__file__).resolve().parent
APP=ROOT.parents[1]

def main():
    local=json.loads((ROOT/'local_stage_report.json').read_text())
    assert local['state']=='complete' and len(local['episodes'])==18
    curves=json.loads((ROOT/'stage_training/test_curves.json').read_text())['selected_calibrated']
    checks=[]; maximum=0.
    for row in local['episodes']:
        i=str(int(row['name'][-6:]))
        result=json.loads((APP/'workspace/warp/scores'/(row['signature']+'.json')).read_text())
        assert result['kind']=='stage_progress' and result['summary']['terminal_verdict']=='unverified'
        if i in curves:
            c=curves[i]
            assert result['stage']==c['stage'] and result['needs_review']==c['review']
            np.testing.assert_allclose(result['times_s'],c['times_s'],atol=0,rtol=0)
            for key in ['confidence','progress']:
                a=np.array([np.nan if v is None else v for v in result[key]])
                b=np.array([np.nan if v is None else v for v in c[key]])
                np.testing.assert_allclose(a,b,atol=2e-6,rtol=0,equal_nan=True)
                maximum=max(maximum,float(np.nanmax(abs(a-b))))
            checks.append(dict(episode=row['name'],signature=row['signature'],stage_and_review_exact=True))
    annotations=json.loads((ROOT/'annotations.v1.json').read_text())
    for a in annotations['episodes']:
        s=Path(a['source']).stat()
        assert dict(size=s.st_size,mtime_ns=s.st_mtime_ns)==a['source_fingerprint']
    db=sqlite3.connect(f'file:{APP}/workspace/reviews.sqlite3?mode=ro',uri=True)
    reviews=db.execute('SELECT id,body FROM reviews ORDER BY id').fetchall()
    audit=db.execute('SELECT count(*) FROM audit').fetchone()[0];db.close()
    assert reviews==[] and audit==0
    report=dict(passed=True,scored_recordings=18,test_recordings_exact_class_parity=checks,
        max_probability_or_progress_abs_error=maximum,original_source_fingerprints_unchanged=18,
        saved_review_rows=len(reviews),review_audit_rows=audit,automatic_success_labels_written=0)
    (ROOT/'runtime_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
