"""Freeze v3 comparisons, reused v2 reference folds, and promotion criteria."""
import hashlib,json,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PREV=ROOT.parent/'stage_v2_20260917';APP=ROOT.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    if (ROOT/'protocol.json').exists():raise ValueError('Protocol already frozen')
    old=json.loads((PREV/'protocol.json').read_text())
    p={k:old[k] for k in ['development','folds','historical_regression','stress','seeds','steps','batch_size','lr','weight_decay','annotation_sha256','source_feature_manifest_sha256']}
    p.update(version=3,candidates=['v2_reference','v2_rw','multiscale','multiscale_rw'],
        reference='Exact completed v2 fusion_tokens folds; same 300 steps, seeds and training/held-out episodes.',
        reference_fold_sha256={str(i):sha(PREV/'training'/f'fusion_tokens-fold{i}.pt') for i in range(5)},
        state_manifest_sha256=sha(PREV/'state_manifest.json'),
        architecture='Multiscale: shared 2-layer encoder for dense last 8 samples and sparse 8 samples at stride 4 (3.5 s and 14 s spans); no future or absolute timestamps.',
        weak_supervision='Random-walk hitting probabilities on a 1-D training-video chain, clamped to existing known-stage anchors. Edge conductance from train-normalized visual and state differences; no extrapolation outside anchors or across invalid gaps. Weak targets are not ground truth.',
        weak_weight=.3,weak_batch_size=32,
        smoothing='No post-hoc smoothing or fixed stage-order enforcement at inference; regrasp and backward transitions remain possible.',
        selection='Pooled 5-fold development macro F1, tie lower NLL, T=1. Final temperature selected on member-level OOF probabilities only, before historical regression.',
        promotion='Winner must beat v2_reference macro F1 by >=0.01 and NLL must be no more than 0.03 higher. Otherwise keep v2 as deployed model.',
        review_policy='Fixed previous review plus current and next 2 Hz sample after a predicted stage change. Adopt added boundary warnings only if OOF unflagged mistakes decrease and review fraction increases by <=0.15. Otherwise retain old review policy. This does not change predictions.',
        calibration='Temperature from 0.5 to 3.0 in steps of 0.1, applying temperature separately to each ensemble member before averaging.',
        budget='300 optimizer steps, batch 64 stage-balanced strong labels, optional 32 weak transition samples, average weights at steps 200/250/300. Weak variants have extra examples/compute, explicitly reported.',
        controls='After selection: historical regression, missing-modality controls, mid-recording start, shuffled visual/state pairs. No re-selection from these results.',
        limitation='Repeated development and previously seen regression, same session and provisional annotations. No new blind test, no final-success classifier.',
        sources=['https://openaccess.thecvf.com/content/ICCV2023/papers/Bahrami_How_Much_Temporal_Long-Term_Context_is_Needed_for_Action_Segmentation_ICCV_2023_paper.pdf',
        'https://openaccess.thecvf.com/content/WACV2024/papers/Hirsch_Random_Walks_for_Temporal_Action_Segmentation_With_Timestamp_Supervision_WACV_2024_paper.pdf'])
    write(ROOT/'protocol.json',p);write(ROOT/'freeze.json',dict(protocol_sha256=sha(ROOT/'protocol.json')))
    db=sqlite3.connect(f'file:{APP}/workspace/reviews.sqlite3?mode=ro',uri=True)
    rows=db.execute('SELECT id,body FROM reviews ORDER BY id').fetchall();audit=db.execute('SELECT count(*) FROM audit').fetchone()[0];db.close()
    write(ROOT/'review_baseline.json',dict(review_rows=len(rows),audit_rows=audit,review_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()))
    print('Frozen v3 protocol, labels, reference folds and review baseline')

if __name__=='__main__':main()
