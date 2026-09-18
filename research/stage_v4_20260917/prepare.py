"""Freeze the next study before training; reuse verified v3 fold references."""
import hashlib,json,sqlite3
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;V3=ROOT.parent/'stage_v3_motion_20260917';APP=ROOT.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    if (ROOT/'protocol.json').exists():raise ValueError('Protocol already frozen')
    old=json.loads((V3/'protocol.json').read_text())
    keys=['development','folds','historical_regression','stress','seeds','steps','batch_size','lr','weight_decay','annotation_sha256','source_feature_manifest_sha256','state_manifest_sha256','weak_weight','weak_batch_size']
    p={k:old[k] for k in keys}
    p.update(version=4,created_at=datetime.now(timezone.utc).isoformat(),candidates=['v3_reference','contrastive','boundary_structure','combined'],
        reference_fold_sha256={str(i):sha(V3/'training'/f'motion_rw-fold{i}.pt') for i in range(5)},
        reference_checkpoint_sha256=json.loads((V3/'installed_model.json').read_text())['checkpoint_sha256'],
        architecture='Exact v3 shared causal motion architecture; optional 96-64-64 contrastive projection and one boundary output; no extra sensors or timestamps.',
        contrastive=dict(weight=.05,temperature=.1,positives='Same stage, different training recording only; hard labels including inherited same-stage interval labels.'),
        structure=dict(boundary_weight=.1,segment_weight=.5,warmup_steps=100,positive_bags=4,stable_samples=16,segments=2,max_segment_samples=12,
            boundary_target='At least one event in (left known anchor, right known anchor] when labels differ; no exact boundary label. No propagation across missing inputs.',
            segment_target='CDF of class confidence should be uniform within known constant-stage training interiors; no loss across transition gaps.'),
        selection='Among eligible candidates, highest pooled development macro F1 at T=1, tie lower NLL. Historical recordings never decide selection.',
        promotion='F1 must not decrease and NLL <= reference+0.02. Additionally require F1 >= reference+0.005 OR NLL <= reference-0.01 OR at least two more correctly bracketed transitions with no more extra switches.',
        boundary_metric='Sparse-label diagnostic only: match predicted a-to-b changes inside (anchor_a, anchor_b]; count unmatched changes between first/last known valid anchor. This is NOT precise boundary ground truth or segmental F1.',
        review_policy='Keep v3 raw/calibrated union at threshold 0.8. Add learned boundary>=0.5 only if development OOF unflagged errors decrease and reviewed fraction increases <=0.10.',
        calibration='Member-wise temperature before ensemble averaging, 0.5..3.0 step0.1, minimum OOF NLL; no historical input.',
        budget='300 optimizer steps; 64 balanced hard and 32 random-walk soft samples. Structure variants add bracket/stable/segment samples; extra compute disclosed. No hyperparameter sweep.',
        limitation='Same-session provisional annotations, repeatedly reused development and known historical regression. No new blind test; no final-success labels or policy-performance claim.',
        sources=['https://ieeexplore.ieee.org/document/10598312/','https://arxiv.org/html/2604.01859v1'])
    write(ROOT/'protocol.json',p);write(ROOT/'freeze.json',dict(protocol_sha256=sha(ROOT/'protocol.json')))
    db=sqlite3.connect(f'file:{APP}/workspace/reviews.sqlite3?mode=ro',uri=True)
    rows=db.execute('SELECT id,body FROM reviews ORDER BY id').fetchall();audit=db.execute('SELECT count(*) FROM audit').fetchone()[0];db.close()
    write(ROOT/'review_baseline.json',dict(review_rows=len(rows),audit_rows=audit,review_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()))
    print('Frozen v4 contrastive/interval-boundary protocol')
if __name__=='__main__':main()
