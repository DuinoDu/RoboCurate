"""Freeze one follow-up hypothesis before training; never edit the first study."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;FIRST=ROOT.parent/'stage_v4_20260917'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    assert not (ROOT/'protocol.json').exists()
    p=json.loads((FIRST/'protocol.json').read_text())
    p.update(created_at=datetime.now(timezone.utc).isoformat(),candidates=['v3_reference','detached_progress'],
        architecture='Unchanged v3 motion network and parameter count. Stage encoder is trained by hard/soft stage losses only. Progress head reads detached stage embedding/probabilities.',
        motivation='SARM2 separately trains stage and value estimators. Lightweight adaptation: stop proxy-progress gradients at the shared stage representation; not SARM2 MMoE or multi-task reproduction.',
        parent_protocol_sha256=sha(FIRST/'protocol.json'),
        adaptation_disclosure='Follow-up designed after observing the first study single-component results and early combined folds. Same repeatedly used development split, not a new blind experiment. No hyperparameter sweep.',
        sources=['https://arxiv.org/html/2606.10305v1'],
        budget='Same 300 steps, shared initial weights, seeds, 64 hard +32 RW soft samples as v3; no extra boundary samples or contrastive head.',
        additional_promotion_gate='OOF global progress-proxy MAE at T=1 must be <= original v3 +0.01. Uses inherited approximate within-stage labels, not independent progress ground truth.')
    write(ROOT/'protocol.json',p);write(ROOT/'freeze.json',dict(protocol_sha256=sha(ROOT/'protocol.json')))
    (ROOT/'review_baseline.json').write_bytes((FIRST/'review_baseline.json').read_bytes())
    print('Frozen independent stage/proxy-gradient study')
if __name__=='__main__':main()
