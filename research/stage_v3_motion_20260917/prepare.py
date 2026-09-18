"""Separate, preregistered motion-difference experiment; original v3 unchanged."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;FIRST=ROOT.parent/'stage_v3_20260917'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    if (ROOT/'protocol.json').exists():raise ValueError('Protocol already frozen')
    p=json.loads((FIRST/'protocol.json').read_text())
    p.update(candidates=['v2_reference','motion','motion_rw'],
        architecture='V2 8-frame temporal stage network with explicit visual and measured-position differences before each stream projection; endpoint masks. No added sensors or absolute time.',
        rationale='Longer-context preliminary development results did not show stable gains; test explicit dynamics instead. Iterative development is acknowledged, not independent validation.',
        parent_protocol_sha256=sha(FIRST/'protocol.json'),
        sources=['https://arxiv.org/abs/2606.28320',
        'https://openaccess.thecvf.com/content/WACV2024/papers/Hirsch_Random_Walks_for_Temporal_Action_Segmentation_With_Timestamp_Supervision_WACV_2024_paper.pdf'])
    write(ROOT/'protocol.json',p);write(ROOT/'freeze.json',dict(protocol_sha256=sha(ROOT/'protocol.json')))
    (ROOT/'review_baseline.json').write_bytes((FIRST/'review_baseline.json').read_bytes())
    print('Frozen separate motion experiment; same v2 comparison and promotion rules')
if __name__=='__main__':main()
