"""Measure the supplied G1 recordings without changing grades or annotations."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from studio import StudioLibrary
from quality_evidence import fixed_spacing

def main():
    lib=StudioLibrary(ROOT/'workspace');before=lib.store.all()
    for path in lib.store.setting('roots',[]):lib.add_root(Path(path))
    rows=[];strict=Counter();raw_over=[]
    for ep in lib.order:
        record=lib.record(lib.ref(ep));original=record['quality']['grading']['grade']
        if lib.state(ep)['state']!='ready':continue
        report=lib.reference_checks(ep)
        with np.load(lib.cache_dir(lib.ref(ep))/'native.npz',allow_pickle=False) as native:
            for name in record['cameras']:
                t=native['camera.'+name+'.t'];strict[fixed_spacing((t-t[0])/1e9,30)['severity']]+=1
                raw_over.append(report['timing']['camera.'+name]['period_violation_pct'])
        assert lib.record(lib.ref(ep))['quality']['grading']['grade']==original
        rows.append(dict(episode=record['episode_id'],baseline_grade=original,evidence=report))
    assert lib.store.all()==before
    result=dict(recordings=len(rows),camera_streams=sum(strict.values()),
                strict_fixed_30hz_results_on_raw=dict(strict),
                raw_10ms_period_violation_pct=dict(min=min(raw_over),median=float(np.median(raw_over)),max=max(raw_over)),
                saved_reviews_unchanged=True,baseline_grades_unchanged=True,
                note='Trigger rates are not false-positive rates; no human fault labels are available for these G1 streams.',records=rows)
    target=ROOT/'workspace/evidence/quality-validation/g1-results.json';target.parent.mkdir(exist_ok=True,parents=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
