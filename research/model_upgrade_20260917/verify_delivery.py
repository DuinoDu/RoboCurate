"""Verify unpacked stage checkpoint and packaged features without original MCAPs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent

def main():
    delivery=json.loads((ROOT/'delivery.json').read_text())
    archive=Path(delivery['path'])
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==delivery['sha256']
    destination=Path(tempfile.mkdtemp(prefix='holocurate-stage-delivery-',dir='/tmp'))
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():
            assert not n.startswith('/') and '..' not in Path(n).parts
        z.extractall(destination)
    root=destination/archive.stem
    app=root/'robot-data-studio';study=root/'research/model_upgrade_20260917'
    sys.path.insert(0,str(app))
    from warp_progress.stages import load_stage_checkpoint,infer_stage_models
    from warp_progress.features import DinoEncoder
    assert Path(sys.modules['warp_progress.stages'].__file__).is_relative_to(app)
    profiles=json.loads((app/'workspace/warp/models/profiles.json').read_text())
    profile=next(p for p in profiles['models'] if p['id']=='g1-stage-v1')
    models,saved=load_stage_checkpoint(app/'workspace/warp/models'/profile['checkpoint'],profile['checkpoint_sha256'])
    torch.set_num_threads(2)
    encoder=DinoEncoder(app/'workspace/warp/models'/profile['backbone_dir'],checksums=profile['backbone_sha256'],backbone_id=profile['backbone_id'])
    del encoder
    manifest=json.loads((study/'features_manifest.json').read_text())
    curves=json.loads((study/'stage_training/test_curves.json').read_text())['selected_calibrated']
    checked=[];maximum=0.
    for row in manifest['episodes']:
        i=str(int(row['episode'][-6:]))
        if i not in curves:continue
        cams=[]
        for name in saved['cameras']:
            c=row['cameras'][name];p=app/'workspace/warp/features'/Path(c['cache']).name
            assert hashlib.sha256(p.read_bytes()).hexdigest()==c['cache_sha256']
            with np.load(p,allow_pickle=False) as a:cams.append({k:a[k] for k in a.files})
        x=np.stack([c['features'] for c in cams],1);valid=np.stack([c['valid'] for c in cams],1).all(1)
        out=infer_stage_models(models,x,valid,saved['priors'],saved['temperature'])
        assert out['stage'].tolist()==curves[i]['stage'] and out['review'].tolist()==curves[i]['review']
        for key in ['confidence','progress']:
            reference=np.array([np.nan if v is None else v for v in curves[i][key]])
            np.testing.assert_allclose(out[key],reference,atol=2e-6,rtol=0,equal_nan=True)
            maximum=max(maximum,float(np.nanmax(abs(out[key]-reference))))
        checked.append(row['episode'])
    assert len(checked)==5
    result=dict(passed=True,archive=str(archive),archive_sha256=delivery['sha256'],unpacked_at=str(root),
        checkpoint_loaded_from_package=True,backbone_loaded_from_package=True,feature_cache_loaded_from_package=True,
        original_mcap_used=False,test_recordings=checked,max_abs_error=maximum,classes_and_review_flags_exact=True)
    (ROOT/'delivery_offline_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    archive.with_suffix('.zip.offline-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
