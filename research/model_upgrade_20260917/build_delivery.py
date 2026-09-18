"""Build a verified local handoff from source and an explicit evidence/model whitelist."""
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import zipfile
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'research/model_upgrade_20260917'
APP=ROOT/'robot-data-studio'
OLD=ROOT/'HoloCurate-WARP方法融合-20260917-002505.zip'

def sha(data):return hashlib.sha256(data).hexdigest()
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()

def main():
    spec=importlib.util.spec_from_file_location('old_delivery',ROOT/'research/warp_rm_20260916/build_handoff.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    expected='c89d93605db3eb3267a44e8b0f7012e381a0350267f56cb0b5b82f7abe9c6970'
    assert sha(OLD.read_bytes())==expected
    payload={};disk={}
    with zipfile.ZipFile(OLD) as z:
        prefix=z.namelist()[0].split('/')[0]+'/'
        for name in z.namelist():
            relative=name[len(prefix):]
            if relative!='ARTIFACT_MANIFEST.json':payload[relative]=z.read(name)
    def add(p,name=None):
        p=Path(p);assert p.is_file() and not p.is_symlink()
        name=name or p.relative_to(ROOT).as_posix()
        assert not name.startswith('/') and '..' not in Path(name).parts
        data=p.read_bytes()
        if p.suffix in helper.TEXT_SUFFIXES:
            assert not any(pattern.search(data) for pattern in helper.SECRET_PATTERNS), 'Credential-like content; inspect locally'
        payload[name]=data;disk[str(p)]=sha(data)
    for p in helper.selected_files(APP):add(p)
    # Add only the two installed open-weight profiles. No unavailable DINOv3 or credentials.
    profiles=json.loads((APP/'workspace/warp/models/profiles.json').read_text())
    chosen=[p for p in profiles['models'] if p['id'] in ['open-dinov2-pilot','g1-stage-v1']]
    assert len(chosen)==2
    payload['robot-data-studio/workspace/warp/models/profiles.json']=encoded(dict(version=1,models=chosen))
    for p in chosen:add(APP/'workspace/warp/models'/p['checkpoint'])
    for p in sorted(STUDY.iterdir()):
        if p.is_file() and p.suffix in {'.py','.md','.json','.log'} and not p.name.startswith('delivery'):
            add(p)
    for folder in ['stage_training','local_evidence']:
        for p in helper.selected_files(STUDY/folder):add(p)
    features=json.loads((STUDY/'features_manifest.json').read_text())
    for row in features['episodes']:
        for camera in row['cameras'].values():
            p=Path(camera['cache']);assert sha(p.read_bytes())==camera['cache_sha256']
            add(p);add(p.with_suffix('.json'))
    scores=json.loads((STUDY/'local_stage_report.json').read_text())
    assert scores['state']=='complete' and len(scores['episodes'])==18
    for row in scores['episodes']:
        p=APP/'workspace/warp/scores'/(row['signature']+'.json')
        add(p);add(p.with_suffix('.npz'))
    for rel in ['regression.xml','regression.log','unit.xml','node-tests.log','live/report.json','live/progress-real-video.png',
                'warp-regression/report.json','warp-regression/progress-real-video.png']:
        add(APP/'workspace/evidence/stages'/rel)
    verify=json.loads((STUDY/'runtime_verification.json').read_text());assert verify['passed']
    # Track exactly which application files changed since this iteration began.
    with zipfile.ZipFile(STUDY/'pre_upgrade_sources.zip') as z:
        changed=[]
        for name in z.namelist():
            before=sha(z.read(name));p=APP/name;after=sha(p.read_bytes())
            if before!=after:changed.append(dict(path=name,before_sha256=before,after_sha256=after))
    audit=dict(changed_existing_source=changed,unrelated_UI_and_3D_sources_preserved=True,
        rollback_source_archive='pre_upgrade_sources.zip',rollback_source_sha256=sha((STUDY/'pre_upgrade_sources.zip').read_bytes()))
    for name in ['static/solid-canvas.js','scripts/build_preview_meshes.py','static/viewer/trails.js','static/robot3d.js','static/workspace.css']:
        assert name not in [c['path'] for c in changed]
    (STUDY/'source_changes.json').write_bytes(encoded(audit));add(STUDY/'source_changes.json')
    add(STUDY/'交接说明.md','交接说明.md')
    now=datetime.now().astimezone()
    target=ROOT/('HoloCurate-阶段模型升级-'+now.strftime('%Y%m%d-%H%M%S')+'.zip')
    assert not target.exists()
    manifest=dict(schema_version=2,built_at=now.isoformat(),parent_archive_sha256=expected,
        scope='Existing WARP method adaptation plus task-specific provisional-label stage model; no verified task-success or policy improvement claim',
        validation=dict(python_regression=97,node_test_files=8,real_stage_ui_checks=15,real_warp_ui_checks=10,runtime=verify),
        models=[dict(id=p['id'],sha256=p['checkpoint_sha256']) for p in chosen],
        exclusions=['raw MCAP','virtual environments','browser profiles','Hugging Face login cache','review database','paper PDFs'],
        files=[dict(path=n,bytes=len(d),sha256=sha(d)) for n,d in sorted(payload.items())])
    payload['ARTIFACT_MANIFEST.json']=encoded(manifest)
    prefix=target.stem+'/'
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,d in sorted(payload.items()):
            info=zipfile.ZipInfo(prefix+n,now.timetuple()[:6]);info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(stat.S_IFREG|(0o755 if n.endswith('.sh') else 0o644))<<16
            z.writestr(info,d)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(payload)
        for item in manifest['files']:assert sha(z.read(prefix+item['path']))==item['sha256']
    for path,digest in disk.items():assert sha(Path(path).read_bytes())==digest
    result=dict(passed=True,path=str(target),bytes=target.stat().st_size,sha256=sha(target.read_bytes()),
        verified_files=len(manifest['files']),models=[p['id'] for p in chosen],source_changes=audit)
    target.with_suffix('.zip.sha256').write_text(result['sha256']+'  '+target.name+'\n')
    target.with_suffix('.zip.verification.json').write_bytes(encoded(result))
    (STUDY/'delivery.json').write_bytes(encoded(result))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
