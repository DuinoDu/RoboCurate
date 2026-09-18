"""Create a verified, versioned v2 handoff from explicit source/evidence allowlists."""
from datetime import datetime
import hashlib,importlib.util,json,stat,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STUDY=ROOT/'research/stage_v2_20260917';APP=ROOT/'robot-data-studio'
PARENT=ROOT/'HoloCurate-阶段模型升级-20260917-124635.zip'
PARENT_SHA='b98638f0524ea7f6fdf519c2d3b4f80eb858aae8339dd69255b7e2ce03a81d64'
def sha(data):return hashlib.sha256(data).hexdigest()
def encoded(x):return (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()

def main():
    spec=importlib.util.spec_from_file_location('handoff',ROOT/'research/warp_rm_20260916/build_handoff.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    assert sha(PARENT.read_bytes())==PARENT_SHA
    payload={};disk={}
    with zipfile.ZipFile(PARENT) as z:
        prefix=z.namelist()[0].split('/')[0]+'/'
        for name in z.namelist():
            rel=name[len(prefix):]
            if rel!='ARTIFACT_MANIFEST.json':payload[rel]=z.read(name)
    previous={n:sha(d) for n,d in payload.items() if n.startswith('robot-data-studio/') and not n.startswith('robot-data-studio/workspace/')}
    def add(p,name=None):
        p=Path(p);assert p.is_file() and not p.is_symlink()
        name=name or p.relative_to(ROOT).as_posix()
        assert not name.startswith('/') and '..' not in Path(name).parts
        data=p.read_bytes()
        if p.suffix in helper.TEXT_SUFFIXES:
            assert not any(pattern.search(data) for pattern in helper.SECRET_PATTERNS),'Credential-like content: inspect locally'
        payload[name]=data;disk[str(p)]=sha(data)
    source=list(helper.selected_files(APP))
    for p in source:add(p)
    current={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in source}
    assert set(previous)<=set(current)
    changed=[dict(path=n.removeprefix('robot-data-studio/'),before_sha256=previous[n],after_sha256=current[n])
             for n in sorted(previous) if previous[n]!=current[n]]
    preserved=['static/solid-canvas.js','scripts/build_preview_meshes.py','static/viewer/g1-preview.json',
               'static/viewer/trails.js','static/robot3d.js','static/workspace.css']
    assert not set(preserved)&{c['path'] for c in changed}
    audit=dict(baseline_archive=PARENT.name,baseline_archive_sha256=PARENT_SHA,changed_existing=changed,
        added=[n.removeprefix('robot-data-studio/') for n in sorted(set(current)-set(previous))],unrelated_source_preserved=preserved)
    (STUDY/'source_changes.json').write_bytes(encoded(audit))
    profiles=json.loads((APP/'workspace/warp/models/profiles.json').read_text())
    chosen=[p for p in profiles['models'] if p['id'] in ['open-dinov2-pilot','g1-stage-v1','g1-stage-v2']]
    assert len(chosen)==3
    payload['robot-data-studio/workspace/warp/models/profiles.json']=encoded(dict(version=1,models=chosen))
    for p in chosen:
        path=APP/'workspace/warp/models'/p['checkpoint'];assert sha(path.read_bytes())==p['checkpoint_sha256'];add(path)
    for p in sorted(STUDY.iterdir()):
        if p.is_file() and p.suffix in {'.py','.md','.json','.log','.png','.svg','.zip'} and not p.name.startswith('delivery'):
            add(p)
    for folder in ['training','states']:
        for p in helper.selected_files(STUDY/folder):add(p)
    local=json.loads((STUDY/'local_stage_report.json').read_text());assert local['state']=='complete' and len(local['episodes'])==18
    for row in local['episodes']:
        p=APP/'workspace/warp/scores'/(row['signature']+'.json');add(p);add(p.with_suffix('.npz'))
    evidence=APP/'workspace/evidence/stages-v2'
    for rel in ['regression.xml','regression.log','node-tests.log']:
        add(evidence/rel)
    ui={}
    for folder in ['live','v1-regression','warp-regression']:
        for name in ['report.json','progress-real-video.png']:add(evidence/folder/name)
        report=json.loads((evidence/folder/'report.json').read_text());assert report['passed'];ui[folder]=len(report['checks'])
    verify=json.loads((STUDY/'runtime_verification.json').read_text());assert verify['passed']
    add(STUDY/'交接说明.md','交接说明.md')
    now=datetime.now().astimezone();target=ROOT/('HoloCurate-多模态时序升级-'+now.strftime('%Y%m%d-%H%M%S')+'.zip')
    assert not target.exists()
    manifest=dict(schema_version=3,built_at=now.isoformat(),parent_archive_sha256=PARENT_SHA,
        scope='Task-specific multimodal stage research model; same-session CV, historical regression; no verified final-success or policy-improvement claim',
        models=[dict(id=p['id'],sha256=p['checkpoint_sha256']) for p in chosen],
        validation=dict(python_regression=102,node_test_files=8,browser_checks=ui,runtime=verify),
        exclusions=['raw MCAP','virtual environments','browser profiles','Hugging Face login cache','review database','paper PDFs'],
        files=[dict(path=n,bytes=len(d),sha256=sha(d)) for n,d in sorted(payload.items())])
    payload['ARTIFACT_MANIFEST.json']=encoded(manifest);prefix=target.stem+'/'
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,d in sorted(payload.items()):
            info=zipfile.ZipInfo(prefix+n,now.timetuple()[:6]);info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(stat.S_IFREG|(0o755 if n.endswith('.sh') else 0o644))<<16;z.writestr(info,d)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(payload)
        for item in manifest['files']:assert sha(z.read(prefix+item['path']))==item['sha256']
    for p,digest in disk.items():assert sha(Path(p).read_bytes())==digest
    result=dict(passed=True,path=str(target),sha256=sha(target.read_bytes()),bytes=target.stat().st_size,
        verified_files=len(manifest['files']),models=[p['id'] for p in chosen],source_changes=audit)
    target.with_suffix('.zip.sha256').write_text(result['sha256']+'  '+target.name+'\n')
    target.with_suffix('.zip.verification.json').write_bytes(encoded(result));(STUDY/'delivery.json').write_bytes(encoded(result))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
