"""Package reproducible research while explicitly retaining the deployed v3."""
from datetime import datetime
import hashlib,importlib.util,json,stat,xml.etree.ElementTree as ET,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STUDY=ROOT/'research/stage_v4_20260917';APP=ROOT/'robot-data-studio'
PARENT=ROOT/'HoloCurate-运动感知模型升级-20260917-211337.zip'
PARENT_SHA='463d0bf84d561966f3207030ebd787db1f939e59c2df184293503000787163ad'
def sha(data):return hashlib.sha256(data).hexdigest()
def encoded(x):return (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()

def main():
    spec=importlib.util.spec_from_file_location('handoff',ROOT/'research/warp_rm_20260916/build_handoff.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    assert sha(PARENT.read_bytes())==PARENT_SHA
    decision=json.loads((STUDY/'promotion.json').read_text());assert not decision['promoted']
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
    chosen=[p for p in profiles['models'] if p['id'] in ['open-dinov2-pilot','g1-stage-v1','g1-stage-v2','g1-stage-v3']]
    assert len(chosen)==4 and not any(p.get('stage_version')==4 for p in profiles['models'])
    payload['robot-data-studio/workspace/warp/models/profiles.json']=encoded(dict(version=1,models=chosen))
    for p in chosen:
        path=APP/'workspace/warp/models'/p['checkpoint'];assert sha(path.read_bytes())==p['checkpoint_sha256'];add(path)
    assert next(p for p in chosen if p['id']=='g1-stage-v3')['checkpoint_sha256']==decision['checkpoint_sha256']
    for folder in [STUDY,ROOT/'research/stage_v4_decoupled_20260917']:
        for p in helper.selected_files(folder):
            if p.name.startswith('delivery') or p.suffix=='.zip':continue
            assert p.suffix in {'.py','.md','.json','.log','.png','.svg','.pt','.npz'},p
            add(p)
    evidence=APP/'workspace/evidence/stages-v4'
    for rel in ['regression.xml','regression.log','node-tests.log']:add(evidence/rel)
    suites=ET.parse(evidence/'regression.xml').getroot().findall('testsuite')
    counts={key:sum(int(s.attrib[key]) for s in suites) for key in ['tests','failures','errors','skipped']}
    assert counts==dict(tests=116,failures=0,errors=0,skipped=0)
    node=(evidence/'node-tests.log').read_text();assert '# fail 0' in node
    ui={}
    for folder in ['v3-regression','warp-regression']:
        for name in ['report.json','progress-real-video.png']:add(evidence/folder/name)
        report=json.loads((evidence/folder/'report.json').read_text());assert report['passed'];ui[folder]=len(report['checks'])
    verify=json.loads((STUDY/'runtime_verification.json').read_text());assert verify['passed']
    replay=json.loads((STUDY/'reproduction_check.json').read_text());assert replay['passed']
    add(STUDY/'交接说明.md','交接说明.md')
    now=datetime.now().astimezone();target=ROOT/('HoloCurate-论文扩展验证-'+now.strftime('%Y%m%d-%H%M%S')+'.zip')
    assert not target.exists()
    manifest=dict(schema_version=5,built_at=now.isoformat(),parent_archive_sha256=PARENT_SHA,promoted=False,active_stage_model='g1-stage-v3',
        scope='Four research adaptations did not pass frozen gates. Preserve v3 classifier; include all negative results and experimental weights without registering v4.',
        models=[dict(id=p['id'],sha256=p['checkpoint_sha256']) for p in chosen],
        validation=dict(python_regression=counts,node_test_files=len(list((APP/'tests').glob('*.test.mjs'))),browser_checks=ui,runtime=verify,experiments=replay),
        exclusions=['raw MCAP','virtual environments','browser profiles','Hugging Face login cache','review database','paper PDFs'],
        files=[dict(path=n,bytes=len(d),sha256=sha(d)) for n,d in sorted(payload.items())])
    payload['ARTIFACT_MANIFEST.json']=encoded(manifest);prefix=target.stem+'/'
    partial=target.with_suffix('.zip.partial');assert not partial.exists()
    with zipfile.ZipFile(partial,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,d in sorted(payload.items()):
            info=zipfile.ZipInfo(prefix+n,now.timetuple()[:6]);info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(stat.S_IFREG|(0o755 if n.endswith('.sh') else 0o644))<<16;z.writestr(info,d)
    with zipfile.ZipFile(partial) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(payload)
        for item in manifest['files']:assert sha(z.read(prefix+item['path']))==item['sha256']
    for p,digest in disk.items():assert sha(Path(p).read_bytes())==digest
    partial.rename(target)
    result=dict(passed=True,path=str(target),sha256=sha(target.read_bytes()),bytes=target.stat().st_size,
        verified_files=len(manifest['files']),promoted=False,models=[p['id'] for p in chosen],source_changes=audit)
    target.with_suffix('.zip.sha256').write_text(result['sha256']+'  '+target.name+'\n')
    target.with_suffix('.zip.verification.json').write_bytes(encoded(result));(STUDY/'delivery.json').write_bytes(encoded(result))
    print(json.dumps({k:v for k,v in result.items() if k!='source_changes'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
