"""Overlay the verified interface upgrade on the preceding full handoff."""
from datetime import datetime
import hashlib,importlib.util,json,stat,xml.etree.ElementTree as ET,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];STUDY=Path(__file__).resolve().parent
APP=ROOT/'robot-data-studio';EVIDENCE=APP/'workspace/evidence/interaction'
PARENT=ROOT/'HoloCurate-论文扩展验证-20260917-220126.zip'
PARENT_SHA='3e9c3eb24937a2257e7c9ce6423df85620101f9effe658080e67067685447e1b'
def sha(data):return hashlib.sha256(data).hexdigest()
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()

def main():
    spec=importlib.util.spec_from_file_location('handoff',ROOT/'research/warp_rm_20260916/build_handoff.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    assert sha(PARENT.read_bytes())==PARENT_SHA
    payload={};disk={}
    with zipfile.ZipFile(PARENT) as archive:
        prefix=archive.namelist()[0].split('/')[0]+'/'
        old_manifest=json.loads(archive.read(prefix+'ARTIFACT_MANIFEST.json'))
        for item in old_manifest['files']:
            data=archive.read(prefix+item['path']);assert sha(data)==item['sha256']
            payload[item['path']]=data
    previous={name:sha(data) for name,data in payload.items() if name.startswith('robot-data-studio/') and not name.startswith('robot-data-studio/workspace/')}
    def add(path,name=None):
        path=Path(path);assert path.is_file() and not path.is_symlink()
        name=name or path.relative_to(ROOT).as_posix()
        assert not name.startswith('/') and '..' not in Path(name).parts
        data=path.read_bytes()
        if path.suffix in helper.TEXT_SUFFIXES:
            assert not any(pattern.search(data) for pattern in helper.SECRET_PATTERNS),'Credential-like content: inspect locally'
        payload[name]=data;disk[str(path)]=sha(data)
    source=list(helper.selected_files(APP))
    for path in source:add(path)
    current={path.relative_to(ROOT).as_posix():sha(path.read_bytes()) for path in source}
    assert set(previous)<=set(current)
    changed=[dict(path=name.removeprefix('robot-data-studio/'),before_sha256=previous[name],after_sha256=current[name])
             for name in sorted(previous) if current[name]!=previous[name]]
    runtime=json.loads((STUDY/'runtime_verification.json').read_text());assert runtime['passed']
    assert not {x['path'] for x in changed}&set(runtime['protected_sources'])
    audit=dict(parent_archive=PARENT.name,parent_sha256=PARENT_SHA,changed_existing=changed,
        added=[name.removeprefix('robot-data-studio/') for name in sorted(set(current)-set(previous))],
        unrelated_source_preserved=runtime['protected_sources'])
    (STUDY/'source_changes.json').write_bytes(encoded(audit))
    for path in helper.selected_files(STUDY):
        if not path.name.startswith('delivery'):add(path)
    for name in ['regression.xml','regression.log','node-tests.log','report.json','task-overview-dark.png',
                 'review-queue-dark.png','stage-evidence-dark.png','stage-evidence-light.png',
                 'stage-evidence-tablet.png','stage-evidence-phone.png',
                 'v3-regression/report.json','v3-regression/progress-real-video.png']:
        add(EVIDENCE/name)
    for name in ['ui-report.json','progress-dark-test-fixture.png','progress-light-test-fixture.png','progress-export-test-fixture.png']:
        add(APP/'workspace/evidence/warp'/name)
    suites=ET.parse(EVIDENCE/'regression.xml').getroot().findall('testsuite')
    counts={key:sum(int(s.attrib[key]) for s in suites) for key in ['tests','failures','errors','skipped']}
    assert counts==dict(tests=122,failures=0,errors=0,skipped=0)
    assert '# fail 0' in (EVIDENCE/'node-tests.log').read_text()
    ui={}
    for key,path in [('task_interaction',EVIDENCE/'report.json'),('stage_regression',EVIDENCE/'v3-regression/report.json'),
                     ('isolated_warp_export',APP/'workspace/evidence/warp/ui-report.json')]:
        report=json.loads(path.read_text());assert report['passed'];ui[key]=len(report['checks'])
    # The parent ships four usable profiles. Its DINOv3-dependent local research profile remains excluded.
    profiles=json.loads(payload['robot-data-studio/workspace/warp/models/profiles.json'])['models']
    for model in profiles:
        checkpoint='robot-data-studio/workspace/warp/models/'+model['checkpoint']
        assert sha(payload[checkpoint])==model['checkpoint_sha256']
    payload['交接说明.md']=('# HoloCurate 任务审核界面升级\n\n'
        '本版新增阶段导航、当前模型证据、优先复核队列和限时回看，与视频及审核草稿同步。保留现有 v3 模型与 WARP 导出流程。\n\n'
        '- [操作说明与界面截图](robot-data-studio/INTERACTION_UPGRADE.md)\n'
        '- [本次改动及验证说明](research/interaction_20260917/交接说明.md)\n'
        '- [项目启动说明](robot-data-studio/README.md)\n\n'
        '这是基于上版完整交接包的界面升级，保留历史模型研究材料。此次没有新增训练或模型准确率提升声明。\n').encode()
    now=datetime.now().astimezone();target=ROOT/('HoloCurate-任务审核界面升级-'+now.strftime('%Y%m%d-%H%M%S')+'.zip')
    assert not target.exists()
    manifest=dict(schema_version=6,built_at=now.isoformat(),parent_archive_sha256=PARENT_SHA,
        active_stage_model='g1-stage-v3',model_changed=False,
        scope='Task-stage evidence, synchronized navigation and draft-only review interaction; retain existing models and WARP exports.',
        models=[dict(id=m['id'],sha256=m['checkpoint_sha256']) for m in profiles],
        validation=dict(python_regression=counts,node_test_files=8,browser_checks=ui,display_verification=runtime),
        exclusions=['raw MCAP','virtual environments','browser profiles','login cache','review database'],
        files=[dict(path=name,bytes=len(data),sha256=sha(data)) for name,data in sorted(payload.items())])
    payload['ARTIFACT_MANIFEST.json']=encoded(manifest);prefix=target.stem+'/'
    partial=target.with_suffix('.zip.partial');assert not partial.exists()
    with zipfile.ZipFile(partial,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(prefix+name,now.timetuple()[:6]);info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(stat.S_IFREG|(0o755 if name.endswith('.sh') else 0o644))<<16
            archive.writestr(info,data)
    with zipfile.ZipFile(partial) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist())==len(set(archive.namelist()))==len(payload)
        for item in manifest['files']:assert sha(archive.read(prefix+item['path']))==item['sha256']
    for path,digest in disk.items():assert sha(Path(path).read_bytes())==digest
    partial.rename(target)
    report=dict(passed=True,path=str(target),sha256=sha(target.read_bytes()),bytes=target.stat().st_size,
        verified_files=len(manifest['files']),active_stage_model='g1-stage-v3',model_changed=False,
        python_regression=counts,browser_checks=ui,source_changes=audit)
    target.with_suffix('.zip.sha256').write_text(report['sha256']+'  '+target.name+'\n')
    target.with_suffix('.zip.verification.json').write_bytes(encoded(report))
    (STUDY/'delivery.json').write_bytes(encoded(report))
    print(json.dumps({k:v for k,v in report.items() if k!='source_changes'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
