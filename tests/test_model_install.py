"""Release installation checks: corruption and conflicts must not replace models."""
import hashlib
import json
import zipfile

import pytest
from scripts.install_models import install


def bundle(tmp_path, extra=None):
    contents = {'head.pt': b'example checkpoint', 'backbone/config.json': b'{}',
                'backbone/model.safetensors': b'example backbone'}
    rows = [dict(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest())
            for name, data in contents.items()]
    hashes = {r['path']: r['sha256'] for r in rows}
    profile = dict(id='example', checkpoint='head.pt', checkpoint_sha256=hashes['head.pt'],
                   backbone_dir='backbone', backbone_sha256={name: hashes['backbone/' + name]
                   for name in ['config.json', 'model.safetensors']})
    profiles = dict(version=1, models=[profile])
    manifest = dict(version=1, release='test', models=['example'], files=rows)
    archive = tmp_path / 'models.zip'
    with zipfile.ZipFile(archive, 'w') as package:
        for name, data in contents.items():
            package.writestr(name, data)
        package.writestr('MODEL_MANIFEST.json', json.dumps(manifest))
        if extra:
            package.writestr(extra, b'unexpected')
    metadata = dict(size=archive.stat().st_size, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    manifest=manifest)
    return archive, metadata, profiles


def test_install_is_idempotent_and_preserves_other_models(tmp_path):
    archive, metadata, profiles = bundle(tmp_path)
    workspace = tmp_path / 'workspace'
    root = workspace / 'warp/models'
    root.mkdir(parents=True)
    (root / 'profiles.json').write_text(json.dumps(dict(version=1, models=[dict(id='local')])))
    install(archive, workspace, metadata, profiles)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    install(archive, workspace, metadata, profiles)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert [p['id'] for p in json.loads((root / 'profiles.json').read_text())['models']] == ['local', 'example']


def test_corrupt_archive_installs_nothing(tmp_path):
    archive, metadata, profiles = bundle(tmp_path)
    archive.write_bytes(archive.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='SHA-256/size'):
        install(archive, tmp_path / 'workspace', metadata, profiles)
    assert not (tmp_path / 'workspace').exists()


def test_existing_different_profile_is_preserved(tmp_path):
    archive, metadata, profiles = bundle(tmp_path)
    root = tmp_path / 'workspace/warp/models'
    root.mkdir(parents=True)
    original = json.dumps(dict(version=1, models=[dict(id='example', checkpoint='personal.pt')]))
    (root / 'profiles.json').write_text(original)
    with pytest.raises(ValueError, match='Existing profile differs'):
        install(archive, tmp_path / 'workspace', metadata, profiles)
    assert (root / 'profiles.json').read_text() == original
    assert not (root / 'head.pt').exists()


@pytest.mark.parametrize('extra', ['../escaped.pt', 'unlisted.pt'])
def test_unlisted_archive_members_are_rejected_before_install(tmp_path, extra):
    archive, metadata, profiles = bundle(tmp_path, extra)
    with pytest.raises(ValueError, match='Unexpected'):
        install(archive, tmp_path / 'workspace', metadata, profiles)
    assert not (tmp_path / 'workspace/warp/models/head.pt').exists()
    assert not (tmp_path / 'workspace/warp/escaped.pt').exists()


def test_member_checksum_failure_installs_nothing(tmp_path):
    archive, metadata, profiles = bundle(tmp_path)
    # Simulate internally inconsistent release metadata with an intact ZIP.
    metadata['manifest']['files'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Checkpoint/profile mismatch'):
        install(archive, tmp_path / 'workspace', metadata, profiles)
    assert not (tmp_path / 'workspace/warp/models/head.pt').exists()
