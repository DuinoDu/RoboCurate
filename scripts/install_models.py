#!/usr/bin/env python3
"""Download and verify the published DINOv2 + trained model pack, or install offline."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile

APP = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False,
                                     prefix='.profiles-', suffix='.tmp') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install(archive_path, workspace, metadata, profiles):
    """Validate everything before installing; refuse to replace different models."""
    archive_path = Path(archive_path)
    if archive_path.stat().st_size != metadata['size'] or digest(archive_path) != metadata['sha256']:
        raise ValueError('Model archive SHA-256/size mismatch; no models installed')
    expected = {r['path']: r for r in metadata['manifest']['files']}
    if len(expected) != len(metadata['manifest']['files']):
        raise ValueError('Duplicate model manifest entries')
    for name in expected:
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or str(p) != name:
            raise ValueError('Unsafe model path')
    root = Path(workspace).resolve() / 'warp/models'
    root.mkdir(parents=True, exist_ok=True)
    profile_path = root / 'profiles.json'
    if profile_path.is_symlink():
        raise ValueError('Refusing a symlinked profiles.json')
    current = read_json(profile_path) if profile_path.exists() else dict(version=1, models=[])
    current_ids = [p['id'] for p in current['models']]
    incoming_ids = [p['id'] for p in profiles['models']]
    if len(set(current_ids)) != len(current_ids) or len(set(incoming_ids)) != len(incoming_ids):
        raise ValueError('Duplicate model IDs')
    merged = {p['id']: p for p in current['models']}
    for profile in profiles['models']:
        old = merged.get(profile['id'])
        if old is not None and old != profile:
            raise ValueError('Existing profile differs: ' + profile['id'] + '; use a separate workspace')
        merged[profile['id']] = profile
        row = expected.get(profile['checkpoint'])
        if not row or row['sha256'] != profile['checkpoint_sha256']:
            raise ValueError('Checkpoint/profile mismatch')
        for name, sha in profile['backbone_sha256'].items():
            if expected.get(profile['backbone_dir'] + '/' + name, {}).get('sha256') != sha:
                raise ValueError('Backbone/profile mismatch')
    for name, row in expected.items():
        target = root / name
        if any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
            raise ValueError('Refusing symlinked model destination: ' + name)
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('Model destination escapes workspace')
        if target.exists() and (not target.is_file() or digest(target) != row['sha256']):
            raise ValueError('Existing model file differs: ' + name + '; use a separate workspace')
    # Temporary extraction is limited to the trusted manifest; no extractall.
    with tempfile.TemporaryDirectory(prefix='.install-', dir=root) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(expected) | {'MODEL_MANIFEST.json'}:
                raise ValueError('Unexpected or duplicate archive members')
            if json.loads(archive.read('MODEL_MANIFEST.json')) != metadata['manifest']:
                raise ValueError('Model manifest mismatch')
            for name, row in expected.items():
                info = archive.getinfo(name)
                if info.file_size != row['size'] or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('Invalid model member: ' + name)
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open('wb') as sink:
                    shutil.copyfileobj(source, sink)
                if digest(target) != row['sha256']:
                    raise ValueError('Model member checksum mismatch: ' + name)
        for name in expected:
            target = root / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / name, target)
        atomic_json(profile_path, dict(current, models=list(merged.values())))
    return dict(installed=incoming_ids, workspace=str(Path(workspace).resolve()),
                archive_sha256=metadata['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=APP / 'workspace')
    parser.add_argument('--archive', type=Path, help='Offline ZIP downloaded from Releases')
    args = parser.parse_args()
    metadata = read_json(APP / 'models/release.json')
    profile_path = APP / 'models/profiles.json'
    if digest(profile_path) != metadata['profiles_sha256']:
        raise ValueError('Source profile checksum mismatch')
    profiles = read_json(profile_path)
    with tempfile.TemporaryDirectory(prefix='robocurate-models-') as temporary:
        path = args.archive
        if path is None:
            path = Path(temporary) / 'models.zip'
            print(f"Downloading {metadata['size'] / 1e6:.1f} MB from GitHub Releases", flush=True)
            request = urllib.request.Request(metadata['url'], headers={'User-Agent': 'RoboCurate-model-installer'})
            with urllib.request.urlopen(request, timeout=60) as source, path.open('wb') as sink:
                total = 0
                while block := source.read(1024 * 1024):
                    total += len(block)
                    if total > metadata['size']:
                        raise ValueError('Download exceeds expected archive size')
                    sink.write(block)
        print(json.dumps(install(path, args.workspace, metadata, profiles), ensure_ascii=False, indent=2))
        print('Model files installed. Install requirements-warp.txt in workspace/warp-env before scoring.')


if __name__ == '__main__':
    main()
