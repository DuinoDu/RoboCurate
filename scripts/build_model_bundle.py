#!/usr/bin/env python3
"""Build the four registered DINOv2 model assets; never include caches or videos."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

APP = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build(model_root, output, tag):
    profiles = json.loads((APP / 'models/profiles.json').read_text())
    entries = {}
    for model in profiles['models']:
        names = {model['checkpoint']: model['checkpoint_sha256']}
        names.update({model['backbone_dir'] + '/' + name: sha
                      for name, sha in model['backbone_sha256'].items()})
        for name, expected in names.items():
            source = model_root / name
            if digest(source) != expected:
                raise ValueError('Model integrity mismatch: ' + name)
            entries[name] = (source, expected)
    # This provenance file is needed by training and feature extraction tools.
    for name in ['dinov2-small-provenance.json']:
        source = model_root / name
        entries[name] = (source, digest(source))
    entries['licenses/dinov2.txt'] = (APP / 'static/licenses/dinov2.txt',
                                    digest(APP / 'static/licenses/dinov2.txt'))
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'RoboCurate-{tag}-models.zip'
    manifest = dict(version=1, release=tag, models=[m['id'] for m in profiles['models']],
                    files=[dict(path=name, size=source.stat().st_size, sha256=sha)
                           for name, (source, sha) in sorted(entries.items())])
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, (source, _) in sorted(entries.items()):
            archive.write(source, name)
        archive.writestr('MODEL_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    sha = digest(path)
    path.with_suffix('.zip.sha256').write_text(f'{sha}  {path.name}\n')
    metadata = dict(version=1, release=tag,
                    url=f'https://github.com/Lee-hz/RoboCurate/releases/download/{tag}/{path.name}',
                    size=path.stat().st_size, sha256=sha,
                    profiles_sha256=digest(APP / 'models/profiles.json'), manifest=manifest)
    target = output / 'model-release.json'
    target.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(archive=str(path), metadata=str(target),
                          bytes=path.stat().st_size, sha256=sha), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag', default='v0.1.0-preview.3')
    args = parser.parse_args()
    if not all(c.isalnum() or c in '._-' for c in args.tag) or not args.tag:
        parser.error('Invalid release tag')
    build(args.model_root.resolve(), args.output.resolve(), args.tag)
