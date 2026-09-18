"""Download only the pinned, public evaluation traces; no model or private data."""
from pathlib import Path
import hashlib
import json
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

ROOT=Path(__file__).resolve().parent
REVISIONS={'n128':'b5ac676cf5dc23ea7d423d3eee89c9caa82252f9',
           'n512':'d3b63a144070f4c24ffe83b66e5cc1bef8dbefa8'}
disable_progress_bars()
for suite,revision in REVISIONS.items():
    output=ROOT/'traces'/suite
    print('Downloading public '+suite+' traces',flush=True)
    snapshot_download('uynitsuj/paper-sim-'+suite+'-traces',repo_type='dataset',revision=revision,
                      local_dir=output,token=False,max_workers=4,
                      allow_patterns=['fullhz*/qpos_trace*.npz','MANIFEST.json','README.md'])
    files=sorted(output.glob('fullhz*/qpos_trace*.npz'))
    digest_lines=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(output).as_posix()+'\n' for p in files)
    sha=hashlib.sha256(digest_lines.encode()).hexdigest()
    expected_count=256 if suite=='n128' else 1024
    if len(files)!=expected_count:raise ValueError('Incomplete trace download')
    result=dict(repo='uynitsuj/paper-sim-'+suite+'-traces',revision=revision,files=len(files),
                size_bytes=sum(p.stat().st_size for p in files),tree_sha256=sha)
    if suite=='n128':
        expected=json.loads((output/'MANIFEST.json').read_text())['content']['tree_sha256']
        result['manifest_tree_sha256']=expected
        result['manifest_tree_matches']=sha==expected
    (output/'local_download_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
