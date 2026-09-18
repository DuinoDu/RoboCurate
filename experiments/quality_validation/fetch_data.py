"""Download only the two pinned public benchmarks; verify upstream LFS hashes."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE.parents[1]/'workspace/public-quality-data');parser.add_argument('--proxy');args=parser.parse_args()
    spec=json.loads((HERE/'datasets.json').read_text());args.output.mkdir(parents=True,exist_ok=True)
    def download(item):
        target=args.output/item['name']
        def valid(path):
            if not path.exists() or path.stat().st_size != item['bytes']:return False
            with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(f.read()).hexdigest()
            return digest==item['sha256']
        if not valid(target):
            temp=target.with_suffix('.partial')
            url=f"https://huggingface.co/datasets/{spec['repository']}/resolve/{spec['revision']}/{item['path']}?download=true"
            command=['curl','-q','-fL','--connect-timeout','15','--max-time','240','--retry','1']
            if args.proxy:command+=['--proxy',args.proxy]
            command += [url,'-o',str(temp)]
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            if not valid(temp):raise ValueError('Source checksum mismatch: '+item['name'])
            temp.replace(target)
        print('VERIFIED',target,item['sha256'],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(download,spec['files']))
    (args.output/'provenance.json').write_text(json.dumps(spec,indent=2))

if __name__=='__main__':main()
