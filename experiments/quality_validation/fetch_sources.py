"""Restore reviewed upstream snapshots by immutable Git blob SHA."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

HERE=Path(__file__).resolve().parent

def main():
    sources=json.loads((HERE/'sources.json').read_text());root=HERE.parents[1]/'workspace/references/snapshots'
    def one(item):
        target=root/item['repo'].split('/')[-1]/item['path']
        def correct(raw):return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==item['git_blob_sha']
        if target.exists() and correct(target.read_bytes()):return
        request=urllib.request.Request(item['source'],headers={'User-Agent':'RoboCurate-quality-validation'})
        with urllib.request.urlopen(request,timeout=30) as response:data=json.load(response)
        raw=base64.b64decode(data['content']);assert correct(raw)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        print('VERIFIED',item['repo'],item['path'],flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,sources))

if __name__=='__main__':main()
