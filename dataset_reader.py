"""Small, torch-compatible reader for an extracted RoboCurate bundle.

No torch dependency is needed; torch.utils.data.DataLoader can wrap this class.
Sequence windows never cross clip boundaries or invalid rows.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


class RobotDataset:
    def __init__(self, root, split='train', sequence_length=1, images=False,
                 image_size=(224,224), head_view='full', curation=None):
        self.root=Path(root).resolve()
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        if self.manifest.get('kind')!='dataset':raise ValueError('This is a manifest-only export, not a training bundle')
        if split not in ('train','validation','all'):raise ValueError('Unknown split')
        if not isinstance(sequence_length,int) or sequence_length<1:raise ValueError('sequence_length must be a positive integer')
        if head_view not in ('full','left','right'):raise ValueError('Unknown head_view')
        if curation not in (None,'warp'):raise ValueError('Unknown curation mode')
        if curation=='warp':
            policy=self.manifest.get('options',{}).get('progress') or {}
            if policy.get('horizon')!=sequence_length:
                raise ValueError('sequence_length must match the exported WARP action horizon')
        self.curation=curation
        self.sequence_length=sequence_length;self.images=images;self.image_size=image_size;self.head_view=head_view
        self.index=[];self.clips=[];self.cache=OrderedDict()
        for ep in self.manifest['episodes']:
            if split!='all' and ep['split']!=split:continue
            for clip in ep['clips']:
                folder=self.safe_path(clip['path'])
                with np.load(folder/'samples.npz',allow_pickle=False) as a:
                    valid=a['valid']; windows=np.convolve(valid.astype(np.int64),np.ones(sequence_length,dtype=np.int64),mode='valid') if len(valid)>=sequence_length else []
                    starts=np.flatnonzero(np.asarray(windows)==sequence_length)
                    if curation=='warp':
                        weights=a['warp.weight'];eligible=a['warp.eligible']
                        starts=starts[eligible[starts] & np.isfinite(weights[starts]) & (weights[starts]>0)]
                ci=len(self.clips);self.clips.append((folder,ep))
                self.index.extend((ci,int(i)) for i in starts)

    def safe_path(self,relative):
        path=(self.root/relative).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Dataset path escapes its root')
        return path

    def __len__(self):return len(self.index)

    def _clip(self,ci):
        if ci not in self.cache:
            folder,ep=self.clips[ci]
            with np.load(folder/'samples.npz',allow_pickle=False) as a:
                arr={k:a[k] for k in ['timestamp_ns','timestamp_s','observation.state','action']}
                if self.curation=='warp':arr['warp.weight']=a['warp.weight']
            rows=[json.loads(line) for line in (folder/'images.jsonl').read_text().splitlines()]
            self.cache[ci]=(arr,rows)
            while len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(ci)
        return self.cache[ci]

    def __getitem__(self,index):
        ci,start=self.index[index];folder,ep=self.clips[ci];arr,rows=self._clip(ci)
        select=start if self.sequence_length==1 else slice(start,start+self.sequence_length)
        out={k:v[select].copy() for k,v in arr.items() if k!='warp.weight'}
        if self.curation=='warp':out['sample_weight']=np.float32(arr['warp.weight'][start])
        out['task']=ep['instruction'];out['episode_id']=ep['id']
        if self.images:
            for cam in rows[start]['images']:
                frames=[]
                for i in range(start,start+self.sequence_length):
                    relative=rows[i]['images'][cam]
                    if relative is None:raise ValueError('valid sample unexpectedly lacks an image')
                    path=self.safe_path(str((folder/self.safe_relative(relative)).relative_to(self.root)))
                    with Image.open(path) as im:
                        im=im.convert('RGB')
                        if cam=='head' and self.head_view!='full':
                            half=im.width//2
                            im=im.crop((0 if self.head_view=='left' else half,0,half if self.head_view=='left' else im.width,im.height))
                        if self.image_size:im=ImageOps.pad(im,self.image_size,color=(0,0,0))
                        frames.append(np.asarray(im,dtype=np.uint8).transpose(2,0,1).copy())
                out['observation.images.'+cam]=frames[0] if self.sequence_length==1 else np.stack(frames)
        return out

    @staticmethod
    def safe_relative(path):
        p=Path(path)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Invalid image path')
        return p


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root');ap.add_argument('--split',default='all');ap.add_argument('--images',action='store_true');ap.add_argument('--sequence-length',type=int,default=1);args=ap.parse_args()
    ds=RobotDataset(args.root,split=args.split,images=args.images,sequence_length=args.sequence_length)
    print(f'{len(ds)} valid samples / sequence windows')
    if len(ds):
        for key,value in ds[0].items():print(key, f'{value.shape} {value.dtype}' if hasattr(value,'shape') else value)
