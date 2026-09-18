"""Extract timestamped camera evidence for task annotation; never write source MCAPs."""
from pathlib import Path
import argparse
import io
import json
import sys
import numpy as np
from PIL import Image, ImageDraw
from mcap.reader import make_reader

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT
sys.path.insert(0, str(APP))
from server import _jpeg_payload


def inspect(episode, samples, cameras):
    out = Path(__file__).resolve().parent/'local_evidence'/episode
    out.mkdir(parents=True, exist_ok=True)
    matches=[]
    for path in (APP/'workspace/cache').glob('*/meta.json'):
        meta=json.loads(path.read_text())
        if episode in Path(meta['source']['path']).parts:
            matches.append(meta)
    if len(matches)!=1:
        raise ValueError(f'Expected one recording for {episode}, got {len(matches)}')
    meta=matches[0]
    source=Path(meta['source']['path'])
    start,end=meta['episode']['start_ns'],meta['episode']['end_ns']
    targets=np.linspace(start+100_000_000,end-150_000_000,samples).astype(np.int64)
    records=[]
    for camera in cameras:
        tiles=[];cursor=0
        with source.open('rb') as stream:
            for _,_,msg in make_reader(stream).iter_messages(topics=[meta['cameras'][camera]['topic']]):
                if cursor==len(targets):break
                if msg.log_time<targets[cursor]:continue
                raw=_jpeg_payload(msg.data)
                if not raw:continue
                with Image.open(io.BytesIO(raw)) as im:
                    rgb=im.convert('RGB')
                    if camera=='head':rgb=rgb.crop((0,0,rgb.width//2,rgb.height))
                    t=(msg.log_time-start)/1e9
                    name=f'{camera}_{cursor:03d}_{t:06.2f}s.jpg'
                    rgb.save(out/name,quality=95)
                    tile=Image.new('RGB',(480,294),'#15171b')
                    rgb.thumbnail((480,270))
                    tile.paste(rgb,((480-rgb.width)//2,24+(270-rgb.height)//2))
                    ImageDraw.Draw(tile).text((8,6),f'{episode} | {camera} | {t:.2f}s',fill='white')
                    tiles.append(tile)
                    records.append(dict(camera=camera,time_s=t,source_timestamp_ns=msg.log_time,file=name))
                    cursor+=1
        if len(tiles)!=samples:raise ValueError('Insufficient frames for evidence sampling')
        sheet=Image.new('RGB',(480*3,294*((len(tiles)+2)//3)),'#15171b')
        for i,tile in enumerate(tiles):sheet.paste(tile,((i%3)*480,(i//3)*294))
        sheet.save(out/(camera+'_contact.jpg'),quality=93)
    report=dict(episode=episode,source=str(source),source_fingerprint=dict(size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns),
                source_start_ns=start,source_end_ns=end,head_view='left',frames=records,
                purpose='Original timestamped frame evidence; no model predictions or outcome annotations included')
    (out/'index.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(dict(episode=episode,frames=len(records),output=str(out)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',nargs='+',required=True)
    p.add_argument('--samples',type=int,default=15)
    p.add_argument('--cameras',nargs='+',default=['head'],choices=['head','left_wrist','right_wrist'])
    a=p.parse_args()
    for ep in a.episodes:inspect(ep,a.samples,a.cameras)
