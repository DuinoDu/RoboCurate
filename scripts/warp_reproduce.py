#!/usr/bin/env python3
"""Run released WARP-RM on one public logical episode and compare reference inference.

All outputs derive from the actual video and published weights. This verifies
reward-model inference, not the paper's policy-training/robot success claims.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
APP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP))
import numpy as np
import pandas as pd
import torch
from warp_progress.core import UPSTREAM,chunk_weights,curve_summary,json_signature
from warp_progress.features import DinoEncoder,extract_video,atomic_json,atomic_npz,feature_contract
from warp_progress.model import load_model,infer_features,file_sha256
from vendor.warp_rm.inference import dense_inference_relative
from scripts.warp_prepare_models import HEAD_SHA256

DATASET_REVISION='897b86679272a47897cc5ecca78c2205586a29d5'


def reproduce(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    root=Path(args.workspace).resolve()/'warp/models'
    torch.set_num_threads(args.threads)
    provenance=json.loads((root/'backbone-provenance.json').read_text())
    profile=dict(camera='top_camera-images-rgb',view='full',crop='squash',
                 backbone_revision=provenance['revision'],backbone_sha256=provenance['sha256'])
    model,config,checkpoint=load_model(root/'paper_sim_sss15.pt',args.device,HEAD_SHA256)
    df=pd.read_parquet(args.episode_metadata)
    rows=df[df['episode_index']==args.episode_index]
    if len(rows)!=1:raise ValueError('Logical episode index is not unique in metadata')
    row=rows.iloc[0]
    start_frame=round(float(row['videos/top_camera-images-rgb/from_timestamp'])*config.fps)
    count=int(row['length'])
    video_hash=file_sha256(args.video)
    if args.video_sha256 and args.video_sha256!=video_hash:raise ValueError('Public video SHA256 mismatch')
    contract=feature_contract(profile,config);contract['decoder']='OpenCV-RGB'
    cache_key=json_signature(dict(video_sha256=video_hash,start_frame=start_frame,n_frames=count,contract=contract))
    feature_path=output/(cache_key+'.features.npz')
    started=time.monotonic()
    if feature_path.is_file() and feature_path.with_suffix('.json').is_file():
        with np.load(feature_path,allow_pickle=False) as a:data={k:a[k] for k in a.files}
    else:
        encoder=DinoEncoder(root/'dinov3-vitb16',args.device,provenance['sha256'])
        data=extract_video(args.video,config,profile,feature_path,encoder,batch_size=args.feature_batch,
                           start_frame=start_frame,n_frames=count,
                           progress=lambda done,total:print(f'features {done}/{total}',flush=True))
        del encoder
    feature_seconds=time.monotonic()-started
    started=time.monotonic()
    scored=infer_features(model,data['features'],data['valid'],config,args.score_batch,args.device,
                          progress=lambda done,total:print(f'windows {done}/{total}',flush=True))
    score_seconds=time.monotonic()-started
    _,reference=dense_inference_relative(model,data['features'],torch.device(args.device),
                                         window_size=config.window_size,standard_feat_steps=config.step,batch_size=args.score_batch)
    error=float(np.max(np.abs(reference[scored['valid']]-scored['velocity'][scored['valid']])))
    if error>2e-5:raise AssertionError(f'Reference velocity parity failed: max error {error}')
    times=data['timestamp_ns']/1e9
    paper_weights,_=chunk_weights(reference,np.isfinite(reference),30,args.threshold,'binary',pad_tail=True)
    conservative_weights,eligible=chunk_weights(scored['velocity'],scored['valid'],30,args.threshold,'binary')
    atomic_npz(output/'predictions.npz',timestamp_ns=data['timestamp_ns'],velocity=scored['velocity'],
               valid=scored['valid'],coverage=scored['coverage'],reference_velocity=reference,
               paper_binary_weight=paper_weights,conservative_binary_weight=conservative_weights,eligible=eligible)
    report=dict(state='complete',upstream_commit=UPSTREAM,dataset_revision=DATASET_REVISION,
                episode_index=args.episode_index,source_episode_id=str(row['source_episode_id']),frames=count,
                video_sha256=video_hash,metadata_sha256=file_sha256(args.episode_metadata),
                checkpoint_sha256=checkpoint['sha256'],backbone=provenance,feature_contract=contract,
                warp_config=asdict(config),max_reference_velocity_error=error,
                feature_seconds=feature_seconds,score_seconds=score_seconds,windows=scored['windows'],
                coverage=float(scored['valid'].mean()),threshold=args.threshold,
                paper_tail_padded_kept=int((paper_weights>0).sum()),conservative_kept=int((conservative_weights>0).sum()),
                summary=curve_summary(times,scored['velocity'],scored['valid']),
                limitations=['One public logical episode, not a full corpus replication',
                             'Published head reused; this run does not retrain the paper checkpoint',
                             'Does not measure BC throughput or G1 task success',
                             'Deployment retains uncovered intervals; paper reference fills them'])
    atomic_json(output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('state','episode_index','frames','max_reference_velocity_error','coverage','feature_seconds','score_seconds')},ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',default=str(APP/'workspace'));p.add_argument('--video',required=True)
    p.add_argument('--video-sha256');p.add_argument('--episode-metadata',required=True)
    p.add_argument('--episode-index',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--device',default='cpu')
    p.add_argument('--feature-batch',type=int,default=8);p.add_argument('--score-batch',type=int,default=16)
    p.add_argument('--threshold',type=float,default=1.)
    reproduce(p.parse_args())
