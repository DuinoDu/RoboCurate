"""Motion model with training-time contrastive and uncertain-boundary objectives.

Task-specific adaptations of TSCL (RA-L 2024) and Mitsuoka & Hotta (CVPRW
2026). Sparse annotation brackets supervise boundary occurrence, never an
invented exact timestamp. Future annotations are used only in training.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .stages_v3 import MotionStageNet,inference_v3
from .stages import context_indices


class RefinedStageNet(MotionStageNet):
    def __init__(self,width=96,context=8,contrastive=False,boundary=False,detach_progress=False):
        super().__init__(width=width,context=context)
        self.config.update(contrastive=contrastive,boundary=boundary,detach_progress=detach_progress)
        # Preserve shared initialization and RNG for fair comparison with v3.
        with torch.random.fork_rng(devices=[]):
            self.projection=nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Linear(64,64)) if contrastive else None
            self.boundary_head=nn.Linear(width,1) if boundary else None

    def forward(self,x,camera_valid,state,state_valid,return_aux=False):
        logits,within,embedding=super().forward(x,camera_valid,state,state_valid,return_embedding=True)
        if self.config['detach_progress']:
            # SARM2 separates stage/value optimization. Here the proxy progress
            # head reads fixed stage features; its MSE cannot train the encoder.
            within=self.within(torch.cat([embedding.detach(),logits.softmax(-1).detach()],-1)).sigmoid()
        if not return_aux:return logits,within
        projected=self.projection(embedding) if self.projection is not None else embedding
        boundary=self.boundary_head(embedding).squeeze(-1) if self.boundary_head is not None else embedding[:,0]*0
        return logits,within,projected,boundary


def cross_recording_contrastive(embedding,labels,recordings,temperature=.1):
    """Same-stage positives must come from a different recording; no time input."""
    z=F.normalize(embedding,dim=-1);similarity=z@z.T/temperature
    self_mask=torch.eye(len(z),dtype=torch.bool,device=z.device)
    positives=(labels[:,None]==labels[None,:])&(recordings[:,None]!=recordings[None,:])&~self_mask
    usable=positives.any(1)
    if not usable.any():return embedding.sum()*0
    logprob=similarity-torch.logsumexp(similarity.masked_fill(self_mask,-torch.inf),dim=1,keepdim=True)
    return -((logprob*positives).sum(1)/positives.sum(1).clamp_min(1))[usable].mean()


def boundary_bag_loss(positive_bags,stable_logits):
    """At least one transition inside each uncertain bracket; stable negatives."""
    positive=[]
    for logits in positive_bags:
        if not len(logits):raise ValueError('Empty boundary bracket')
        log_none=F.logsigmoid(-logits).sum()
        positive.append(-torch.log((-torch.expm1(log_none)).clamp_min(1e-7)))
    if not positive or not len(stable_logits):raise ValueError('Boundary supervision needs brackets and stable samples')
    return .5*(torch.stack(positive).mean()+F.softplus(stable_logits).mean())


def segment_cdf_loss(probability):
    """CDF shape regularizer inside one known constant-stage training interval."""
    if probability.ndim!=1 or len(probability)<2:raise ValueError('Segment needs at least two ordered samples')
    p=probability/probability.sum().clamp_min(1e-8)
    uniform=torch.arange(1,len(p)+1,device=p.device,dtype=p.dtype)/len(p)
    return (p.cumsum(0)-uniform).square().mean()


def inference_v4(models,features,camera_valid,state,state_valid,priors,temperature=1.,review_boundary=False):
    out=inference_v3(models,features,camera_valid,state,state_valid,priors,temperature,False,True)
    available=all(m.boundary_head is not None for m in models)
    probability=np.full(len(features),np.nan,np.float32)
    if available:
        indices=context_indices(len(features),models[0].config['context']);predictions=[]
        arrays=[np.asarray(features,np.float32),np.asarray(camera_valid,bool),np.asarray(state,np.float32),np.asarray(state_valid,bool)]
        with torch.inference_mode():
            for m in models:
                m.eval();parts=[]
                for start in range(0,len(indices),128):
                    ix=indices[start:start+128]
                    b=m(*[torch.from_numpy(a[ix]) for a in arrays],return_aux=True)[-1]
                    parts.append(b.sigmoid().numpy())
                predictions.append(np.concatenate(parts))
        probability=np.mean(predictions,axis=0);probability[~out['valid']]=np.nan
    elif review_boundary:raise ValueError('Boundary review requires a trained boundary head')
    added=out['valid']&(probability>=.5)&~out['review'] if review_boundary else np.zeros(len(features),bool)
    out['review'] |= added;out['review_reasons'][added] |= 64
    out.update(boundary_probability=probability,learned_boundary_review=added)
    return out


def load_stage_v4(path,expected_sha256):
    from .model import file_sha256
    if file_sha256(path)!=expected_sha256:raise ValueError('阶段模型文件校验失败')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('kind')!='stage_progress' or saved.get('version')!=4:raise ValueError('阶段模型版本不匹配')
    config=dict(saved['config']);config.pop('use_state',None);models=[]
    for state in saved['states']:
        m=RefinedStageNet(**config);m.load_state_dict(state,strict=True);models.append(m.eval())
    if not models:raise ValueError('缺少模型权重')
    return models,saved
