"""Causal motion-aware and short/sparse-long context research architectures.

Motion differences follow WARP-RM; the experimental context branches follow
LTContext, with FACT-inspired stage tokens. These are task-specific adaptations.
"""
import math
import numpy as np
import torch
from torch import nn
from .stages_v2 import FusionStageNet,infer_fusion


class MultiScaleStageNet(FusionStageNet):
    def __init__(self,width=96,context=29,branch_size=8,long_stride=4):
        if context!=(branch_size-1)*long_stride+1 or branch_size!=8:
            raise ValueError('长短窗口配置不匹配')
        super().__init__(temporal=True,stage_tokens=True,use_state=True,width=width,context=branch_size)
        self.config=dict(width=width,context=context,branch_size=branch_size,long_stride=long_stride,use_state=True)
        self.long_position=nn.Parameter(torch.randn(1,branch_size,width)*.01)
        self.scale_embedding=nn.Parameter(torch.randn(2,1,width)*.01)

    def forward(self,x,camera_valid,state,state_valid):
        x=torch.where(camera_valid[...,None],(x-self.mean)/self.scale,0.)
        q=torch.where(state_valid[...,None],(state-self.state_mean)/self.state_scale,0.)
        streams=[self.vision[j](x[:,:,j])*camera_valid[:,:,j,None] for j in range(2)]
        streams.append(self.state_encoder(q)*state_valid[...,None])
        mask=torch.cat([camera_valid,state_valid[...,None]],-1).to(x.dtype)
        fused=self.fusion(torch.cat(streams+[mask],-1));valid=camera_valid.any(-1)
        short=fused[:,-8:];long=fused[:,::self.config['long_stride']]
        masks=[valid[:,-8:].clone(),valid[:,::self.config['long_stride']].clone()]
        for m in masks:m[:,-1]=True
        short=self.encoder(short+self.position+self.scale_embedding[0],src_key_padding_mask=~masks[0])
        long=self.encoder(long+self.long_position+self.scale_embedding[1],src_key_padding_mask=~masks[1])
        history=torch.cat([short,long],1);safe=torch.cat(masks,1)
        current=short[:,-1];tokens=self.tokens.expand(len(x),-1,-1)
        addition,_=self.to_stages(tokens,history,history,key_padding_mask=~safe,need_weights=False)
        tokens=self.norm_tokens(tokens+addition)
        addition,_=self.to_frame(current[:,None],tokens,tokens,need_weights=False)
        current=self.norm_frame(current+addition[:,0])
        logits=(tokens*current[:,None]).sum(-1)/math.sqrt(current.shape[-1])+self.stage_bias
        within=self.within(torch.cat([current,logits.softmax(-1)],-1)).sigmoid()
        return logits,within


class MotionStageNet(FusionStageNet):
    """Explicit current-minus-previous visual/state inputs, following WARP's idea.

    Differences are finite changes on the 2 Hz grid, not measured joint velocity.
    A difference is masked unless both of its endpoints are valid.
    """
    def __init__(self,width=96,context=8):
        super().__init__(temporal=True,stage_tokens=True,use_state=True,width=width,context=context)
        self.config=dict(width=width,context=context,use_state=True)
        self.vision=nn.ModuleList([nn.Sequential(nn.Linear(768,64),nn.LayerNorm(64),nn.GELU()) for _ in range(2)])
        self.state_encoder=nn.Sequential(nn.Linear(82,32),nn.LayerNorm(32),nn.GELU())
        self.fusion=nn.Sequential(nn.Linear(128+32+6,width),nn.LayerNorm(width),nn.GELU())

    def forward(self,x,camera_valid,state,state_valid,return_embedding=False):
        x=torch.where(camera_valid[...,None],(x-self.mean)/self.scale,0.)
        q=torch.where(state_valid[...,None],(state-self.state_mean)/self.state_scale,0.)
        cv_delta=torch.zeros_like(camera_valid);qv_delta=torch.zeros_like(state_valid)
        cv_delta[:,1:]=camera_valid[:,1:]&camera_valid[:,:-1]
        qv_delta[:,1:]=state_valid[:,1:]&state_valid[:,:-1]
        dx=torch.zeros_like(x);dq=torch.zeros_like(q)
        dx[:,1:]=torch.where(cv_delta[:,1:,:,None],x[:,1:]-x[:,:-1],0.)
        dq[:,1:]=torch.where(qv_delta[:,1:,None],q[:,1:]-q[:,:-1],0.)
        streams=[self.vision[j](torch.cat([x[:,:,j],dx[:,:,j]],-1))*camera_valid[:,:,j,None] for j in range(2)]
        streams.append(self.state_encoder(torch.cat([q,dq],-1))*state_valid[...,None])
        masks=torch.cat([camera_valid,state_valid[...,None],cv_delta,qv_delta[...,None]],-1).to(x.dtype)
        h=self.fusion(torch.cat(streams+[masks],-1))
        safe=camera_valid.any(-1).clone();safe[:,-1]=True
        h=self.encoder(h+self.position,src_key_padding_mask=~safe)
        current=h[:,-1];tokens=self.tokens.expand(len(x),-1,-1)
        addition,_=self.to_stages(tokens,h,h,key_padding_mask=~safe,need_weights=False)
        tokens=self.norm_tokens(tokens+addition)
        addition,_=self.to_frame(current[:,None],tokens,tokens,need_weights=False)
        current=self.norm_frame(current+addition[:,0])
        logits=(tokens*current[:,None]).sum(-1)/math.sqrt(current.shape[-1])+self.stage_bias
        within=self.within(torch.cat([current,logits.softmax(-1)],-1)).sigmoid()
        return (logits,within,current) if return_embedding else (logits,within)


def make_stage_v3(config):
    config=dict(config);architecture=config.pop('architecture')
    if architecture=='multiscale':
        config.pop('use_state',None)
        return MultiScaleStageNet(**config)
    if architecture=='motion':
        config.pop('use_state',None)
        return MotionStageNet(**config)
    if architecture=='v2':return FusionStageNet(**config)
    raise ValueError('未知的阶段网络架构')


def inference_v3(models,features,camera_valid,state,state_valid,priors,temperature=1.,boundary_review=True,review_guard=False):
    out=infer_fusion(models,features,camera_valid,state,state_valid,priors,temperature)
    guard=np.zeros(len(features),bool)
    if review_guard and temperature!=1.:
        raw=infer_fusion(models,features,camera_valid,state,state_valid,priors,1.)
        guard=raw['review']&~out['review']
        out['review'] |= raw['review']
    # At 2 Hz: current and next sample after a prediction change. Past/current only.
    transition=np.zeros(len(features),bool)
    change=out['valid'][1:]&out['valid'][:-1]&(out['stage'][1:]!=out['stage'][:-1])
    transition[1:]=change
    transition[2:] |= change[:-1]
    transition &= out['valid']
    out['boundary_review']=transition if boundary_review else np.zeros(len(features),bool)
    out['review'] |= out['boundary_review']
    # Explainable reasons are independent flags; no final task-success assertion.
    reasons=np.zeros(len(features),np.uint8)
    reasons[out['confidence']<.8] |= 1
    reasons[out['disagreement']] |= 2
    reasons[out['degraded']] |= 4
    reasons[~out['valid']] |= 8
    reasons[out['boundary_review']] |= 16
    reasons[guard] |= 32
    out['review_reasons']=reasons
    out['guard_review']=guard
    return out


def load_stage_v3(path,expected_sha256):
    from .model import file_sha256
    if file_sha256(path)!=expected_sha256:raise ValueError('阶段模型文件校验失败')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('kind')!='stage_progress' or saved.get('version')!=3:raise ValueError('阶段模型版本不匹配')
    models=[]
    for state in saved['states']:
        model=make_stage_v3(saved['config']);model.load_state_dict(state,strict=True);models.append(model.eval())
    if not models:raise ValueError('缺少模型权重')
    return models,saved
