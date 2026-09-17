"""Small multimodal temporal stage network: visual evidence, measured state and stage tokens.

SARM motivates measured-state/vision stage-progress modeling. FACT motivates
frame/stage token cross-attention. This is a task adaptation, not either full model.
"""
import math
import numpy as np
import torch
from torch import nn
from .stages import context_indices


class FusionStageNet(nn.Module):
    def __init__(self, temporal=True, stage_tokens=False, use_state=True, width=96, context=8):
        super().__init__()
        self.config=dict(temporal=temporal,stage_tokens=stage_tokens,use_state=use_state,width=width,context=context)
        self.register_buffer('mean',torch.zeros(2,384));self.register_buffer('scale',torch.ones(2,384))
        self.register_buffer('state_mean',torch.zeros(41));self.register_buffer('state_scale',torch.ones(41))
        self.vision=nn.ModuleList([nn.Sequential(nn.Linear(384,64),nn.LayerNorm(64),nn.GELU()) for _ in range(2)])
        self.state_encoder=nn.Sequential(nn.Linear(41,32),nn.LayerNorm(32),nn.GELU()) if use_state else None
        self.fusion=nn.Sequential(nn.Linear(128+(32 if use_state else 0)+3,width),nn.LayerNorm(width),nn.GELU())
        self.temporal=temporal;self.use_state=use_state;self.stage_tokens=stage_tokens
        if temporal:
            layer=nn.TransformerEncoderLayer(width,4,2*width,.1,activation='gelu',batch_first=True)
            self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.position=nn.Parameter(torch.randn(1,context,width)*.01)
        else:self.encoder=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Dropout(.1))
        if stage_tokens:
            self.tokens=nn.Parameter(torch.randn(1,5,width)*.02)
            self.to_stages=nn.MultiheadAttention(width,4,.1,batch_first=True)
            self.to_frame=nn.MultiheadAttention(width,4,.1,batch_first=True)
            self.norm_tokens=nn.LayerNorm(width);self.norm_frame=nn.LayerNorm(width)
            self.stage_bias=nn.Parameter(torch.zeros(5))
        else:self.classifier=nn.Linear(width,5)
        self.within=nn.Sequential(nn.Linear(width+5,width),nn.GELU(),nn.Linear(width,5))

    def forward(self,x,camera_valid,state,state_valid):
        x=torch.where(camera_valid[...,None],(x-self.mean)/self.scale,0.)
        v=[self.vision[j](x[:,:,j])*camera_valid[:,:,j,None] for j in range(2)]
        if self.use_state:
            q=torch.where(state_valid[...,None],(state-self.state_mean)/self.state_scale,0.)
            v.append(self.state_encoder(q)*state_valid[...,None])
        mask=torch.cat([camera_valid,state_valid[...,None]],-1).to(x.dtype)
        if not self.use_state:mask=mask.clone();mask[:,:,-1]=0
        h=self.fusion(torch.cat(v+[mask],-1))
        frame_valid=camera_valid.any(-1)
        safe=frame_valid.clone();safe[:,-1]=True
        if self.temporal:h=self.encoder(h+self.position,src_key_padding_mask=~safe)
        else:h=self.encoder(h)
        current=h[:,-1]
        if self.stage_tokens:
            tokens=self.tokens.expand(len(x),-1,-1)
            addition,_=self.to_stages(tokens,h,h,key_padding_mask=~safe,need_weights=False)
            tokens=self.norm_tokens(tokens+addition)
            addition,_=self.to_frame(current[:,None],tokens,tokens,need_weights=False)
            current=self.norm_frame(current+addition[:,0])
            logits=(tokens*current[:,None]).sum(-1)/math.sqrt(current.shape[-1])+self.stage_bias
        else:logits=self.classifier(current)
        within=self.within(torch.cat([current,logits.softmax(-1)],-1)).sigmoid()
        return logits,within


def infer_fusion(models,features,camera_valid,state,state_valid,priors,temperature=1.,batch_size=128):
    x=np.asarray(features,np.float32);cv=np.asarray(camera_valid,bool)
    q=np.asarray(state,np.float32);qv=np.asarray(state_valid,bool)
    if x.ndim!=3 or x.shape[1:]!=(2,384) or cv.shape!=x.shape[:2] or q.shape!=(len(x),41) or qv.shape!=(len(x),):
        raise ValueError('多模态阶段输入维数不匹配')
    if not len(x) or not models or not np.isfinite(x[cv]).all() or not np.isfinite(q[qv]).all():
        raise ValueError('多模态阶段输入为空或含无效值')
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('校准温度无效')
    valid=cv.any(1)
    ix=context_indices(len(x),models[0].config['context'])
    probabilities=[];local=[]
    for model in models:
        model.eval();ps=[];ws=[]
        with torch.inference_mode():
            for start in range(0,len(x),batch_size):
                ids=ix[start:start+batch_size]
                logits,within=model(torch.from_numpy(x[ids]),torch.from_numpy(cv[ids]),torch.from_numpy(q[ids]),torch.from_numpy(qv[ids]))
                ps.append((logits/temperature).softmax(-1).numpy());ws.append(within.numpy())
        probabilities.append(np.concatenate(ps));local.append(np.concatenate(ws))
    mp=np.stack(probabilities);p=mp.mean(0);stage=p.argmax(-1);confidence=p.max(-1)
    weights=np.asarray(priors,np.float32)
    if weights.shape!=(5,) or np.any(weights<0) or not np.isclose(weights.sum(),1):raise ValueError('阶段先验无效')
    offsets=np.r_[0.,np.cumsum(weights)[:-1]]
    progress=np.mean([(p*(offsets+w*weights)).sum(-1) for p,w in zip(probabilities,local)],0)
    disagreement=(mp.argmax(-1)!=stage).any(0)
    degraded=~cv.all(1) | (~qv if models[0].config['use_state'] else False)
    review=(confidence<.8)|disagreement|degraded|~valid
    p[~valid]=np.nan;progress[~valid]=np.nan;confidence[~valid]=np.nan;stage[~valid]=-1
    return dict(probability=p,progress=progress,confidence=confidence,stage=stage,valid=valid,
                review=review,disagreement=disagreement,degraded=degraded)


def load_fusion_checkpoint(path,expected_sha256):
    from .model import file_sha256
    if file_sha256(path)!=expected_sha256:raise ValueError('阶段模型文件校验失败')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('kind')!='stage_progress' or saved.get('version')!=2:raise ValueError('多模态阶段模型版本不匹配')
    models=[]
    for state in saved['states']:
        model=FusionStageNet(**saved['config']);model.load_state_dict(state,strict=True);models.append(model.eval())
    if not models:raise ValueError('缺少模型权重')
    return models,saved
