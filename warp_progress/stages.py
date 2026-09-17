"""Task-specific stage model inspired by SARM; separate from WARP velocity units.

The last state is post-release observation, not an assertion of task success.
No absolute timestamp or episode duration is an input to the neural network.
"""
import numpy as np
import torch
from torch import nn

STAGE_NAMES = ['接近左桌', '抓取香蕉', '持物搬运', '向篮子放置', '释放后观察']
STAGE_VERSION = 1


class StageNet(nn.Module):
    def __init__(self, views=2, temporal=True, width=96, context=8):
        super().__init__()
        self.config = dict(views=views, temporal=temporal, width=width, context=context)
        self.register_buffer('mean', torch.zeros(views, 384))
        self.register_buffer('scale', torch.ones(views, 384))
        self.project = nn.Sequential(nn.Linear(384*views, width), nn.LayerNorm(width), nn.GELU())
        self.temporal = temporal
        if temporal:
            layer = nn.TransformerEncoderLayer(width, 4, width*2, .1, activation='gelu', batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            # Position is relative to a fixed context window, never global video time.
            self.position = nn.Parameter(torch.randn(1, context, width)*.01)
        else:
            self.encoder = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(.1))
        self.stage = nn.Linear(width, len(STAGE_NAMES))
        self.within = nn.Sequential(nn.Linear(width+len(STAGE_NAMES), width), nn.GELU(), nn.Linear(width, len(STAGE_NAMES)))

    def forward(self, x, valid):
        x = (x - self.mean) / self.scale
        x = torch.where(valid[..., None, None], x, 0.)
        h = self.project(x.flatten(-2))
        if self.temporal:
            safe = valid.clone()
            safe[:, -1] = True  # avoid all-masked attention; output validity is retained separately
            h = self.encoder(h+self.position, src_key_padding_mask=~safe)[:, -1]
        else:
            h = self.encoder(h[:, -1])
        logits = self.stage(h)
        within = self.within(torch.cat([h, logits.softmax(-1)], -1)).sigmoid()
        return logits, within


def context_indices(length, context=8):
    return np.maximum(0, np.arange(length)[:, None] - np.arange(context-1, -1, -1)[None])


def infer_stage_models(models, features, valid, priors, temperature=1., batch_size=128):
    """Causal fixed-window inference. Missing current observations stay missing."""
    features = np.asarray(features, np.float32)
    valid = np.asarray(valid, bool)
    if not models or features.ndim != 3 or len(features) != len(valid):
        raise ValueError('Invalid stage model input')
    if features.shape[1:] != (models[0].config['views'], 384):
        raise ValueError('Stage model camera count or feature dimension mismatch')
    if not np.isfinite(features[valid]).all() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Nonfinite visual features or invalid temperature')
    ix = context_indices(len(features), models[0].config['context'])
    model_probs, model_within = [], []
    for model in models:
        model.eval()
        probs, within = [], []
        with torch.inference_mode():
            for start in range(0, len(ix), batch_size):
                indices = ix[start:start+batch_size]
                logits, local = model(torch.from_numpy(features[indices]), torch.from_numpy(valid[indices]))
                probs.append((logits/temperature).softmax(-1).numpy())
                within.append(local.numpy())
        model_probs.append(np.concatenate(probs))
        model_within.append(np.concatenate(within))
    mp = np.stack(model_probs)
    probability = mp.mean(0)
    weights = np.asarray(priors, np.float32)
    if weights.shape != (5,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.):
        raise ValueError('Invalid stage priors')
    offsets = np.r_[0., np.cumsum(weights)[:-1]]
    # Stage-aligned interpolation target, not calibrated percent completion.
    progress = np.mean([np.sum(p * (offsets+w*weights), -1) for p,w in zip(mp, model_within)], axis=0)
    prediction = probability.argmax(-1)
    confidence = probability.max(-1)
    disagreement = (mp.argmax(-1) != prediction).any(0)
    review = (confidence < .8) | disagreement | ~valid
    probability[~valid] = np.nan
    progress[~valid] = np.nan
    confidence[~valid] = np.nan
    prediction[~valid] = -1
    return dict(probability=probability, progress=progress, stage=prediction, confidence=confidence,
                review=review, valid=valid, disagreement=disagreement)


def load_stage_checkpoint(path, expected_sha256):
    from .model import file_sha256
    if file_sha256(path) != expected_sha256:
        raise ValueError('阶段模型文件校验失败')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('kind') != 'stage_progress' or saved.get('version') != STAGE_VERSION:
        raise ValueError('阶段模型版本不匹配')
    if not saved.get('states'):
        raise ValueError('阶段模型缺少权重')
    models = []
    for state in saved['states']:
        model = StageNet(**saved['config'])
        model.load_state_dict(state, strict=True)
        models.append(model.eval())
    return models, saved
