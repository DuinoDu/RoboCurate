"""CPU/CUDA model adapter, compatible with the published WARP-RM checkpoint."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from vendor.warp_rm.aggregator import TransformerAggregator
from .core import WarpConfig, plan_windows, aggregate_windows


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_model(path, device="cpu", expected_sha256=None):
    path = Path(path)
    digest = file_sha256(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("模型权重校验失败，请重新安装已核验的模型")
    # Never fall back to unrestricted pickle loading.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a model state dictionary")
    if checkpoint.get("label_mode", "relative") != "relative" or checkpoint.get("attention", "bidirectional") != "bidirectional":
        raise ValueError("This adapter implements the paper's relative, bidirectional model")
    state = checkpoint["model"]
    architecture = checkpoint.get("architecture", {})
    dim = int(state["pos_embed"].shape[2])
    window = int(state["pos_embed"].shape[1])
    layers = 1 + max(int(k.split(".")[2]) for k in state if k.startswith("transformer.layers."))
    bins = state["rel_bin_centers"]
    backbone_dim = int(checkpoint.get("backbone_dim", 768))
    if checkpoint.get("n_cameras", 1) != 1:
        raise ValueError("This release scores one explicitly selected camera per model")
    model = TransformerAggregator(d_model=dim, n_heads=int(architecture.get("n_heads", 8)),
                                  n_layers=layers, max_seq_len=window, backbone_dim=backbone_dim,
                                  use_temporal_diffs=state["input_proj.weight"].shape[1] == 2 * backbone_dim,
                                  n_rel_bins=len(bins), rel_bin_min=float(bins[0]), rel_bin_max=float(bins[-1]),
                                  dropout=float(architecture.get("dropout", .15)),
                                  stochastic_depth_p=float(architecture.get("stochastic_depth_p", .1)))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    config = WarpConfig(**checkpoint["warp_config"]) if "warp_config" in checkpoint else WarpConfig(
        window_size=window, feature_stride=int(checkpoint.get("feature_stride", 3)),
        source_standard_stride=int(checkpoint.get("standard_stride_src", 45)))
    if config.window_size != window:
        raise ValueError('Checkpoint window size differs from its label configuration')
    model._warp_rm_abs_head_trained = False
    meta = {k: v for k, v in checkpoint.items() if k not in ("model", "optimizer")}
    meta.update(sha256=digest, parameters=sum(p.numel() for p in model.parameters()))
    return model, config, meta


def infer_features(model, features, valid, config, batch_size=16, device="cpu", progress=None):
    features = np.asarray(features, np.float32)
    valid = np.asarray(valid, bool)
    if features.ndim != 2 or valid.shape != (len(features),) or type(batch_size) is not int or batch_size < 1:
        raise ValueError('Invalid feature matrix, validity mask or inference batch size')
    valid = valid & np.isfinite(features).all(axis=1)
    indices, scales, spans = plan_windows(valid, config)
    if not len(indices):
        raise ValueError("连续图像不足，无法形成一个有效的 WARP 观察窗口")
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = torch.from_numpy(features[indices[start:start + batch_size]]).to(device)
            values = model(batch)[0].detach().float().cpu().numpy()
            predictions.append(values)
            if progress:
                progress(min(start + batch_size, len(indices)), len(indices))
    prediction = np.concatenate(predictions)
    velocity, coverage = aggregate_windows(indices, prediction, len(features), scales)
    return dict(velocity=velocity, coverage=coverage, valid=coverage > 0,
                windows=len(indices), shortened_windows=int((scales != 1).sum()),
                span_s_min=float(spans.min()), span_s_max=float(spans.max()))
