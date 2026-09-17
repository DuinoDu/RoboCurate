"""WARP action-chunk weighting for an existing PyTorch BC training loop."""
import torch


def weighted_bc_loss(prediction, target, sample_weight):
    """Mean(weight * per-chunk MSE); batch must contain only valid full chunks.

Use RobotDataset(curation='warp', sequence_length=the_exported_horizon).
The caller retains control of its policy architecture and optimizer.
"""
    if prediction.shape!=target.shape or prediction.ndim<2:
        raise ValueError('BC prediction and target chunk shapes must match')
    weights=torch.as_tensor(sample_weight,device=prediction.device,dtype=prediction.dtype)
    if weights.shape!=(prediction.shape[0],) or not torch.isfinite(weights).all() or (weights<0).any():
        raise ValueError('Each action chunk requires a finite nonnegative weight')
    if not torch.isfinite(target).all():raise ValueError('BC targets contain invalid samples')
    per_chunk=(prediction-target).square().flatten(1).mean(1)
    return (weights*per_chunk).mean()
