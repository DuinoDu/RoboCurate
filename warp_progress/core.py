"""Paper WARP sampling, label calibration and conservative curation contracts.

AR sampling follows the pinned public implementation. RNG state is explicit
here so independent runs can be reproduced. Inference averages the same
interval velocities, but retains an explicit missing-coverage mask instead of
interpolating through missing data or inventing a completion curve.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numpy as np

VERSION = 1
UPSTREAM = "26f9894abdaf883af8db78a38e164764e5bf7c00"


@dataclass(frozen=True)
class WarpConfig:
    window_size: int = 32
    fps: int = 30
    feature_stride: int = 1
    source_standard_stride: int = 15
    ar_alpha: float = .5
    ar_sigma: float = math.log(2)
    path_center_s: float = 1.5
    path_half_range_s: float = 1.
    reversal_rate: float = 1.
    flip_probability: float = .5

    def __post_init__(self):
        for key in ("window_size", "fps", "feature_stride", "source_standard_stride"):
            v = getattr(self, key)
            if type(v) is not int or v < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 2 <= self.window_size <= 256 or self.fps > 240:
            raise ValueError("Unsupported window size or FPS")
        if self.source_standard_stride % self.feature_stride:
            raise ValueError("feature_stride must divide source_standard_stride")
        if not all(math.isfinite(v) for v in (self.ar_alpha, self.ar_sigma, self.path_center_s,
                                             self.path_half_range_s, self.reversal_rate, self.flip_probability)):
            raise ValueError("Nonfinite sampler configuration")
        if not 0 <= self.ar_alpha < 1 or self.ar_sigma < 0 or self.reversal_rate < 0:
            raise ValueError("Invalid AR or reversal parameters")
        if not 0 <= self.flip_probability <= 1 or self.path_center_s <= 0 or self.path_half_range_s < 0:
            raise ValueError("Invalid path parameters")

    @property
    def step(self):
        return self.source_standard_stride // self.feature_stride

    @property
    def span_s(self):
        return (self.window_size - 1) * self.source_standard_stride / self.fps


def sample_warp(n_frames, config, rng):
    """Sample feature indices using the released AR path-budget convention."""
    if n_frames < 2:
        raise ValueError("A trajectory needs at least two feature frames")
    n = config.window_size - 1
    rate = config.fps / config.feature_stride
    lower = max(n, round((config.path_center_s - config.path_half_range_s) * rate * n))
    upper = max(n, round((config.path_center_s + config.path_half_range_s) * rate * n))
    path = min(float(rng.integers(lower, upper + 1)), float(n_frames - 1))
    z = np.empty(n)
    z[0] = rng.normal(0, config.ar_sigma)
    scale = math.sqrt(1 - config.ar_alpha ** 2) * config.ar_sigma
    for i in range(1, n):
        z[i] = config.ar_alpha * z[i - 1] + scale * rng.normal()
    speed = np.exp(z)
    signs = np.ones(n)
    reversals = min(int(rng.poisson(config.reversal_rate)), max(0, n - 1))
    if reversals:
        points = np.sort(rng.choice(np.arange(1, n), size=reversals, replace=False))
        boundaries = np.r_[0, points, n]
        for i in range(1, len(boundaries) - 1, 2):
            signs[boundaries[i]:boundaries[i + 1]] = -1
    displacement = np.r_[0., np.cumsum(signs * (path / speed.sum()) * speed)]
    start = rng.uniform(0, max(n_frames - 1 - np.ptp(displacement), 0)) - displacement.min()
    positions = start + displacement
    if rng.random() < config.flip_probability:
        positions = positions[::-1]
    return np.clip(np.rint(positions), 0, n_frames - 1).astype(np.int64)


def relative_targets(indices, config):
    indices = np.asarray(indices)
    if indices.shape[-1] != config.window_size:
        raise ValueError("Window size differs from label normalization")
    return ((indices - indices[..., :1]) * config.feature_stride /
            ((config.window_size - 1) * config.source_standard_stride)).astype(np.float32)


def two_hot(values, low=-3., high=3., bins=30):
    if bins < 2 or not low < high:
        raise ValueError("Invalid categorical support")
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite target")
    x = (np.clip(values, low, high) - low) / ((high - low) / (bins - 1))
    lower = np.clip(x.astype(int), 0, bins - 2)
    upper_weight = np.clip(x - lower, 0, 1)
    out = np.zeros(values.shape + (bins,), np.float32)
    np.put_along_axis(out, lower[..., None], (1 - upper_weight)[..., None], -1)
    np.put_along_axis(out, (lower + 1)[..., None], upper_weight[..., None], -1)
    return out


def contiguous_runs(valid):
    v = np.asarray(valid, dtype=bool)
    edges = np.diff(np.r_[False, v, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def plan_windows(valid, config, allow_short=True):
    """Never cross a missing frame; return indices and per-window calibration."""
    n = config.window_size
    windows, scales, spans = [], [], []
    for start, end in contiguous_runs(valid):
        length = end - start
        step = config.step
        max_start = length - (n - 1) * step - 1
        if max_start <= 0 and allow_short and length >= n + 1:
            step = max(1, (length - 2) // (n - 1))
            max_start = length - (n - 1) * step - 1
        if max_start <= 0:
            continue
        offsets = np.arange(n) * step
        for s in range(max_start + 1):
            windows.append(start + s + offsets)
            scales.append(config.step / step)
            spans.append((n - 1) * step * config.feature_stride / config.fps)
    return (np.asarray(windows, dtype=np.int64).reshape(-1, n),
            np.asarray(scales), np.asarray(spans))


def aggregate_windows(indices, predictions, n_frames, scales=1.):
    """Overlap-average interval derivatives. Missing coverage stays NaN."""
    indices = np.asarray(indices, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if indices.ndim != 2 or predictions.shape != indices.shape or indices.shape[1] < 2:
        raise ValueError("Malformed prediction windows")
    if indices.size and (indices.min() < 0 or indices.max() >= n_frames or np.any(np.diff(indices) <= 0)):
        raise ValueError("Inference window indices must be increasing and in bounds")
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite model predictions")
    count = np.zeros(n_frames, dtype=np.int64)
    velocity = np.full(n_frames, np.nan, dtype=np.float32)
    if not len(indices):
        return velocity, count
    a, b = indices[:, :-1].ravel(), indices[:, 1:].ravel()
    scale = np.broadcast_to(np.asarray(scales), (len(indices),))
    values = (np.diff(predictions, axis=1) * (indices.shape[1] - 1) * scale[:, None]).ravel()
    delta = np.bincount(a, weights=values, minlength=n_frames + 1) - np.bincount(b, weights=values, minlength=n_frames + 1)
    changes = np.bincount(a, minlength=n_frames + 1) - np.bincount(b, minlength=n_frames + 1)
    count = np.cumsum(changes[:-1])
    covered = count > 0
    velocity[covered] = (np.cumsum(delta[:-1])[covered] / count[covered]).astype(np.float32)
    return velocity, count


def align_velocity(times_ns, velocity, covered, target_ns, max_age_ns):
    """Previous feature-frame association, with int64 timestamps and no gap fill."""
    t = np.asarray(times_ns, dtype=np.int64)
    v = np.asarray(velocity, dtype=np.float32)
    covered = np.asarray(covered, dtype=bool)
    grid = np.asarray(target_ns, dtype=np.int64)
    if max_age_ns < 0 or any(a.ndim != 1 for a in (t,v,covered,grid)):
        raise ValueError('Invalid timestamp or score axis')
    if len(t) != len(v) or len(t) != len(covered) or (len(t) > 1 and np.any(np.diff(t) <= 0)):
        raise ValueError("Invalid progress time axis")
    out = np.full(len(grid), np.nan, np.float32)
    if not len(t):
        return out, np.zeros(len(grid), bool)
    ix = np.searchsorted(t, grid, side="right") - 1
    safe = np.clip(ix, 0, len(t) - 1)
    age = grid - t[safe]
    valid = (ix >= 0) & (grid <= t[-1]) & (age <= max_age_ns) & covered[safe] & np.isfinite(v[safe])
    out[valid] = v[safe[valid]]
    return out, valid


def chunk_weights(velocity, valid, horizon=30, threshold=1., mode="continuous", pad_tail=False):
    """WARP-BC terminal-frame gate; optional tail padding only for paper repro."""
    v = np.asarray(velocity, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if v.ndim != 1 or valid.shape != v.shape:
        raise ValueError('Malformed velocity or validity mask')
    valid = valid & np.isfinite(v)
    if type(horizon) is not int or horizon < 1 or horizon > 10000 or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid chunk horizon or threshold")
    if mode not in ("continuous", "binary"):
        raise ValueError("Unknown WARP weighting mode")
    size = len(v)
    weights = np.zeros(size, dtype=np.float32)
    eligible = np.zeros(size, dtype=bool)
    end_index = np.arange(size) + horizon - 1
    if size:
        ends = np.minimum(end_index + 1, size)
        bad = np.r_[0, np.cumsum(~valid)]
        eligible = (bad[ends] == bad[np.arange(size)]) & ((end_index < size) | pad_tail)
        terminal = v[np.minimum(end_index, size - 1)]
        keep = eligible & (terminal > threshold)
        weights[keep] = 1. if mode == "binary" else terminal[keep]
    return weights, eligible


def curve_summary(times_s, velocity, valid, deadband=.15, min_duration=.4, signature=""):
    """Human review suggestions, not automatic saved grades or success labels."""
    t = np.asarray(times_s, float)
    v = np.asarray(velocity, float)
    ok = np.asarray(valid, bool) & np.isfinite(v)
    if not len(t):
        return dict(coverage=0., mean_velocity=None, suggestions=[])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.
    suggestions = []
    for name, mask in (("forward", ok & (v > deadband)), ("stall", ok & (np.abs(v) <= deadband)),
                       ("regression", ok & (v < -deadband))):
        for a, b in contiguous_runs(mask):
            stop = min(float(t[-1]), float(t[b - 1] + dt))
            if stop - t[a] < min_duration:
                continue
            sid = hashlib.sha256(f"{signature}:{name}:{t[a]:.9f}:{stop:.9f}".encode()).hexdigest()[:20]
            suggestions.append(dict(id=sid, kind=name, start_s=float(t[a]), end_s=stop,
                                    mean_velocity=float(v[a:b].mean())))
    return dict(coverage=float(ok.mean()), mean_velocity=float(v[ok].mean()) if ok.any() else None,
                forward_fraction=float((ok & (v > deadband)).sum() / max(1, ok.sum())),
                stall_fraction=float((ok & (np.abs(v) <= deadband)).sum() / max(1, ok.sum())),
                regression_fraction=float((ok & (v < -deadband)).sum() / max(1, ok.sum())),
                deadband=deadband, suggestions=sorted(suggestions, key=lambda s: s["start_s"]))


def json_signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
