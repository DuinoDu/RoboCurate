"""Frozen, provenance-checked DINO features from timestamped MCAP or RGB video."""
from pathlib import Path
import hashlib
import io
import json
import os
import uuid
import numpy as np
from PIL import Image
from .core import json_signature

BACKBONE_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
OPEN_BACKBONE_ID = "facebook/dinov2-small"
BACKBONES = {
    BACKBONE_ID: dict(dimension=768, model_type="dinov3_vit", directory="dinov3-vitb16",
                      provenance="backbone-provenance.json"),
    OPEN_BACKBONE_ID: dict(dimension=384, model_type="dinov2", directory="dinov2-small",
                           provenance="dinov2-small-provenance.json"),
}
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def preprocess_rgb(rgb, view="full", crop="squash", size=224):
    import cv2
    frame = np.asarray(rgb, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[-1] != 3 or view not in ("full", "left", "right"):
        raise ValueError("Invalid RGB frame or camera view")
    if view != "full":
        half = frame.shape[1] // 2
        frame = frame[:, :half] if view == "left" else frame[:, half:]
    if crop == "center":
        h, w = frame.shape[:2]
        side = min(h, w)
        frame = frame[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2]
        interpolation = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    elif crop == "squash":
        interpolation = cv2.INTER_LINEAR
    else:
        raise ValueError("Unknown image crop mode")
    if frame.shape[:2] != (size, size):
        frame = cv2.resize(frame, (size, size), interpolation=interpolation)
    return np.ascontiguousarray(((frame.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))


class DinoEncoder:
    def __init__(self, model_dir, device="cpu", checksums=None, backbone_id=BACKBONE_ID):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.device = device
        if backbone_id not in BACKBONES:
            raise ValueError("Unsupported visual backbone")
        self.dimension = BACKBONES[backbone_id]['dimension']
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").is_file():
            raise ValueError("图像模型尚未安装，请运行对应模型的准备脚本")
        if checksums:
            from .model import file_sha256
            for filename, digest in checksums.items():
                if Path(filename).name != filename or file_sha256(model_dir / filename) != digest:
                    raise ValueError('图像模型文件校验失败，请重新安装官方模型')
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
        if self.model.config.model_type != BACKBONES[backbone_id]['model_type'] or self.model.config.hidden_size != self.dimension:
            raise ValueError('Installed visual model does not match its feature contract')
        for p in self.model.parameters():
            p.requires_grad_(False)

    def __call__(self, batch):
        with self.torch.inference_mode():
            x = self.torch.from_numpy(np.stack(batch)).to(self.device)
            y = self.model(pixel_values=x).pooler_output.float().cpu().numpy()
        if y.shape != (len(batch), self.dimension) or not np.isfinite(y).all():
            raise ValueError("Visual encoder returned invalid features")
        return y


def feature_contract(profile, config):
    backbone = profile.get('backbone_id', BACKBONE_ID)
    if backbone not in BACKBONES:
        raise ValueError('Unsupported visual backbone')
    return dict(backbone=backbone, backbone_revision=profile["backbone_revision"],
                backbone_sha256=profile.get('backbone_sha256', {}),
                camera=profile["camera"], view=profile["view"], crop=profile["crop"],
                image_size=224, normalization="imagenet", fps=config.fps,
                feature_stride=config.feature_stride, dimension=BACKBONES[backbone]['dimension'],
                decoder="Pillow-RGB", resize="OpenCV", version=1)


def installed_backbone_profile(model_root, backbone_id, camera='head', view='left', crop='squash'):
    spec = BACKBONES[backbone_id]
    root = Path(model_root)
    provenance = json.loads((root/spec['provenance']).read_text())
    if provenance['repo'] != backbone_id:
        raise ValueError('Visual model provenance has a different repository ID')
    return dict(backbone_id=backbone_id, backbone_dir=str(root/spec['directory']),
                backbone_revision=provenance['revision'], backbone_sha256=provenance['sha256'],
                camera=camera, view=view, crop=crop)


def validate_training_contract(profile, config, checkpoint):
    """Prevent silent backbone/preprocessing substitutions at scoring time."""
    current = feature_contract(profile, config)
    trained = checkpoint.get('provenance', {}).get('feature_contract')
    if not trained:
        if current['backbone'] != BACKBONE_ID:
            raise ValueError('论文进度模型必须使用其原始 DINOv3 图像特征')
        return current
    transfer = profile.get('training_feature_contract')
    if transfer is not None and transfer != trained:
        raise ValueError('登记的迁移来源与进度模型训练记录不同')
    for key, value in current.items():
        if transfer is not None and key in ('camera', 'view', 'decoder'):
            continue
        if trained.get(key) != value:
            raise ValueError('进度模型与图像特征的训练来源或预处理不匹配：' + key)
    return current


def extract_mcap(source, meta, config, profile, cache_root, encoder, batch_size=8, progress=None):
    from mcap.reader import make_reader
    from server import _jpeg_payload
    source = Path(source)
    stat = source.stat()
    fingerprint = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    contract = feature_contract(profile, config)
    descriptor = dict(source=str(source.resolve()), fingerprint=fingerprint, contract=contract,
                      start_ns=meta["episode"]["start_ns"], end_ns=meta["episode"]["end_ns"])
    key = json_signature(descriptor)
    cache_path = Path(cache_root) / (key + ".npz")
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as a:
            return {k: a[k] for k in a.files}, descriptor, cache_path
    camera = profile["camera"]
    if camera not in meta["cameras"]:
        raise ValueError("模型指定的相机在本记录中不存在")
    start, end = meta["episode"]["start_ns"], meta["episode"]["end_ns"]
    count = int(np.ceil((end - start) / 1e9 * config.fps / config.feature_stride))
    times = start + np.rint(np.arange(count) * 1e9 * config.feature_stride / config.fps).astype(np.int64)
    times = times[times < end]
    features = np.full((len(times), contract['dimension']), np.nan, np.float32)
    source_times = np.full(len(times), -1, np.int64)
    valid = np.zeros(len(times), bool)
    pending, pending_ids = [], []
    cursor = 0

    def flush():
        if pending:
            vectors = encoder(pending)
            for ids, vector in zip(pending_ids, vectors):
                features[ids] = vector
                valid[ids] = True
            pending.clear()
            pending_ids.clear()
            if progress:
                progress(int(valid.sum()), len(times))

    def use_frame(stamp, payload, until):
        nonlocal cursor
        if stamp is None:
            return
        upper = int(np.searchsorted(times, until, side="left"))
        ix = np.arange(cursor, upper)
        cursor = upper
        ix = ix[(times[ix] >= stamp) & (times[ix] - stamp <= 100_000_000)]
        if not len(ix):
            return
        try:
            jpeg = _jpeg_payload(payload)
            if not jpeg:
                return
            with Image.open(io.BytesIO(jpeg)) as im:
                rgb = np.asarray(im.convert("RGB"))
            frame = preprocess_rgb(rgb, profile["view"], profile["crop"])
        except (ValueError, OSError):
            return
        source_times[ix] = stamp
        pending.append(frame)
        pending_ids.append(ix)
        if len(pending) >= batch_size:
            flush()

    stamp, payload = None, None
    with source.open("rb") as f:
        for _, _, msg in make_reader(f).iter_messages(topics=[meta["cameras"][camera]["topic"]]):
            if stamp is not None and msg.log_time != stamp:
                use_frame(stamp, payload, msg.log_time)
            stamp, payload = msg.log_time, msg.data  # last duplicate wins
    use_frame(stamp, payload, end)
    flush()
    if fingerprint != dict(size=source.stat().st_size, mtime_ns=source.stat().st_mtime_ns):
        raise ValueError("特征提取期间原始文件发生变化")
    arrays = dict(timestamp_ns=times, source_timestamp_ns=source_times, features=features, valid=valid)
    atomic_npz(cache_path, **arrays)
    atomic_json(cache_path.with_suffix(".json"), descriptor)
    return arrays, descriptor, cache_path


def extract_video(video, config, profile, output, encoder, batch_size=8, start_frame=0, n_frames=None, progress=None):
    """For an explicit episode slice in the public LeRobot video stream."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if abs(actual_fps - config.fps) > .05:
        cap.release()
        raise ValueError("Video FPS differs from label calibration")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = total - start_frame if n_frames is None else n_frames
    if start_frame < 0 or count < 1 or start_frame + count > total:
        cap.release()
        raise ValueError("Invalid episode video slice")
    indices = list(range(start_frame, start_frame + count, config.feature_stride))
    features, pending = [], []
    source_digest = hashlib.sha256()
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, start_frame + count):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"Video decode failed at frame {i}")
            source_digest.update(bgr.tobytes())
            if (i - start_frame) % config.feature_stride:
                continue
            pending.append(preprocess_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), profile["view"], profile["crop"]))
            if len(pending) >= batch_size:
                features.extend(encoder(pending)); pending.clear()
                if progress: progress(len(features), len(indices))
        if pending: features.extend(encoder(pending))
    finally:
        cap.release()
    arrays = dict(features=np.asarray(features, np.float32), valid=np.ones(len(features), bool),
                  timestamp_ns=np.rint((np.asarray(indices) - start_frame) * 1e9 / config.fps).astype(np.int64))
    atomic_npz(output, **arrays)
    contract = feature_contract(profile, config)
    contract["decoder"] = "OpenCV-RGB"
    atomic_json(Path(output).with_suffix(".json"), dict(source=str(Path(video).resolve()), contract=contract,
                                                       source_sha256=source_digest.hexdigest(), source_hash_kind='decoded BGR frame sequence of logical episode',
                                                       start_frame=start_frame, n_frames=count))
    return arrays
