"""Freeze provisional visual annotations and a whole-recording evaluation protocol."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
APP = ROOT.parents[1]
# Each entry corresponds to the 15 timestamped, visually inspected head frames.
# 4 means post-release observation, NOT verified task success; -1 means unknown.
LABELS = {}  # Populated from a local label_map.json in main().

def write(name, value):
    path = ROOT / name
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite frozen study file: {path}')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    global LABELS
    LABELS = {int(k):v for k,v in json.loads((ROOT/'label_map.json').read_text()).items()}
    records = []
    for i in range(124, 142):
        episode = f'episode_{i:06d}'
        idx = json.loads((ROOT/'local_evidence'/episode/'index.json').read_text())
        points = []
        for f, stage in zip(idx['frames'], LABELS.get(i, [-1]*15)):
            points.append(dict(time_s=f['time_s'], stage=stage, image=f['file'],
                               source_timestamp_ns=f['source_timestamp_ns']))
        notes = {124:'Left table to right table; no basket. Task variant, excluded from fitting and ordinary test.',
                 125:'Right table to left table reset/reverse task, not a labeled grasp failure.',
                 127:'Recovery stress case: banana visibly outside basket at 30.56, 34.39, 38.18 s; regrasp then replacement.',
                 126:'Last sampled frame contains operator reset; unknown.',
                 141:'Post-release banana rests partly against basket rim; terminal success remains unverified.'}
        records.append(dict(episode=episode, source=idx['source'],
                            source_fingerprint=idx['source_fingerprint'], points=points,
                            terminal_success=None, note=notes.get(i, 'Compatible task; sampled phase labels only.')))
    annotation_hash = write('annotations.v1.json', dict(
        version=1, author='assistant visual inspection', status='provisional_not_human_reviewed',
        evidence='15 timestamped head-camera frames per recording; sparse visual inspection, not dense expert annotation',
        stage_names=['接近左桌','抓取香蕉','持物搬运','向篮子放置','释放后观察'],
        semantics='Phase 4 is a visible post-release state, never an automatic success label. Negative labels are unknown.',
        interpolation='Training only: between adjacent equal known phase labels, plus nearest 2 Hz sample within 0.26 s of a known point. Other boundaries stay unknown. Evaluation only uses original known points.',
        episodes=records))
    protocol_hash = write('protocol.v1.json', dict(
        version=1, frozen_before_training=True, label_status='provisional_not_human_reviewed',
        annotation_sha256=annotation_hash,
        train=[126,128,129,130,131,132,133,134], validation=[135,136], test=[137,138,139,140,141], stress=[124,125,127],
        split_unit='entire source MCAP recording; later five ordinary recordings held out chronologically',
        sampling_hz=2, context_frames=8, feature_backbone='facebook/dinov2-small',
        candidates=['head_mlp','dual_mlp','dual_temporal','dual_temporal_rewind'],
        seeds=[17,29,43], train_steps=500, validate_every=25,
        selection='Max mean validation macro F1 across three seeds, then min validation NLL; best checkpoint per seed chosen only by validation. No hyperparameter changes based on test.',
        primary='Five-class macro F1 on original known sparse points, with per-episode and confusion matrix reporting.',
        secondary=['balanced accuracy','accuracy','negative log likelihood','Brier score','expected calibration error','uncertainty review coverage','stage-aligned progress proxy MAE'],
        controls=['train-only elapsed-time phase prior','head-only MLP','dual-view MLP','temporal no-rewind ablation','reversed input order with labels tied to visual state','shuffled visual input','constant visual input'],
        prohibitions=['No test-based model selection','No terminal success claim','No source operator flags as phase truth','No policy improvement claim without policy trials'],
        limitation='One collection session, 15 ordinary episodes, sparse assistant annotations. This is a development benchmark, not independent expert validation or cross-session generalization.'))
    write('freeze.json', dict(annotation_sha256=annotation_hash, protocol_sha256=protocol_hash))
    # Compact source rollback point; never package raw videos, envs, tokens or reviews.
    with zipfile.ZipFile(ROOT/'pre_upgrade_sources.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for folder in ['warp_progress','scripts','tests','static']:
            for p in (APP/folder).rglob('*'):
                if p.is_file() and p.suffix in {'.py','.js','.mjs','.css','.html'}:
                    z.write(p, p.relative_to(APP))
        for p in APP.glob('*.py'):
            z.write(p, p.name)
    print(json.dumps(dict(annotation_sha256=annotation_hash, protocol_sha256=protocol_hash)))

if __name__ == '__main__':
    main()
