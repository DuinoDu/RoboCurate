#!/usr/bin/env python3
"""Install verified upstream weights locally. HF credentials stay in the SDK."""
import argparse
import json
from pathlib import Path
import shutil
import sys
APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from warp_progress.features import atomic_json, BACKBONE_ID
from warp_progress.model import file_sha256

HEAD_REPO = 'uynitsuj/warp-rm-sim-bottles-sss15'
HEAD_REVISION = '6417be056ed0bc2b2549f7f414fed76d4ca8501f'
HEAD_SHA256 = '9c74aa3934b12dd6b169f8945dc2501a8c625ef0becb9f27e4f297725e2c775f'


def prepare(workspace, register_only=False):
    from scripts.warp_network import configure_download_proxy
    if not register_only:
        configure_download_proxy()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.errors import GatedRepoError
    root = Path(workspace).resolve() / 'warp' / 'models'
    root.mkdir(parents=True, exist_ok=True)
    head = root / 'paper_sim_sss15.pt'
    if not head.is_file() and not register_only:
        downloaded = hf_hub_download(HEAD_REPO, 'warp_rm_sss15.pt', revision=HEAD_REVISION,
                                    cache_dir=root/'download-cache')
        shutil.copyfile(downloaded, head)
    if head.is_file() and file_sha256(head) != HEAD_SHA256:
        raise ValueError('Published WARP head SHA256 does not match its manifest')
    backbone = root / 'dinov3-vitb16'
    provenance_path = root / 'backbone-provenance.json'
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    profile = dict(id='paper-sim-sss15-head-left', label='WARP-RM · 论文模型 / 头部左目',
                   checkpoint=head.name, checkpoint_sha256=HEAD_SHA256,
                   checkpoint_repo=HEAD_REPO, checkpoint_revision=HEAD_REVISION,
                   backbone_dir=backbone.name, backbone_revision=provenance.get('revision', 'not-installed'),
                   backbone_sha256=provenance.get('sha256', {}), camera='head', view='left', crop='squash',
                   validation_status='transfer_unvalidated', task_scope='Sim Bottles top-camera task',
                   domain_note='论文模型学习的是仿真瓶子任务；当前 G1 头部视角为迁移试算，尚未验证任务判别效果。')
    registry = root/'profiles.json'
    def save_profile():
        models = json.loads(registry.read_text())['models'] if registry.exists() else []
        models = [p for p in models if p['id'] != profile['id']]
        atomic_json(registry, dict(version=1, models=[profile, *models]))
    save_profile()
    if register_only:
        return
    try:
        revision = provenance.get('revision') or HfApi().model_info(BACKBONE_ID).sha
        snapshot_download(BACKBONE_ID, revision=revision, local_dir=backbone,
                          allow_patterns=['config.json','model.safetensors','preprocessor_config.json','README.md','LICENSE*'])
    except GatedRepoError as exc:
        reason = exc.response.headers.get('x-error-message', '') if exc.response is not None else ''
        print('DINOv3 模型访问被拒绝：' + reason, file=sys.stderr)
        print('账号登录与模型访问是两层授权。请在模型页面确认访问申请已获批准；凭据保留在本机。', file=sys.stderr)
        raise SystemExit(3 if 'has been rejected' in reason.lower() else 2)
    checksums = {p.name:file_sha256(p) for p in (backbone/'config.json', backbone/'model.safetensors')}
    provenance = dict(repo=BACKBONE_ID, revision=revision, sha256=checksums,
                      obtained_via='Hugging Face authenticated SDK; token not stored in project',
                      license='DINOv3 upstream license; weights excluded from handoff archives')
    atomic_json(provenance_path, provenance)
    profile.update(backbone_revision=revision, backbone_sha256=checksums)
    save_profile()
    print(json.dumps(dict(ready=True, workspace=str(workspace), model_id=profile['id'],
                          backbone_revision=revision, checkpoint_sha256=HEAD_SHA256), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=APP/'workspace')
    p.add_argument('--register-only', action='store_true')
    args = p.parse_args()
    prepare(args.workspace, args.register_only)
