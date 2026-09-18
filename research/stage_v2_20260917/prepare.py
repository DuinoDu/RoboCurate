"""Freeze the next development experiment before fitting; previous test is regression only."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;APP=ROOT.parents[1];PREV=ROOT.parent/'model_upgrade_20260917'
sys.path.insert(0,str(APP))
import numpy as np
from warp_progress.features import atomic_json,atomic_npz
from warp_progress.state_features import state_contract,read_state

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if (ROOT/'protocol.json').exists():raise ValueError('Protocol already frozen')
    manifest=json.loads((PREV/'features_manifest.json').read_text())
    baseline=json.loads((PREV/'installed_model.json').read_text())
    protocol=dict(version=2,annotation_sha256=sha(PREV/'annotations.v1.json'),
        source_feature_manifest_sha256=sha(PREV/'features_manifest.json'),baseline_checkpoint_sha256=baseline['checkpoint_sha256'],
        development=[126,128,129,130,131,132,133,134,135,136],
        folds=[[126,132],[128,134],[129,135],[130,136],[131,133]],
        historical_regression=[137,138,139,140,141],stress=[124,125,127],
        candidates=['baseline_mlp','baseline_temporal','fusion_mlp','fusion_temporal','fusion_tokens','fusion_tokens_aug'],
        seeds=[17,29,43],steps=300,batch_size=64,lr=.0003,weight_decay=.001,
        checkpoints='Mean parameter weights at steps 200, 250, 300; no early stopping on held-out fold.',
        input='2 Hz frozen DINOv2 head/right-wrist, optional 41 measured q values; no timestamps/duration/operator outcomes as model inputs.',
        augmentation='Augmented candidate: per-window single-camera dropout 20%, state dropout 15%, temporal stride 1 or 2. Other candidates unaugmented.',
        selection='Highest pooled out-of-fold sparse-point macro F1, tie by lower NLL. Each fold fits preprocessing/priors/weights on its other 8 episodes only.',
        promotion='New fusion model must exceed retrained baseline_mlp OOF macro F1 by >=0.02 and must not trail baseline_temporal by >0.01. Otherwise keep current model; preserve candidates.',
        calibration='Final temperature fitted to development OOF ensemble probabilities; calibration metrics are development diagnostics, not independent test.',
        regression_policy='Previously observed test clips are used only after selection, reported as historical regression. No new independent test claim.',
        label_status='Unchanged sparse assistant visual annotations, not independently human reviewed.',
        terminal_outcome_policy='Insufficient independent failure episodes; no final success head is promoted this round.',
        sources=['https://qianzhong-chen.github.io/sarm.github.io/',
                 'https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.html'])
    atomic_json(ROOT/'protocol.json',protocol)
    atomic_json(ROOT/'freeze.json',dict(protocol_sha256=sha(ROOT/'protocol.json')))
    records=[];contract=None
    for row in manifest['episodes']:
        meta_path=Path(row['meta']);meta=json.loads(meta_path.read_text());current=state_contract(meta)
        if contract is None:contract=current
        assert current==contract
        with np.load(row['cameras']['head']['cache'],allow_pickle=False) as z:times=z['timestamp_ns']
        native=meta_path.parent/'native.npz'
        states=read_state(native,times,meta,contract)
        out=ROOT/'states'/(row['episode']+'.npz')
        atomic_npz(out,timestamp_ns=times,**states)
        records.append(dict(episode=row['episode'],path=str(out),sha256=sha(out),valid=int(states['state_valid'].sum()),
            total=len(times),native_path=str(native),native_sha256=sha(native),source_sha256=row['source_sha256']))
    atomic_json(ROOT/'state_manifest.json',dict(contract=contract,episodes=records))
    with zipfile.ZipFile(ROOT/'source_before_v2.zip','x',zipfile.ZIP_DEFLATED) as z:
        for folder in ['warp_progress','static','tests']:
            for p in (APP/folder).rglob('*'):
                if p.is_file() and p.suffix in {'.py','.js','.mjs','.css','.html'}:z.write(p,p.relative_to(APP))
        for p in APP.glob('*.py'):z.write(p,p.name)
    print('Frozen 5 folds and extracted state features for',len(records),'recordings')

if __name__=='__main__':main()
