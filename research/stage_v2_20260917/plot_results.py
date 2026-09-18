"""Export source-backed scientific figures; no new model evaluation or tuning."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent;PREV=ROOT.parent/'model_upgrade_20260917'
cv=json.loads((ROOT/'training/cross_validation.json').read_text())
r=json.loads((ROOT/'training/report.json').read_text())
old=json.loads((PREV/'stage_training/test_report.json').read_text())['results']['selected_calibrated']
names=list(cv);labels=['Visual MLP','Visual temporal','Fusion MLP','Fusion temporal','Fusion + tokens','Tokens + augmentation']
fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
x=np.arange(len(names));scores=[cv[n]['overall']['macro_f1'] for n in names]
axes[0].barh(x,scores,color=['#238c8a' if n==r['selection']['winner'] else '#98aabd' for n in names])
axes[0].set(yticks=x,yticklabels=labels,xlim=(0,1.07),xlabel='Macro F1',title='Development: 5 folds, 10 recordings, 139 points')
axes[0].invert_yaxis()
for i,v in enumerate(scores):axes[0].text(v+.01,i,f'{v:.3f}',va='center',fontsize=9)
episodes=['137','138','139','140','141'];x=np.arange(5)
a=[old['per_episode'][e]['accuracy'] for e in episodes];b=[r['historical_regression']['per_episode'][e]['accuracy'] for e in episodes]
axes[1].bar(x-.19,a,.36,label='Installed v1',color='#98aabd')
axes[1].bar(x+.19,b,.36,label='Selected v2',color='#238c8a')
axes[1].set(xticks=x,xticklabels=episodes,ylim=(0,1.13),xlabel='Recording (14 annotated points each)',ylabel='Stage accuracy',title='Previously seen recordings: regression only')
axes[1].legend(loc='lower right')
for i,v in enumerate(b):axes[1].text(i+.19,v+.015,f'{round(v*14)}/14',ha='center',fontsize=9)
fig.suptitle('G1 banana transport | provisional sparse labels, one session | no task-success claim',fontsize=11)
for suffix in ['png','svg']:fig.savefig(ROOT/('comparison.'+suffix),dpi=180)
print('Wrote comparison.png and comparison.svg')
