"""Standalone research figure, generated entirely from saved measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'stage_training/test_report.json').read_text())
v=json.loads((ROOT/'stage_training/validation.json').read_text())
names=['head_mlp','dual_mlp','dual_temporal','dual_temporal_rewind']
labels=['Head MLP','Dual MLP\n(selected by validation)','Dual Transformer','Dual Transformer\n+ rewind']
fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
x=np.arange(4)
axes[0].bar(x-.18,[v[n]['mean_validation_f1'] for n in names],.34,label='Validation mean (3 seeds)',color='#96abc5')
axes[0].bar(x+.18,[r['results'][n]['macro_f1'] for n in names],.34,label='Test ensemble (70 points)',color='#267e88')
axes[0].set(xticks=x,xticklabels=labels,ylim=(0,1.08),ylabel='Macro F1',title='Whole-recording split; no test-based selection')
axes[0].tick_params(axis='x',labelsize=8);axes[0].legend(loc='lower right',fontsize=8)
for i,n in enumerate(names):axes[0].text(i+.18,r['results'][n]['macro_f1']+.015,f"{r['results'][n]['macro_f1']:.3f}",ha='center',fontsize=8)
cm=np.array(r['results']['selected_calibrated']['confusion'])
axes[1].imshow(cm,cmap='Blues',vmin=0)
phases=['Approach','Grasp','Carry','Place','Post-release']
axes[1].set(xticks=np.arange(5),yticks=np.arange(5),xticklabels=phases,yticklabels=phases,
            xlabel='Predicted stage',ylabel='Provisional visual annotation',title='Selected model: 63 / 70 correct')
axes[1].tick_params(axis='x',rotation=30,labelsize=8)
for i in range(5):
 for j in range(5):axes[1].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>12 else '#123')
fig.suptitle('G1 banana transport | one session, 5 held-out recordings | not task-success accuracy',fontsize=11)
fig.savefig(ROOT/'stage_training/comparison.png',dpi=180)
fig.savefig(ROOT/'stage_training/comparison.svg')
print('Wrote comparison.png and comparison.svg')
