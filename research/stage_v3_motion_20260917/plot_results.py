"""Source-backed comparison figure; no independent-test or significance claim."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    first=json.loads((ROOT.parent/'stage_v3_20260917/training/cross_validation.json').read_text())
    second=json.loads((ROOT/'training/cross_validation.json').read_text())
    names=['v2 reference','v2 + soft labels','Long context','Long context + soft labels','Motion differences','Motion + soft labels (v3)']
    scores=[first[k]['overall']['macro_f1']*100 for k in ['v2_reference','v2_rw','multiscale','multiscale_rw']]
    scores += [second[k]['overall']['macro_f1']*100 for k in ['motion','motion_rw']]
    guard=json.loads((ROOT/'review_guard/report.json').read_text())
    old=json.loads((ROOT.parent/'stage_v2_20260917/training/report.json').read_text())['historical_regression']
    old_curves=json.loads((ROOT.parent/'stage_v2_20260917/training/regression_curves.json').read_text())
    unflagged=sum(c['stage'][i]!=y and not c['review'][i] for c in old_curves.values()
                  for i,y in zip(c['annotation_indices'],c['annotation_stages']))
    hist=[dict(correct=round(old['n']*old['accuracy']),reviewed=round(old['n']*old['review_fraction']),unflagged_mistakes=unflagged)]
    hist += [dict(x,correct=x['n']-x['errors']) for x in [guard['historical_before'],guard['historical_after']]]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,6.2),gridspec_kw={'width_ratios':[1.25,1]})
    ax=axes[0];y=np.arange(6);colors=['#7a8696']*5+['#167c80']
    ax.barh(y,scores,color=colors,height=.62);ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,110)
    ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Macro F1 (%)');ax.set_title('Development: 5 folds / 139 sparse points',loc='left',pad=14)
    ax.axvline(scores[0]+1,color='#99703e',linestyle='--',linewidth=1,label='Required F1 gain: +1 point')
    for i,s in enumerate(scores):ax.text(108,i,f'{s:.2f}',va='center',ha='right',fontsize=9)
    ax=axes[1];x=np.arange(3)
    for offset,key,label,color in [(-.25,'correct','Correct stage','#167c80'),(0,'reviewed','Review requested','#cf9f51'),(.25,'unflagged_mistakes','Unflagged error','#bb5b60')]:
        vals=[h[key] for h in hist];bars=ax.bar(x+offset,vals,.23,label=label,color=color)
        ax.bar_label(bars,padding=3,fontsize=9)
    ax.set_ylim(0,78);ax.set_yticks([0,14,28,42,56,70]);ax.set_ylabel('Points (out of 70)')
    ax.set_xticks(x,['v2','v3 before\nreview guard','v3 released'])
    ax.set_title('Previously seen 5-recording regression',loc='left',pad=14)
    handles=[];labels=[]
    for axis in axes:
        h,l=axis.get_legend_handles_labels();handles.extend(h);labels.extend(l)
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.91),ncol=4,frameon=False,fontsize=9)
    fig.suptitle('HoloCurate v3: a development gain; historical stage accuracy unchanged',x=.04,ha='left',fontsize=13)
    fig.text(.04,.025,'Same recording session; provisional visual annotations; repeated development. No new blind test or task-success accuracy.',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.07,1,.84));fig.savefig(ROOT/'comparison.png',dpi=180);fig.savefig(ROOT/'comparison.svg');plt.close(fig)
if __name__=='__main__':main()
