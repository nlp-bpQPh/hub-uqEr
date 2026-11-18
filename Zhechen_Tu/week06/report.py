"""Create tables/plots and inspect errors from measured predictions only."""
import csv
import html
import json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from data import ROOT,LABELS,load,tokens
from experiment import OUT,write_csv

summary=json.loads((OUT/'summary.json').read_text())
rows=list(csv.DictReader((OUT/'predictions.csv').open()))
reports=json.loads((OUT/'class_reports.json').read_text())
latency={r['method']:r for r in csv.DictReader((OUT/'cpu_inference.csv').open())}
selected={r['method']:r['representative_seed'] for r in summary}
test=load('test'); y=np.array([r['label'] for r in test]); ids={r['id']:r for r in test}
def preds(method):
    return [r for r in rows if r['method']==method and int(r['seed'])==selected[method]]
left=np.array([int(r['prediction']) for r in preds('LinearSVM')])==y
right=np.array([int(r['prediction']) for r in preds('LogisticRegression')])==y
wins,losses=int((left & ~right).sum()),int((right & ~left).sum())
n=len(y);rng=np.random.default_rng(20261118)
counts=rng.multinomial(n,[wins/n,losses/n,1-(wins+losses)/n],size=10000)
diff=(counts[:,0]-counts[:,1])/n
paired=dict(comparison='LinearSVM minus LogisticRegression',accuracy_difference=float(left.mean()-right.mean()),
  paired_bootstrap_95_percentile_interval=np.quantile(diff,[.025,.975]).tolist(),
  bootstrap_draws=10000,svm_only_correct=wins,lr_only_correct=losses,
  mcnemar_exact_two_sided_p=float(binomtest(wins,wins+losses,.5).pvalue),note='Exploratory pair, not adjusted for model comparison multiplicity.')
(OUT/'paired_comparison.json').write_text(json.dumps(paired,indent=2)+'\n')
lengths=[len(tokens(r['text'])) for r in test]
(OUT/'token_length.json').write_text(json.dumps(dict(test_over_100=sum(n>100 for n in lengths),test_total=n,max_tokens=max(lengths)),indent=2)+'\n')
errors=[]
for method in selected:
    for r in preds(method):
        if r['prediction']!=r['label']:
            original=ids[r['id']]
            errors.append(dict(method=method,seed=selected[method],id=r['id'],gold=LABELS[int(r['label'])],prediction=LABELS[int(r['prediction'])],text=original['text']))
(OUT/'errors.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in errors))
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
labels=[r['method'] for r in summary]
axes[0].bar(labels,[100*r['f1_mean'] for r in summary],yerr=[100*(r['f1_sd'] or 0) for r in summary],capsize=4,color=['#6988a6']*3+['#218b83']*2)
axes[0].set(ylabel='Test macro F1 (%)',ylim=(0,100),title='Same test set | neural: mean ± SD of 3 seeds')
for i,r in enumerate(summary):axes[0].text(i,100*r['f1_mean']+2,f'{100*r["f1_mean"]:.2f}',ha='center',fontsize=9)
axes[1].bar(labels,[float(latency[k]['cpu_seconds_median']) for k in labels],color='#6988a6')
axes[1].set(ylabel='Seconds / 7600 texts',title='CPU inference including preprocessing\nMedian of 3 warm runs; loading excluded')
for ax in axes:ax.tick_params(axis='x',labelrotation=35);ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,method in zip(axes,['LinearSVM','TextCNN']):
    cm=np.array(reports[f'{method}_{selected[method]}']['confusion_matrix'])
    ax.imshow(cm,cmap='Blues');ax.set(xticks=range(4),yticks=range(4),xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted',ylabel='True',title=(f'{method} | single fit' if method=='LinearSVM' else f'{method} | seed {selected[method]}'))
    for i in range(4):
        for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
fig.tight_layout();fig.savefig(OUT/'confusion_matrices.png',dpi=160);plt.close(fig)
lines=['Week 06 | Zhechen Tu | AG News classification comparison','Train 12000 / Validation 2000 / Test 7600','Neural: 3 seeds, random embeddings, best validation epoch','', 'Method                 Accuracy     Macro F1       SD(F1)   CPU seconds']
for r in summary:
    sd='n/a' if r['f1_sd'] is None else f'{100*r["f1_sd"]:.3f}'
    lines.append(f'{r["method"]:22} {100*r["accuracy_mean"]:6.2f}%      {100*r["f1_mean"]:6.2f}%     {sd:>6}     {float(latency[r["method"]]["cpu_seconds_median"]):.3f}')
lo,hi=paired['paired_bootstrap_95_percentile_interval']
lines+=['',f'SVM-LR accuracy gap: {100*paired["accuracy_difference"]:.3f} percentage points',f'Paired bootstrap 95% interval: [{100*lo:.3f}, {100*hi:.3f}] pp',f'Exact McNemar p={paired["mcnemar_exact_two_sided_p"]:.4f}', 'Raw numerical ranking does not establish universal superiority.']
(OUT/'comparison.txt').write_text('\n'.join(lines)+'\n')
(ROOT/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Week 06 results</title><style>body{max-width:1150px;margin:36px auto;font:16px system-ui;color:#173044}pre{padding:24px;background:#eef4f8;white-space:pre-wrap;line-height:1.7}img{width:100%}</style><h1>Week 06 · Classification methods</h1><pre>'+html.escape('\n'.join(lines))+'</pre><img src="results/comparison.png" alt="Comparison"><img src="results/confusion_matrices.png" alt="Confusion matrices"></html>')
print('\n'.join(lines))
