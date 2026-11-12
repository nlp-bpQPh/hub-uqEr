"""Render measured results and the first held-out sentence's three answers."""
import csv,html,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import ROOT
out=ROOT/'results'
metrics=json.loads((out/'metrics.json').read_text())
history=list(csv.DictReader((out/'history.csv').open()))
fig,ax=plt.subplots(figsize=(8,4))
for key,title in [('train_nll','Training'),('val_nll','Validation')]:
    ax.plot([int(r['epoch']) for r in history],[float(r[key]) for r in history],label=title)
ax.set(yscale='log',xlabel='Epoch',ylabel='Mean answer loss (nats / character)',title='Week 05 | Supervised fine-tuning')
ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out/'learning_curve.png',dpi=150); plt.close(fig)
base,sft=[[json.loads(s) for s in (out/f'{name}_predictions.jsonl').read_text().splitlines()] for name in ('base','sft')]
sent=base[0]['input']
lines=['Week 05 | Zhechen Tu | Supervised fine-tuning',f"Week 04 Transformer | {metrics['epochs']} epochs | selected epoch {metrics['best_epoch']}",'Evaluation: held-out sentences / greedy generation / exact answer + newline','', 'Model       Exact matches       Answer-only NLL']
for name in ('base','sft'):
    m=metrics[name]; lines.append(f"{name:8}    {m['correct']:3} / {m['count']}           {m['response_nll']:.6f}")
lines+=['','First held-out sentence:',sent]
for a,b in zip(base,sft):
    if a['input']==sent:
        lines.append(f"{a['instruction']}  base={a['prediction']!r}  SFT={b['prediction']!r}  gold={b['response']!r}")
lines+=['','Scope: one English sentence template, known cue words and instructions.','This test measures new cue combinations, not general conversation.']
summary='\n'.join(lines)+'\n'
(out/'comparison.txt').write_text(summary)
(ROOT/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Week 05 | SFT evaluation</title><style>body{font:17px system-ui;max-width:1000px;margin:32px auto;padding:0 24px;color:#152636}pre{white-space:pre-wrap;background:#eff5fa;padding:20px;line-height:1.5}img{width:80%}</style><h1>Week 05 · Supervised fine-tuning</h1><pre>'+html.escape(summary)+'</pre><img src="results/learning_curve.png" alt="Training and validation answer loss"><p>Source: saved model predictions and training history.</p></html>')
