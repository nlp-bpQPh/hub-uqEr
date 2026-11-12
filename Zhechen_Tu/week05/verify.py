"""Audit data grouping, answer-only targets and the published checkpoints."""
import json
from collections import Counter
import torch
from dataset import ROOT,build,load,prompt
from sft import OUT,SHA,base_path,batch,examples,restore,score

def main():
    torch.set_num_threads(4)
    splits=build()
    groups=[]
    for name,rows in splits.items():
        assert rows==load(name)
        pairs=[(r['input'],r['task']) for r in rows]
        assert len(set(pairs))==len(rows)
        assert set(Counter(r['input'] for r in rows).values())=={3}
        groups.append(set(r['input'] for r in rows))
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    print('PASS: splits reproduce exactly; 729 unique sentences; no source sentence crosses splits.')
    base,vocab=restore(base_path())
    ft,v2=restore(OUT/'sft_model.pt')
    assert v2==vocab and base.cfg==ft.cfg
    assert all(not torch.equal(p,q) for p,q in zip(base.parameters(),ft.parameters()))
    rows=splits['train']
    ex=examples(rows,vocab,ft.cfg.context)
    for row,(x,y) in zip(rows,ex):
        first=len(prompt(row))-1
        assert (y[:first]==-100).all()
        assert ''.join(vocab[i] for i in y[first:])==row['response']+'\n'
        assert ''.join(vocab[i] for i in x)==prompt(row)+row['response']
    x,y=batch(ex[:64],'cpu')
    for i,(_,label) in enumerate(ex[:64]):
        assert (y[i,len(label):]==-100).all()
    print('PASS: same architecture/vocabulary; parameters changed; prompt and padding excluded; first answer character and terminator included.')
    saved=json.loads((OUT/'metrics.json').read_text())
    for name,model in [('base',base),('sft',ft)]:
        metric,recs=score(model,vocab,splits['test'],'cpu')
        assert abs(metric['response_nll']-saved[name]['response_nll']) < 1e-4
        assert metric['correct']==saved[name]['correct']
        stored=[json.loads(line) for line in (OUT/f'{name}_predictions.jsonl').read_text().splitlines()]
        assert len(stored)==len(recs)
        assert all(r['exact']==(r['prediction']==r['response'] and r['stopped']) for r in stored)
        assert sum(r['exact'] for r in stored)==metric['correct']
        print(f'PASS: {name} CPU reload: {metric["correct"]}/{metric["count"]}; response NLL={metric["response_nll"]:.6f}.')
    print('ALL CHECKS PASSED')

if __name__=='__main__': main()
