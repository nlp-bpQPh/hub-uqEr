"""Check split isolation, every checkpoint's predictions, and fair CPU latency."""
from collections import Counter
import csv
import hashlib
import json
import statistics
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from data import load,normalize,ROOT
from experiment import OUT,metric,write_csv
from predict import restore,predict_many

def main():
    torch.set_num_threads(4)
    rows={s:load(s) for s in ('train','val','test')}
    meta=json.loads((ROOT/'data/manifest.json').read_text())
    keys=[]
    for split,data in rows.items():
        texts=[normalize(r['text']) for r in data]
        assert len(texts)==len(set(texts))
        assert len(set(Counter(r['label'] for r in data).values()))==1
        assert hashlib.sha256('\n'.join(r['id'] for r in data).encode()).hexdigest()==meta['split_ids_sha256'][split]
        keys.append(set(texts))
    assert all(keys[i].isdisjoint(keys[j]) for i in range(3) for j in range(i))
    print('PASS: split-ID hashes, class balance, normalized exact-text deduplication, no split overlap.')
    runs=list(csv.DictReader((OUT/'runs.csv').open()))
    stored=list(csv.DictReader((OUT/'predictions.csv').open()))
    history=list(csv.DictReader((OUT/'history.csv').open()))
    test=rows['test'];y=np.array([r['label'] for r in test]);texts=[r['text'] for r in test]
    for run in runs:
        name,seed=run['method'],int(run['seed'])
        model,feat=restore(name,seed)
        pred=predict_many(name,model,feat,texts)
        saved=[p for p in stored if p['method']==name and int(p['seed'])==seed]
        assert [p['id'] for p in saved]==[r['id'] for r in test]
        assert np.array_equal(pred,[int(p['prediction']) for p in saved])
        for k,v in metric(y,pred).items():assert abs(v-float(run[k]))<1e-12
        if seed:
            h=[h for h in history if h['method']==name and int(h['seed'])==seed]
            best=max(h,key=lambda h:float(h['val_macro_f1']))
            assert int(run['selected_epoch'])==int(best['epoch'])
        print(f'PASS: {name} seed={seed} CPU reload reproduces all {len(test)} predictions and metrics.')
    # Same CPU, same 7600 texts, preprocessing included; three warm runs, loading excluded.
    times=[]
    for summary in json.loads((OUT/'summary.json').read_text()):
        name=summary['method'];seed=summary['representative_seed'];model,feat=restore(name,seed)
        predict_many(name,model,feat,texts[:64])
        durations=[]
        for _ in range(3):
            start=time.perf_counter();predict_many(name,model,feat,texts);durations.append(time.perf_counter()-start)
        times.append(dict(method=name,seed=seed,cpu_seconds_median=statistics.median(durations),
                          milliseconds_per_text=statistics.median(durations)*1000/len(texts)))
    write_csv(OUT/'cpu_inference.csv',times)
    print('PASS: CPU end-to-end inference timing recorded for all five methods (median of 3).')
    print('ALL CHECKS PASSED')

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
