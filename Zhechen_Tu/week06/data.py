"""Versioned AG News download; exact-text deduplication and fixed splits."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata
import urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parent
LABELS=['World','Sports','Business','SciTech']
URL='https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/'
SEED=20261118

def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC',html.unescape(text)).lower().split())

def tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+",normalize(text))

def load(split):
    return [json.loads(line) for line in (ROOT/'data'/f'{split}.jsonl').read_text().splitlines()]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,default=ROOT/'cache')
    a=p.parse_args(); a.cache.mkdir(exist_ok=True,parents=True)
    hashes={}; original={}; raw={}
    manifest_path=ROOT/'data/manifest.json'
    old=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    for split in ('train','test'):
        file=a.cache/f'{split}.csv'
        if not file.exists(): urllib.request.urlretrieve(URL+f'{split}.csv',file)
        hashes[split]=hashlib.sha256(file.read_bytes()).hexdigest()
        if old and hashes[split]!=old['source_sha256'][split]: raise ValueError('Upstream source changed')
        rows=[]
        with file.open(newline='',encoding='utf-8') as f:
            for i,(label,title,description) in enumerate(csv.reader(f)):
                text=title+' '+description
                rows.append(dict(id=f'{split}_{i}',label=int(label)-1,text=text))
        raw[split]=rows; original[split]=len(rows)
    # Keep first normalized exact duplicate. Remove train/test overlaps from training,
    # preserving evaluation priority. Grouping uses text only, not test labels.
    def unique(rows):
        seen=set(); kept=[]
        for r in rows:
            key=normalize(r['text'])
            if key and key not in seen:
                kept.append(r); seen.add(key)
        return kept
    test=unique(raw['test']); testkeys={normalize(r['text']) for r in test}
    unique_train=unique(raw['train'])
    pool=[r for r in unique_train if normalize(r['text']) not in testkeys]
    rng=np.random.default_rng(SEED); train=[]; val=[]
    for label in range(4):
        group=[r for r in pool if r['label']==label]; rng.shuffle(group)
        train.extend(group[:3000]); val.extend(group[3000:3500])
    rng.shuffle(train); rng.shuffle(val)
    splits=dict(train=train,val=val,test=test)
    target=ROOT/'data'; target.mkdir(exist_ok=True)
    for name,rows in splits.items():
        (target/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    meta=dict(source=URL,source_sha256=hashes,seed=SEED,original_counts=original,
        train_exact_duplicates=len(raw['train'])-len(unique_train),test_exact_duplicates=len(raw['test'])-len(test),
        train_test_overlap_removed=len(unique_train)-len(pool),eligible_train_pool=len(pool),
        counts={s:len(r) for s,r in splits.items()},class_counts={s:dict(Counter(x['label'] for x in r)) for s,r in splits.items()},
        split_ids_sha256={s:hashlib.sha256('\n'.join(x['id'] for x in r).encode()).hexdigest() for s,r in splits.items()})
    manifest_path.write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
