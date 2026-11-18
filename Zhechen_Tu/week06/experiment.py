"""Controlled comparison: fixed splits, frozen design, validation-only selection."""
import argparse
from collections import Counter
import copy
import csv
import json
import random
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F
from data import ROOT,LABELS,load,normalize,tokens

SEEDS=[17,29,43]
MAXLEN=100
EMBED=64
OUT=ROOT/'results'

def encode(rows,vocab):
    x=torch.zeros((len(rows),MAXLEN),dtype=torch.long)
    for i,r in enumerate(rows):
        ids=[vocab.get(t,1) for t in tokens(r['text'])[:MAXLEN]] or [1]
        x[i,:len(ids)]=torch.tensor(ids)
    return x

class Network(nn.Module):
    def __init__(self,kind,nwords):
        super().__init__(); self.kind=kind
        self.embed=nn.Embedding(nwords,EMBED,padding_idx=0)
        self.dropout=nn.Dropout(.3)
        if kind=='MeanPool':
            self.fc=nn.Sequential(nn.Linear(EMBED,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,4))
        else:
            self.convs=nn.ModuleList([nn.Conv1d(EMBED,64,k) for k in (3,4,5)])
            self.fc=nn.Linear(192,4)
    def forward(self,x):
        e=self.embed(x); mask=x.ne(0)
        if self.kind=='MeanPool':
            h=(e*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        else:
            # Include windows whose first position is real; exclude padding-only windows.
            z=self.dropout(e).transpose(1,2)
            pooled=[]
            for conv in self.convs:
                h=F.relu(conv(z)); length=h.shape[-1]
                h=h.masked_fill(~mask[:,:length].unsqueeze(1),-1e4)
                pooled.append(h.max(2).values)
            h=torch.cat(pooled,1)
        return self.fc(self.dropout(h))

@torch.inference_mode()
def logits(model,x,device):
    model.eval()
    return torch.cat([model(x[i:i+256].to(device)).cpu() for i in range(0,len(x),256)])

def metric(y,p):
    return dict(accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,average='macro')))

def sync(device):
    if device=='mps': torch.mps.synchronize()
    elif device=='cuda': torch.cuda.synchronize()

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def run(device,epochs):
    OUT.mkdir(exist_ok=True,parents=True); (OUT/'models').mkdir(exist_ok=True)
    torch.set_num_threads(4)
    train,val=load('train'),load('val')
    yt,yv=[np.array([r['label'] for r in rows]) for rows in (train,val)]
    text_train,text_val=[[normalize(r['text']) for r in rows] for rows in (train,val)]
    begin=time.perf_counter()
    tfidf=TfidfVectorizer(ngram_range=(1,2),max_features=40000,min_df=2,sublinear_tf=True,dtype=np.float32)
    xt=tfidf.fit_transform(text_train); xv=tfidf.transform(text_val)
    feature_seconds=time.perf_counter()-begin
    joblib.dump(tfidf,OUT/'models/tfidf.joblib',compress=3)
    plans=[]; history=[]
    print('Week 06 | Zhechen Tu | AG News four-class comparison',flush=True)
    print(f'Train={len(train)} val={len(val)} TF-IDF={xt.shape[1]} features; device={device}',flush=True)
    factories={'NaiveBayes':lambda:MultinomialNB(alpha=.5),
               'LogisticRegression':lambda:LogisticRegression(C=4,max_iter=500,solver='lbfgs'),
               'LinearSVM':lambda:LinearSVC(C=1,dual='auto',max_iter=5000,random_state=17)}
    for name,make in factories.items():
        start=time.perf_counter()
        model=make();model.fit(xt,yt)
        seconds=time.perf_counter()-start
        vm=metric(yv,model.predict(xv))
        path=OUT/f'models/{name}.joblib';joblib.dump(model,path,compress=3)
        plans.append(dict(method=name,seed=0,fit_seconds=seconds,val_macro_f1=vm['macro_f1'],selected_epoch=0,path=str(path)))
        print(f'{name}: val F1={vm["macro_f1"]:.4f}, fit {seconds:.2f}s',flush=True)
    start=time.perf_counter()
    counts=Counter(t for r in train for t in tokens(r['text']))
    vocab={'<PAD>':0,'<UNK>':1}
    vocab.update({t:i+2 for i,(t,n) in enumerate(counts.most_common(14998))})
    (OUT/'models/vocab.json').write_text(json.dumps(vocab)+'\n')
    ntx,nvx=encode(train,vocab),encode(val,vocab)
    neural_feature_seconds=time.perf_counter()-start
    for kind in ('MeanPool','TextCNN'):
        for seed in SEEDS:
            random.seed(seed); np.random.seed(seed);torch.manual_seed(seed)
            model=Network(kind,len(vocab)).to(device)
            opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
            rng=torch.Generator().manual_seed(seed)
            best=-1.; best_state=None
            sync(device);start=time.perf_counter()
            for epoch in range(1,epochs+1):
                model.train(); order=torch.randperm(len(yt),generator=rng); loss_sum=0.
                for i in range(0,len(yt),128):
                    ids=order[i:i+128]; x=ntx[ids].to(device); y=torch.tensor(yt[ids.numpy()],device=device)
                    opt.zero_grad(set_to_none=True); pred=model(x); loss=F.cross_entropy(pred,y)
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                    loss_sum+=loss.item()*len(ids)
                vl=logits(model,nvx,device); vm=metric(yv,vl.argmax(1).numpy())
                history.append(dict(method=kind,seed=seed,epoch=epoch,train_loss=loss_sum/len(yt),val_macro_f1=vm['macro_f1']))
                if vm['macro_f1']>best:
                    best,best_epoch=vm['macro_f1'],epoch
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                print(f'{kind} seed={seed} epoch={epoch:02d} train CE={loss_sum/len(yt):.4f} val F1={vm["macro_f1"]:.4f}',flush=True)
            sync(device);seconds=time.perf_counter()-start
            path=OUT/f'models/{kind}_{seed}.pt'
            torch.save(dict(kind=kind,nwords=len(vocab),state=best_state,seed=seed,epoch=best_epoch,maxlen=MAXLEN,embed=EMBED),path)
            plans.append(dict(method=kind,seed=seed,fit_seconds=seconds,val_macro_f1=best,selected_epoch=best_epoch,path=str(path)))
    # Test labels are loaded only after every model's training/selection is complete.
    test=load('test'); ytest=np.array([r['label'] for r in test])
    test_text=[normalize(r['text']) for r in test]
    test_sparse=tfidf.transform(test_text); test_dense=encode(test,vocab)
    records=[]; predictions=[]; reports={}
    for plan in plans:
        path=Path(plan['path']); kind=plan['method']
        if kind in factories:
            model=joblib.load(path)
            start=time.perf_counter(); pred=model.predict(test_sparse);seconds=time.perf_counter()-start
        else:
            ck=torch.load(path,map_location='cpu',weights_only=True)
            model=Network(kind,ck['nwords']);model.load_state_dict(ck['state']);model.to(device)
            sync(device);start=time.perf_counter();pred=logits(model,test_dense,device).argmax(1).numpy();sync(device)
            seconds=time.perf_counter()-start
        row={k:v for k,v in plan.items() if k!='path'}
        row.update(metric(ytest,pred));row.update(predict_seconds=seconds,model_bytes=path.stat().st_size)
        records.append(row)
        for item,p in zip(test,pred): predictions.append(dict(id=item['id'],label=item['label'],method=kind,seed=plan['seed'],prediction=int(p)))
        reports[f'{kind}_{plan["seed"]}']=dict(classification_report=classification_report(ytest,pred,target_names=LABELS,output_dict=True),confusion_matrix=confusion_matrix(ytest,pred).tolist())
        print(f'TEST {kind} seed={plan["seed"]} acc={row["accuracy"]:.4f} macro F1={row["macro_f1"]:.4f}',flush=True)
    write_csv(OUT/'runs.csv',records);write_csv(OUT/'history.csv',history);write_csv(OUT/'predictions.csv',predictions)
    (OUT/'class_reports.json').write_text(json.dumps(reports,indent=2)+'\n')
    summary=[]
    for kind in list(factories)+['MeanPool','TextCNN']:
        runs=[r for r in records if r['method']==kind]
        summary.append(dict(method=kind,runs=len(runs),accuracy_mean=float(np.mean([r['accuracy'] for r in runs])),
            f1_mean=float(np.mean([r['macro_f1'] for r in runs])),
            f1_sd=float(np.std([r['macro_f1'] for r in runs],ddof=1)) if len(runs)>1 else None,
            mean_fit_seconds=float(np.mean([r['fit_seconds'] for r in runs])),
            mean_predict_seconds=float(np.mean([r['predict_seconds'] for r in runs])),
            representative_seed=max(runs,key=lambda r:r['val_macro_f1'])['seed']))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'settings.json').write_text(json.dumps(dict(device=device,epochs=epochs,seeds=SEEDS,tfidf_feature_seconds=feature_seconds,
        neural_feature_seconds=neural_feature_seconds,context=MAXLEN,vocab=len(vocab),split_seed=20261118,
        majority_baseline_accuracy=float(np.mean(ytest==Counter(yt).most_common(1)[0][0]))),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');p.add_argument('--epochs',type=int,default=10)
    a=p.parse_args()
    if a.epochs<1:p.error('epochs must be positive')
    with threadpool_limits(limits=4): run(a.device,a.epochs)
