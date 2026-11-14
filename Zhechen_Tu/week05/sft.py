"""Full-parameter response-only SFT of the exact week 04 checkpoint."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import urllib.request

import torch
from torch.nn import functional as F
from dataset import ROOT, SEED, load, prompt
from model import Config, LanguageModel

OUT = ROOT/'results'
BASE = ROOT.parent/'week04/results/best_model.pt'
SHA = '8410fd04d261b0e40cb7291a98f4302a460f1a4c9172b4d6a269ceed143f015d'
URL = 'https://raw.githubusercontent.com/nlp-bpQPh/hub-uqEr/6af9b2a20fc846e9a499acffa21dbf51fe68339f/Zhechen_Tu/week04/results/best_model.pt'

def base_path():
    p = BASE if BASE.exists() else ROOT/'base_model.pt'
    if not p.exists():
        with urllib.request.urlopen(URL,timeout=60) as resp:
            raw=resp.read()
        if hashlib.sha256(raw).hexdigest()!=SHA:
            raise ValueError('Unexpected download hash')
        p.write_bytes(raw)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=SHA:
        raise ValueError('Wrong week 04 checkpoint')
    return p

def restore(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    model = LanguageModel(Config(**ck['config']))
    model.load_state_dict(ck['model'], strict=True)
    return model.to(device).eval(), ck['vocabulary']

def examples(rows,vocab,ctx):
    ids={ch:i for i,ch in enumerate(vocab)}
    result=[]
    for r in rows:
        prefix,ans=prompt(r),r['response']+'\n'
        seq=[ids[c] for c in prefix+ans]
        if len(seq)-1>ctx:
            raise ValueError('Example exceeds context')
        # Logits at index len(prefix)-1 predict the first response character.
        y=[-100]*(len(prefix)-1)+[ids[c] for c in ans]
        result.append((torch.tensor(seq[:-1]),torch.tensor(y)))
    return result

def batch(ex,device):
    x,y=zip(*ex)
    pad=torch.nn.utils.rnn.pad_sequence
    return pad(x,batch_first=True,padding_value=0).to(device),pad(y,batch_first=True,padding_value=-100).to(device)

@torch.inference_mode()
def nll(model,ex,device):
    model.eval()
    loss,n=0.,0
    for i in range(0,len(ex),64):
        x,y=batch(ex[i:i+64],device)
        loss+=F.cross_entropy(model(x).flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum').item()
        n+=int((y!=-100).sum().item())
    return loss/n

@torch.inference_mode()
def decode(model,vocab,prompts,limit=14):
    # Greedy generation reads prompts alone; no label list, response, or classifier.
    model.eval()
    ids={c:i for i,c in enumerate(vocab)}
    seq=[[ids[c] for c in s] for s in prompts]
    if not all(0<len(s)<=model.cfg.context for s in seq):
        raise ValueError('Empty or overlength prompt')
    outputs=['']*len(seq)
    done=[False]*len(seq)
    device=next(model.parameters()).device
    for _ in range(limit):
        inputs=[torch.tensor(s[-model.cfg.context:]) for s in seq]
        x=torch.nn.utils.rnn.pad_sequence(inputs,batch_first=True,padding_value=0).to(device)
        scores=model(x)[torch.arange(len(seq),device=device),torch.tensor([len(s)-1 for s in inputs],device=device)]
        preds=scores.argmax(1).tolist()
        for i,v in enumerate(preds):
            if done[i]: continue
            if vocab[v]=='\n':
                done[i]=True
            else:
                outputs[i]+=vocab[v]
                seq[i].append(v)
        if all(done): break
    return list(zip(outputs,done))

def score(model,vocab,rows,device):
    records=[]
    for start in range(0,len(rows),48):
        group=rows[start:start+48]
        for r,(response,ended) in zip(group,decode(model,vocab,[prompt(r) for r in group])):
            records.append(dict(**r,prediction=response,stopped=ended,exact=ended and response==r['response']))
    by={k:sum(r['exact'] for r in records if r['task']==k) for k in ('mood','place','time')}
    return dict(correct=sum(by.values()),count=len(records),by_task_correct=by,
                exact_match=sum(by.values())/len(records),
                response_nll=nll(model,examples(rows,vocab,model.cfg.context),device)),records

def train(device,epochs,out):
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rng=random.Random(SEED)
    model,vocab=restore(base_path(),device)
    train_rows,val_rows=load('train'),load('val')
    tr,va=[examples(rows,vocab,model.cfg.context) for rows in (train_rows,val_rows)]
    opt=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.00005)
    history=[]
    initial=nll(model,va,device)
    print(f'Week 05 | Zhechen Tu | response-only SFT | device={device} | seed={SEED}',flush=True)
    print(f'Week 04 SHA256: {SHA}\nTrain: {len(tr)} | Validation: {len(va)} | Base validation response NLL: {initial:.5f}',flush=True)
    best=float('inf')
    for epoch in range(1,epochs+1):
        model.train()
        rng.shuffle(tr)
        total,count=0.,0
        for start in range(0,len(tr),48):
            x,y=batch(tr[start:start+48],device)
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x).flatten(0,1),y.flatten(),ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            n=int((y!=-100).sum().item())
            total+=loss.item()*n
            count+=n
        v=nll(model,va,device)
        history.append(dict(epoch=epoch,train_nll=total/count,val_nll=v,lr=opt.param_groups[0]['lr']))
        sched.step()
        if v<best:
            best,best_epoch=v,epoch
            torch.save(dict(config=asdict(model.cfg),model={k:t.detach().cpu() for k,t in model.state_dict().items()},
                       vocabulary=vocab,base_sha256=SHA,epoch=epoch),out/'sft_model.pt')
        print(f'Epoch {epoch:02d} | train response NLL {total/count:.5f} | val {v:.5f}',flush=True)
    with (out/'history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]),lineterminator='\n'); w.writeheader(); w.writerows(history)
    # Open held-out test after checkpoint selection. Same prompts and decoding for both.
    test_rows=load('test')
    results=dict(base_sha256=SHA,device=device,seed=SEED,epochs=epochs,best_epoch=best_epoch,
                 best_val_nll=best,initial_val_nll=initial,train_count=len(tr),val_count=len(va))
    for name,path in [('base',base_path()),('sft',out/'sft_model.pt')]:
        m,vocab=restore(path,device)
        metric,records=score(m,vocab,test_rows,device)
        results[name]=metric
        (out/f'{name}_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        print(f'{name.upper()} TEST | exact: {metric["correct"]}/{len(records)} ({metric["exact_match"]:.2%}) | response NLL {metric["response_nll"]:.5f}',flush=True)
    (out/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--device',choices=['mps','cpu','cuda'],default='cpu')
    p.add_argument('--epochs',type=int,default=36)
    p.add_argument('--output-dir',type=Path,default=OUT)
    a=p.parse_args()
    if a.epochs<1: p.error('epochs must be positive')
    train(a.device,a.epochs,a.output_dir)
