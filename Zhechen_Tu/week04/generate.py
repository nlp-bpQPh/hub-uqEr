"""Autoregressively sample new characters from the saved Transformer."""
import argparse
import json
from pathlib import Path
import torch
from train import ROOT, load_checkpoint

@torch.no_grad()
def generate(model, vocabulary, prompt, length=400, temperature=.8, top_k=20, seed=20261104):
    if not prompt:
        raise ValueError('prompt cannot be empty')
    if not temperature > 0 or length < 0 or top_k < 1:
        raise ValueError('temperature/top_k must be positive and length nonnegative')
    stoi = {ch:i for i,ch in enumerate(vocabulary)}
    unseen = sorted(set(prompt)-set(stoi))
    if unseen:
        raise ValueError(f'Prompt contains characters outside the training vocabulary: {unseen}')
    device = next(model.parameters()).device
    rng = torch.Generator(device='cpu').manual_seed(seed)
    ids = torch.tensor([[stoi[ch] for ch in prompt]], device=device)
    model.eval()
    for _ in range(length):
        logits = model(ids[:, -model.cfg.context:])[:, -1, :].cpu()/temperature
        logits[:,0] = -float('inf')  # Never emit the reserved unknown token.
        k = min(top_k,len(vocabulary)-1)
        threshold = torch.topk(logits,k).values[:,-1:]
        logits[logits<threshold] = -float('inf')
        nxt = torch.multinomial(logits.softmax(-1),1,generator=rng).to(device)
        ids = torch.cat((ids,nxt),dim=1)
    return ''.join(vocabulary[i] for i in ids[0].tolist())

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=ROOT/'results/best_model.pt')
    p.add_argument('--prompt',default='ROMEO:\n')
    p.add_argument('--length',type=int,default=400)
    p.add_argument('--temperature',type=float,default=.8)
    p.add_argument('--top-k',type=int,default=20)
    p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--examples',action='store_true',help='Save three fixed, unedited demonstrations next to checkpoint')
    args=p.parse_args()
    torch.set_num_threads(4)
    model,ckpt=load_checkpoint(args.checkpoint)
    try:
        if args.examples:
            samples=[]
            for prompt in ['ROMEO:\n','KING RICHARD II:\n','First Citizen:\n']:
                result=generate(model,ckpt['vocabulary'],prompt,args.length,args.temperature,args.top_k,args.seed)
                samples.append(dict(prompt=prompt,seed=args.seed,temperature=args.temperature,top_k=args.top_k,new_characters=args.length,text=result))
            args.checkpoint.with_name('generation_examples.json').write_text(json.dumps(samples,indent=2)+'\n')
            args.checkpoint.with_name('generation_examples.txt').write_text('\n\n'.join(f'PROMPT: {s["prompt"]!r}\n{s["text"]}' for s in samples)+'\n')
            print(args.checkpoint.with_name('generation_examples.txt').read_text())
        else:
            print(generate(model,ckpt['vocabulary'],args.prompt,args.length,args.temperature,args.top_k,args.seed))
    except ValueError as exc:
        p.error(str(exc))
