"""Train on disjoint text offsets, select by validation NLL, test once."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from model import Config, LanguageModel

ROOT = Path(__file__).resolve().parent
SEED = 20261104
SOURCE = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

def get_data():
    raw = (ROOT/'data/input.txt').read_bytes()
    text = raw.decode('utf-8')
    a, b = int(len(text)*.9), int(len(text)*.95)
    chars = ['<UNK>'] + sorted(set(text[:a]))  # Training text only.
    vocab = {ch: i for i, ch in enumerate(chars)}
    segments = {'train': text[:a], 'val': text[a:b], 'test': text[b:]}
    arrays = {s: torch.tensor([vocab.get(ch, 0) for ch in t], dtype=torch.long) for s, t in segments.items()}
    manifest = dict(source=SOURCE, sha256=hashlib.sha256(raw).hexdigest(), total_characters=len(text),
        split_offsets={'train': [0,a], 'val': [a,b], 'test': [b,len(text)]},
        split_characters={s:len(t) for s,t in segments.items()},
        unknown_counts={s:int((v==0).sum()) for s,v in arrays.items()}, vocabulary=chars)
    return arrays, chars, manifest

@torch.no_grad()
def evaluate(model, data, device, batch=32):
    """Score each target index 1..N-1 exactly once; reset context every 128 chars."""
    model.eval()
    ctx = model.cfg.context
    starts = list(range(0, len(data)-1, ctx))
    total, correct, count = 0., 0, 0
    # Full windows batch together; score final shorter window separately.
    full = [s for s in starts if s+ctx < len(data)]
    groups = [full[i:i+batch] for i in range(0,len(full),batch)]
    groups += [[s] for s in starts if s+ctx >= len(data)]
    for group in groups:
        x = torch.stack([data[s:min(s+ctx,len(data)-1)] for s in group]).to(device)
        y = torch.stack([data[s+1:min(s+ctx+1,len(data))] for s in group]).to(device)
        logits = model(x)
        total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum').item())
        correct += int((logits.argmax(-1) == y).sum().item())
        count += y.numel()
    nll = total/count
    return dict(nll=nll, perplexity=math.exp(nll), accuracy=correct/count, tokens=count)

def baselines(train, test, vocab_size):
    counts = torch.bincount(train, minlength=vocab_size).double()+1
    uni = counts/counts.sum()
    pairs = torch.bincount(train[:-1]*vocab_size+train[1:], minlength=vocab_size**2).reshape(vocab_size,vocab_size).double()+1
    probs = pairs/pairs.sum(1,keepdim=True)
    losses = {'uniform': math.log(vocab_size), 'unigram': float(-uni[test[1:]].log().mean()),
              'bigram': float(-probs[test[:-1],test[1:]].log().mean())}
    return {key:dict(nll=value, perplexity=math.exp(value)) for key,value in losses.items()}

def load_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = LanguageModel(Config(**ckpt['config']))
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval(), ckpt

def train(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = args.device
    data, chars, manifest = get_data()
    cfg = Config(len(chars))
    (out/'data_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    model = LanguageModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9,.95), weight_decay=.01)
    rng = torch.Generator().manual_seed(SEED)
    history, log = [], []
    started = time.monotonic()
    def emit(message):
        print(message, flush=True)
        log.append(message)
        (out/'training_log.txt').write_text('\n'.join(log)+'\n')
    parameters = sum(p.numel() for p in model.parameters())
    emit(f'Week 04 | Zhechen Tu | Causal Transformer language model | seed={SEED}')
    emit(f'PyTorch={torch.__version__} device={device} parameters={parameters} config={asdict(cfg)}')
    emit(f'Data={manifest["split_characters"]}; vocabulary fitted on train; UNK={manifest["unknown_counts"]}')
    initial = evaluate(model, data['val'], device)
    emit(f'Initial validation NLL={initial["nll"]:.5f} perplexity={initial["perplexity"]:.4f}')
    best, best_step, running = float('inf'), 0, []
    for step in range(1, args.steps+1):
        model.train()
        # 100-step warmup, then cosine decay to 1e-4.
        lr = 1e-3*step/100 if step <= 100 else 1e-4+.5*9e-4*(1+math.cos(math.pi*(step-100)/max(1,args.steps-100)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        starts = torch.randint(len(data['train'])-cfg.context, (24,), generator=rng)
        x = torch.stack([data['train'][s:s+cfg.context] for s in starts]).to(device)
        y = torch.stack([data['train'][s+1:s+cfg.context+1] for s in starts]).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1,cfg.vocab_size), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        running.append(loss.item())
        if step % 100 == 0 or step == args.steps:
            val = evaluate(model, data['val'], device)
            row = dict(step=step, train_nll=float(np.mean(running)), val_nll=val['nll'],
                       val_perplexity=val['perplexity'], learning_rate=lr, seconds=time.monotonic()-started)
            history.append(row)
            running.clear()
            emit(f'Step {step:04d}: train_nll={row["train_nll"]:.5f} val_nll={val["nll"]:.5f} val_ppl={val["perplexity"]:.4f} elapsed={row["seconds"]:.1f}s')
            if val['nll'] < best:
                best, best_step = val['nll'], step
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    config=asdict(cfg), vocabulary=chars, step=step, seed=SEED, data_sha256=manifest['sha256']), out/'best_model.pt')
            with (out/'training_history.csv').open('w',newline='') as f:
                writer = csv.DictWriter(f,fieldnames=list(row),lineterminator='\n')
                writer.writeheader()
                writer.writerows(history)
    model, ckpt = load_checkpoint(out/'best_model.pt', device)
    test = evaluate(model, data['test'], device)
    baseline = baselines(data['train'], data['test'], len(chars))
    metrics = dict(seed=SEED, device=device, torch_version=str(torch.__version__), parameters=parameters,
                   config=asdict(cfg), steps=args.steps, batch_size=24, tokens_seen=args.steps*24*cfg.context,
                   best_step=best_step, initial_validation=initial, best_validation=evaluate(model,data['val'],device),
                   test=test, baselines=baseline, elapsed_seconds=time.monotonic()-started)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    emit(f'BEST step={best_step} | TEST NLL={test["nll"]:.5f} perplexity={test["perplexity"]:.4f} next_char_accuracy={test["accuracy"]:.4%} scored_tokens={test["tokens"]}')
    emit('BASELINES '+json.dumps(baseline))
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    steps = [r['step'] for r in history]
    axes[0].plot(steps,[r['train_nll'] for r in history],label='Train (online, dropout on)')
    axes[0].plot(steps,[r['val_nll'] for r in history],label='Full validation')
    axes[0].set(xlabel='Optimizer step',ylabel='Cross entropy (nats / character)')
    axes[0].legend()
    axes[1].plot(steps,[r['val_perplexity'] for r in history],label='Validation perplexity')
    axes[1].set(xlabel='Optimizer step',ylabel='Character-level perplexity')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Causal Transformer | Zhechen Tu | Week 04')
    fig.tight_layout()
    fig.savefig(out/'training_curves.png',dpi=160)
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=1500)
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    train(args)
