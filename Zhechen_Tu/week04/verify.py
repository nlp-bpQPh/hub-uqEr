"""Check actual causal masking, split accounting, reload metrics, and sampling."""
import json
import math
import torch
from train import ROOT, get_data, load_checkpoint, evaluate
from generate import generate

def main():
    torch.set_num_threads(4)
    out=ROOT/'results'
    data,chars,manifest=get_data()
    saved=json.loads((out/'data_manifest.json').read_text())
    assert manifest==saved
    spans=list(manifest['split_offsets'].values())
    assert spans[0][0]==0 and spans[0][1]==spans[1][0] and spans[1][1]==spans[2][0]
    assert spans[2][1]==manifest['total_characters']
    assert all(int((values==0).sum())==0 for values in data.values())
    model,ckpt=load_checkpoint(out/'best_model.pt')
    assert ckpt['data_sha256']==manifest['sha256'] and ckpt['vocabulary']==chars
    assert set(chars[1:])==set((ROOT/'data/input.txt').read_text()[:spans[0][1]])
    ctx=model.cfg.context
    x=data['train'][100:100+ctx].unsqueeze(0)
    y=data['train'][101:101+ctx].unsqueeze(0)
    assert torch.equal(x[:,1:],y[:,:-1])
    print('PASS: source hash, train-only vocabulary, contiguous disjoint offsets, shifted targets.')
    changed=x.clone()
    changed[:,64:]=(changed[:,64:]+1)%len(chars)
    with torch.no_grad():
        logits=model(x)
        perturbed=model(changed)
    assert logits.shape==(1,ctx,len(chars)) and torch.isfinite(logits).all()
    assert torch.allclose(logits[:,:64],perturbed[:,:64],atol=1e-6,rtol=0)
    assert not torch.allclose(logits[:,64:],perturbed[:,64:])
    print('PASS: changing future input leaves earlier logits unchanged (causal mask).')
    metrics=json.loads((out/'metrics.json').read_text())
    test=evaluate(model,data['test'],'cpu')
    assert test['tokens']==len(data['test'])-1==metrics['test']['tokens']
    assert abs(test['nll']-metrics['test']['nll'])<1e-4  # CPU/MPS float32 tolerance.
    assert abs(test['perplexity']-math.exp(test['nll']))<1e-12
    assert test['nll']<metrics['baselines']['bigram']['nll']
    assert metrics['best_validation']['nll']<metrics['initial_validation']['nll']
    print(f'PASS: CPU-reloaded test NLL={test["nll"]:.6f}, PPL={test["perplexity"]:.6f}; every test target counted once; beats bigram baseline.')
    a=generate(model,chars,'ROMEO:\n',length=64)
    b=generate(model,chars,'ROMEO:\n',length=64)
    assert a==b and len(a)==len('ROMEO:\n')+64 and a.startswith('ROMEO:\n')
    for prompt in ['', '你好']:
        try:
            generate(model,chars,prompt,length=1)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid prompt accepted')
    print('PASS: seeded generation reproducible; correct length; empty/unknown prompt rejected.')
    (out/'cpu_reload_metrics.json').write_text(json.dumps(test,indent=2)+'\n')
    print('ALL CHECKS PASSED')

if __name__=='__main__':
    main()
