"""Compare actual greedy responses; no gold labels are needed."""
import argparse
import torch
from dataset import QUESTIONS,prompt
from sft import OUT,base_path,decode,restore

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=list(QUESTIONS),required=True)
    p.add_argument('--text',required=True)
    p.add_argument('--compare',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(2)
    inp=prompt(dict(instruction=QUESTIONS[a.task],input=a.text))
    models=[('SFT',OUT/'sft_model.pt')]
    if a.compare: models.insert(0,('Base',base_path()))
    for name,path in models:
        model,vocab=restore(path)
        if not a.text.strip() or set(inp)-set(vocab) or len(inp)>model.cfg.context:
            p.error('Please use nonempty English text in the base vocabulary and keep the full prompt within 128 characters.')
        answer,ended=decode(model,vocab,[inp])[0]
        print(f'{name}: {answer!r} | stop: {"newline" if ended else "length limit"}')
