"""Post-training sensitivity check on the same 81 held-out cue triples."""
import json
import re
import torch
from dataset import load
from sft import OUT,restore,score

if __name__=='__main__':
    torch.set_num_threads(4)
    model,vocab=restore(OUT/'sft_model.pt')
    sets={'reordered_sentence':[], 'reworded_instruction':[]}
    questions={'mood':'How did I feel?', 'place':'Where was I?', 'time':'When was it?'}
    for row in load('test'):
        m=re.fullmatch(r'I felt (.+?) near the (.+?) (at .+|after .+|before .+|under .+)\.',row['input'])
        mood,place,time=m.groups()
        sets['reordered_sentence'].append(dict(row,input=f'{time.capitalize()} I was near the {place} and felt {mood}.'))
        sets['reworded_instruction'].append(dict(row,instruction=questions[row['task']]))
    results={k:score(model,vocab,rows,'cpu')[0] for k,rows in sets.items()}
    (OUT/'sensitivity.json').write_text(json.dumps(results,indent=2)+'\n')
    for k,m in results.items(): print(k, m['correct'], '/', m['count'])
