"""Generate grouped instruction / input / response records; no external data."""
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 20261112
FIELDS = {
    'mood': {'happy': ['joyful','cheerful','delighted'], 'sad': ['gloomy','unhappy','tearful'], 'angry': ['furious','annoyed','irate']},
    'place': {'school': ['classroom','schoolyard','lecture hall'], 'beach': ['seashore','sandy coast','ocean shore'], 'farm': ['barn','cowshed','wheat field']},
    'time': {'morning': ['at sunrise','after dawn','before breakfast'], 'afternoon': ['after lunch','before dinner','at midday'], 'night': ['at midnight','after dusk','under the stars']},
}
QUESTIONS = {'mood': 'Name the mood.', 'place': 'Name the place.', 'time': 'Name the time of day.'}

def prompt(row):
    return f"Task: {row['instruction']}\nText: {row['input']}\nAnswer: "

def build():
    rng = random.Random(SEED)
    sets = {s: [] for s in ('train','val','test')}
    for mood,place,time in itertools.product(*[list(d) for d in FIELDS.values()]):
        labels = dict(mood=mood,place=place,time=time)
        triples = list(itertools.product(*(FIELDS[k][v] for k,v in labels.items())))
        rng.shuffle(triples)
        for i,(m,p,t) in enumerate(triples):
            split = 'train' if i<21 else 'val' if i<24 else 'test'
            sentence = f'I felt {m} near the {p} {t}.'
            for task in FIELDS:
                sets[split].append(dict(task=task,instruction=QUESTIONS[task],input=sentence,response=labels[task]))
    for rows in sets.values():
        rng.shuffle(rows)
    return sets

def load(split):
    return [json.loads(s) for s in (ROOT/'data'/f'{split}.jsonl').read_text().splitlines()]

if __name__=='__main__':
    (ROOT/'data').mkdir(exist_ok=True)
    for s,rows in build().items():
        (ROOT/'data'/f'{s}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        print(s,len(rows),'instructions;',len({r['input'] for r in rows}),'sentences')
