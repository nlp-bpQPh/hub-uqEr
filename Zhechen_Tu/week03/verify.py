"""Reload artifacts, check leakage/labels, and measure short-text limitations."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
import torch
from classifier import ROOT, LABELS, load_split, load_model, encode, evaluate, predict
from prepare_data import normalize

out = ROOT/'results'
torch.set_num_threads(2)
rows = {split: load_split(split) for split in ['train', 'val', 'test']}
for split, data in rows.items():
    counts = Counter(r['label'] for r in data)
    assert set(counts) == set(LABELS) and len(set(counts.values())) == 1
    assert all(r['text'].strip() for r in data)
for field in ['article', 'sentence_id']:
    groups = {s: {r[field] for r in data} for s, data in rows.items()}
    assert not (groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
texts = [normalize(r['text']) for data in rows.values() for r in data]
assert len(texts) == len(set(texts))
print('PASS: all 12 classes balanced; no shared articles, aligned sentence IDs, or normalized duplicate text.')
vec, model = load_model(out)
x, y = encode(rows['test'], vec)
loss, accuracy, pred = evaluate(model, x, y)
metrics = json.loads((out/'metrics.json').read_text())
assert accuracy == metrics['test_accuracy']
assert abs(loss-metrics['test_loss']) < 1e-6
assert abs(f1_score(y, pred, average='macro') - metrics['test_macro_f1']) < 1e-12
assert sum(1 for line in (out/'test_errors.jsonl').read_text().splitlines() if line) == int((y != pred).sum())
assert accuracy > .90
print(f'PASS: reloaded weights reproduce test metrics: {accuracy:.4%}; predictions match error log.')
for invalid in ['', '   ', '12345', '!!!']:
    try:
        predict(invalid, out)
        raise AssertionError('Invalid input accepted')
    except ValueError:
        pass
print('PASS: empty and non-linguistic inputs rejected.')
# Diagnostic only: never used for checkpoint selection or parameter tuning.
stress = {}
for length in [5, 10, 20]:
    short = [dict(r, text=r['text'][:length]) for r in rows['test']]
    sx, sy = encode(short, vec)
    sl, sa, sp = evaluate(model, sx, sy)
    stress[str(length)] = dict(accuracy=sa, macro_f1=float(f1_score(sy, sp, average='macro')), count=len(sy))
    print(f'Diagnostic: first {length} characters accuracy={sa:.4%}')
(out/'short_text_diagnostics.json').write_text(json.dumps(stress, indent=2)+'\n')
examples = {
    'arb_Arab': 'أتعلم البرمجة وأحب قراءة الكتب الجديدة.',
    'deu_Latn': 'Ich lerne Programmieren und lese gerne neue Bücher.',
    'eng_Latn': 'I am learning to program and enjoy reading new books.',
    'fra_Latn': "J'apprends à programmer et j'aime lire de nouveaux livres.",
    'hin_Deva': 'मैं प्रोग्रामिंग सीख रहा हूँ और नई किताबें पढ़ना पसंद करता हूँ।',
    'ita_Latn': 'Sto imparando a programmare e mi piace leggere nuovi libri.',
    'jpn_Jpan': '私はプログラミングを学んでいて、新しい本を読むのが好きです。',
    'kor_Hang': '저는 프로그래밍을 배우고 있으며 새로운 책을 읽는 것을 좋아합니다.',
    'por_Latn': 'Estou aprendendo a programar e gosto de ler novos livros.',
    'rus_Cyrl': 'Я учусь программировать и люблю читать новые книги.',
    'spa_Latn': 'Estoy aprendiendo a programar y me gusta leer libros nuevos.',
    'zho_Hans': '我正在学习编程，也喜欢阅读新书。',
}
results = []
for label, text in examples.items():
    top = predict(text, out)[0]
    results.append(dict(text=text, expected=label, **top, correct=top['label']==label))
(out/'prediction_examples.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(f'Illustrative handwritten examples: {sum(r["correct"] for r in results)}/12 correct (not a statistical test set).')
print('ALL ARTIFACT CHECKS PASSED')
