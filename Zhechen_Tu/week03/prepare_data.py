"""Build a reproducible 12-language subset; group articles before splitting."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import unicodedata
import urllib.request
import numpy as np

LANGUAGES = {'arb_Arab': '阿拉伯语', 'deu_Latn': '德语', 'eng_Latn': '英语',
             'fra_Latn': '法语', 'hin_Deva': '印地语', 'ita_Latn': '意大利语',
             'jpn_Jpan': '日语', 'kor_Hang': '韩语', 'por_Latn': '葡萄牙语',
             'rus_Cyrl': '俄语', 'spa_Latn': '西班牙语', 'zho_Hans': '简体中文'}
SEED = 20261029
URL = 'https://tinyurl.com/flores200dataset'
ROOT = Path(__file__).resolve().parent

def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=ROOT / 'flores200.tar.gz')
    args = parser.parse_args()
    if not args.archive.exists():
        urllib.request.urlretrieve(URL, args.archive)
    rows = []
    with tarfile.open(args.archive) as archive:
        def read(name):
            return archive.extractfile('./flores200_dataset/' + name).read().decode('utf-8')
        for split in ['dev', 'devtest']:
            metadata = list(csv.DictReader(io.StringIO(read(f'metadata_{split}.tsv')), delimiter='\t'))
            for lang in LANGUAGES:
                lines = read(f'{split}/{lang}.{split}').splitlines()
                assert len(lines) == len(metadata)
                for i, (text, meta) in enumerate(zip(lines, metadata)):
                    rows.append(dict(id=f'{split}_{i+1}_{lang}', sentence_id=f'{split}_{i+1}',
                                     article=meta['URL'], label=lang, text=text))
    # Remove every sentence group containing duplicate/empty normalized text.
    # Dropping the whole aligned group preserves class balance and removes ambiguity.
    seen, bad = {}, set()
    for row in rows:
        key = normalize(row['text'])
        if not key:
            bad.add(row['sentence_id'])
        if key in seen:
            bad.update([row['sentence_id'], seen[key]])
        seen[key] = row['sentence_id']
    kept = [r for r in rows if r['sentence_id'] not in bad]
    articles = sorted({r['article'] for r in kept})
    np.random.default_rng(SEED).shuffle(articles)
    a, b = int(len(articles)*.70), int(len(articles)*.85)
    mapping = {url: split for split, urls in [('train', articles[:a]), ('val', articles[a:b]), ('test', articles[b:])] for url in urls}
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    counts = {}
    for split in ['train', 'val', 'test']:
        selected = [r for r in kept if mapping[r['article']] == split]
        (data / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected), encoding='utf-8')
        counts[split] = {'rows': len(selected), 'per_language': len(selected)//12, 'articles': len({r['article'] for r in selected})}
    manifest = dict(source=URL, source_documentation='https://github.com/facebookresearch/flores/tree/main/flores200',
                    license='CC-BY-SA-4.0', archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                    seed=SEED, languages=LANGUAGES, original_rows=len(rows), removed_sentence_groups=len(bad),
                    split_method='70/15/15 percent of shuffled unique article URLs; all translations stay together', counts=counts)
    (data / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
