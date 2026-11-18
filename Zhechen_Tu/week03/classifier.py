"""Character TF-IDF + learned PyTorch 12-class softmax classifier."""
import argparse
import csv
import json
import random
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
from torch import nn
from prepare_data import LANGUAGES, ROOT, SEED, normalize

LABELS = list(LANGUAGES)

def load_split(split):
    return [json.loads(line) for line in (ROOT / 'data' / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]

def encode(rows, vectorizer):
    return vectorizer.transform([normalize(r['text']) for r in rows]), np.array([LABELS.index(r['label']) for r in rows])

@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits = torch.cat([model(torch.from_numpy(x[i:i+256].toarray())) for i in range(0, len(y), 256)])
    pred = logits.argmax(1).numpy()
    return float(nn.functional.cross_entropy(logits, torch.tensor(y))), float(accuracy_score(y, pred)), pred

def load_model(out):
    vectorizer = joblib.load(out / 'vectorizer.joblib')  # Load only this trusted, locally trained artifact.
    checkpoint = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
    assert checkpoint['labels'] == LABELS
    model = nn.Linear(checkpoint['features'], 12)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return vectorizer, model

def predict(text, out):
    clean = normalize(text)
    if not clean or not any(c.isalpha() for c in clean):
        raise ValueError('请输入包含文字的非空文本。')
    vectorizer, model = load_model(out)
    x = vectorizer.transform([clean])
    if x.nnz == 0:
        raise ValueError('输入未包含模型词表中的字符特征。')
    with torch.no_grad():
        scores = model(torch.from_numpy(x.toarray())).softmax(1)[0].numpy()
    order = np.argsort(-scores)[:3]
    return [{'label': LABELS[i], 'language': LANGUAGES[LABELS[i]], 'softmax_score': float(scores[i])} for i in order]

def train(out, epochs):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    def emit(message):
        print(message, flush=True)
        log.append(message.rstrip())
        (out / 'training_log.txt').write_text('\n'.join(log)+'\n', encoding='utf-8')
    train_rows, val_rows = load_split('train'), load_split('val')
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=30000,
                                 min_df=2, sublinear_tf=True, lowercase=False, dtype=np.float32)
    vectorizer.fit([normalize(r['text']) for r in train_rows])
    x, y = encode(train_rows, vectorizer)
    vx, vy = encode(val_rows, vectorizer)
    model = nn.Linear(x.shape[1], 12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.0001)
    rng = np.random.default_rng(SEED)
    best, history = float('inf'), []
    initial_loss, initial_accuracy, _ = evaluate(model, vx, vy)
    emit('Week 03 | Zhechen Tu | 12-language text classification')
    emit(f'PyTorch {torch.__version__} | CPU | seed={SEED}')
    emit(f'Train={len(y)} | Validation={len(vy)} | Features={x.shape[1]} | Classes=12')
    emit(f'Initial validation: loss={initial_loss:.5f}, accuracy={initial_accuracy:.4%}')
    for epoch in range(1, epochs+1):
        model.train()
        order = rng.permutation(len(y))
        total_loss, correct = 0., 0
        for start in range(0, len(y), 256):
            idx = order[start:start+256]
            bx, by = torch.from_numpy(x[idx].toarray()), torch.tensor(y[idx])
            optimizer.zero_grad()
            logits = model(bx)
            loss = nn.functional.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(idx)
            correct += int((logits.argmax(1) == by).sum())
        vl, va, _ = evaluate(model, vx, vy)
        history.append(dict(epoch=epoch, train_loss=total_loss/len(y), train_accuracy=correct/len(y), val_loss=vl, val_accuracy=va))
        if vl < best:
            best, best_epoch = vl, epoch
            torch.save(dict(state_dict=model.state_dict(), features=x.shape[1], labels=LABELS, seed=SEED, epoch=epoch), out/'model.pt')
        scheduler.step()
        emit(f'Epoch {epoch:02d}: train_loss={total_loss/len(y):.5f} train_acc={correct/len(y):.4%} val_loss={vl:.5f} val_acc={va:.4%}')
    joblib.dump(vectorizer, out/'vectorizer.joblib', compress=3)
    vectorizer, model = load_model(out)
    # Test set is first transformed/evaluated only after checkpoint selection.
    test_rows = load_split('test')
    tx, ty = encode(test_rows, vectorizer)
    loss, acc, pred = evaluate(model, tx, ty)
    report = classification_report(ty, pred, labels=list(range(12)), target_names=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(ty, pred, labels=list(range(12)))
    metrics = dict(seed=SEED, best_epoch=best_epoch, epochs=epochs, features=x.shape[1],
                   initial_val_accuracy=initial_accuracy, test_loss=loss, test_accuracy=acc,
                   test_macro_f1=float(f1_score(ty, pred, average='macro')), test_count=len(ty),
                   correct=int((ty==pred).sum()), uniform_baseline_accuracy=1/12,
                   classification_report=report, confusion_matrix=cm.tolist())
    (out/'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (out/'training_history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(history)
    errors = [dict(**row, predicted=LABELS[int(p)]) for row, p in zip(test_rows, pred) if row['label'] != LABELS[int(p)]]
    (out/'test_errors.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in errors), encoding='utf-8')
    emit(f'Best validation checkpoint: epoch {best_epoch}')
    emit(f'TEST accuracy={acc:.4%} ({metrics["correct"]}/{len(ty)}) | macro_F1={metrics["test_macro_f1"]:.6f}')
    emit(f'Test errors={len(errors)} | balanced chance baseline={1/12:.4%}')
    emit(classification_report(ty, pred, target_names=LABELS, digits=4, zero_division=0))
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for key, title, ax in [('loss','Cross entropy',axes[0]), ('accuracy','Accuracy',axes[1])]:
        ax.plot([h['epoch'] for h in history], [h['train_'+key] for h in history], label='Train (online)')
        ax.plot([h['epoch'] for h in history], [h['val_'+key] for h in history], label='Validation')
        ax.set(xlabel='Epoch', ylabel=title)
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle('12-language classifier | Zhechen Tu')
    fig.tight_layout()
    fig.savefig(out/'training_curves.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,9))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax, shrink=.8)
    ax.set(xticks=range(12), yticks=range(12), xticklabels=LABELS, yticklabels=LABELS,
           xlabel='Predicted language', ylabel='True language', title=f'Held-out articles | Accuracy {acc:.2%}')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(12):
        for j in range(12):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout()
    fig.savefig(out/'confusion_matrix.png',dpi=160)
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results')
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    if args.predict is not None:
        try:
            print(json.dumps(predict(args.predict, args.output_dir), ensure_ascii=False, indent=2))
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if args.epochs < 1:
            parser.error('--epochs must be positive')
        train(args.output_dir, args.epochs)
