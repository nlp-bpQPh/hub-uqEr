"""Classify text with a saved method; neural seed chosen on validation only."""
import argparse
import json
import joblib
import torch
from data import normalize,LABELS
from experiment import OUT,Network,encode,logits

METHODS=['NaiveBayes','LogisticRegression','LinearSVM','MeanPool','TextCNN']

def restore(method,seed=None):
    if method in METHODS[:3]:
        return joblib.load(OUT/f'models/{method}.joblib'),joblib.load(OUT/'models/tfidf.joblib')
    if seed is None:
        summaries=json.loads((OUT/'summary.json').read_text())
        seed=next(s['representative_seed'] for s in summaries if s['method']==method)
    ck=torch.load(OUT/f'models/{method}_{seed}.pt',map_location='cpu',weights_only=True)
    model=Network(method,ck['nwords']);model.load_state_dict(ck['state']);model.eval()
    return model,json.loads((OUT/'models/vocab.json').read_text())

def predict_many(method,model,features,texts):
    if method in METHODS[:3]:
        return model.predict(features.transform([normalize(t) for t in texts]))
    x=encode([dict(text=t) for t in texts],features)
    return logits(model,x,'cpu').argmax(1).numpy()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,default='LinearSVM');p.add_argument('--text',required=True)
    a=p.parse_args()
    if not a.text.strip():p.error('Text must be nonempty')
    torch.set_num_threads(4)
    model,features=restore(a.method)
    pred=int(predict_many(a.method,model,features,[a.text])[0])
    print(json.dumps(dict(method=a.method,label=pred,category=LABELS[pred]),ensure_ascii=False))
