#!/usr/bin/env python3
"""P1 multi-source LR training: mix HC3 (Q&A) with CUDRT (news/academic)
to fix register-overfitting where the detector mis-flags formal Chinese
news as AI.

Sources:
  - HC3 ChatGPT vs HC3 human (Q&A register, what we trained on before)
  - CUDRT Baichuan-Rewrite vs CUDRT human (news register from Sina/Guangming)

Output: same lr_coef_cn.json as train_lr_scorer.py — drop-in replacement.
"""
import argparse
import json
import math
import os
import random
import sys
from statistics import mean, stdev

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = '/Users/mac/claudeclaw/humanize'
HC3_PATH = os.path.join(WORKSPACE, 'data/hc3_chinese_all.jsonl')
CUDRT_PATH = '/tmp/cudrt_rewrite/Rewrite.json'
AI_LONGFORM_PATH = os.path.join(WORKSPACE, 'data/ai_longform_corpus.jsonl')
HUMAN_NOVEL_PATH = os.path.join(WORKSPACE, 'data/human_novel_corpus.jsonl')
HUMAN_NEWS_PATH = os.path.join(WORKSPACE, 'data/human_news_corpus.jsonl')
DEFAULT_OUT = os.path.join(SCRIPT_DIR, 'lr_coef_cn.json')

sys.path.insert(0, SCRIPT_DIR)
import ngram_model as nm


def load_hc3(path, n_per_class=300, seed=42, min_cn=100):
    rng = random.Random(seed)
    ai, hum = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            try: row = json.loads(line)
            except: continue
            for a in row.get('chatgpt_answers', []) or []:
                if a and sum(1 for c in a if '\u4e00' <= c <= '\u9fff') >= min_cn:
                    ai.append(a)
            for h in row.get('human_answers', []) or []:
                if h and sum(1 for c in h if '\u4e00' <= c <= '\u9fff') >= min_cn:
                    hum.append(h)
    rng.shuffle(ai)
    rng.shuffle(hum)
    return ai[:n_per_class], hum[:n_per_class]


def load_cudrt(path, n_per_class=300, seed=42, min_cn=200):
    rng = random.Random(seed + 1)
    with open(path) as f:
        data = json.load(f)
    rng.shuffle(data)
    ai, hum = [], []
    for d in data:
        h = d.get('Human_Content', '') or ''
        a = d.get('AI_Content', '') or ''
        if (sum(1 for c in h if '\u4e00' <= c <= '\u9fff') >= min_cn
            and sum(1 for c in a if '\u4e00' <= c <= '\u9fff') >= min_cn):
            if len(ai) < n_per_class:
                ai.append(a)
            if len(hum) < n_per_class:
                hum.append(h)
            if len(ai) >= n_per_class and len(hum) >= n_per_class:
                break
    return ai, hum


def load_ai_longform(path, n=80, seed=42, min_cn=200):
    """AI long-form (modern LLMs across 5 genres) — addresses issue #5
    undercount on novel/blog register."""
    rng = random.Random(seed + 2)
    texts = []
    if not os.path.exists(path):
        return texts
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get('text', '')
            if t and sum(1 for c in t if '\u4e00' <= c <= '\u9fff') >= min_cn:
                texts.append(t)
    rng.shuffle(texts)
    return texts[:n]


def load_human_novel(path, n=80, seed=42, min_cn=400):
    """Human-written fiction passages from v3ucn/chinese-novel-dataset
    (pre-LLM era, Chinese literary register)."""
    rng = random.Random(seed + 3)
    texts = []
    if not os.path.exists(path):
        return texts
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get('text', '')
            if t and sum(1 for c in t if '\u4e00' <= c <= '\u9fff') >= min_cn:
                texts.append(t)
    rng.shuffle(texts)
    return texts[:n]


def load_human_news(path, n=200, seed=42, min_cn=600):
    """Human-written long-form news from CNewSum (pre-LLM era, journalism)."""
    rng = random.Random(seed + 4)
    texts = []
    if not os.path.exists(path):
        return texts
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get('text', '')
            if t and sum(1 for c in t if '\u4e00' <= c <= '\u9fff') >= min_cn:
                texts.append(t)
    rng.shuffle(texts)
    return texts[:n]


def standardize(X_train, X_holdout):
    n_feat = len(X_train[0])
    means = [mean(x[f] for x in X_train) for f in range(n_feat)]
    scales = [(stdev([x[f] for x in X_train]) or 1.0) for f in range(n_feat)]
    def s(x): return [(x[f] - means[f]) / scales[f] for f in range(n_feat)]
    return [s(x) for x in X_train], [s(x) for x in X_holdout], means, scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--n-hc3', type=int, default=300)
    ap.add_argument('--n-cudrt', type=int, default=300)
    ap.add_argument('--n-ai-longform', type=int, default=80)
    ap.add_argument('--n-human-novel', type=int, default=80)
    ap.add_argument('--n-human-news', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--c', type=float, default=1.0)
    args = ap.parse_args()

    print(f'Loading HC3 (n={args.n_hc3} per class)...')
    hc3_ai, hc3_hum = load_hc3(HC3_PATH, n_per_class=args.n_hc3, seed=args.seed)
    print(f'  HC3 AI={len(hc3_ai)}, human={len(hc3_hum)}')

    print(f'Loading CUDRT (n={args.n_cudrt} per class)...')
    cudrt_ai, cudrt_hum = load_cudrt(CUDRT_PATH, n_per_class=args.n_cudrt, seed=args.seed)
    print(f'  CUDRT AI={len(cudrt_ai)}, human={len(cudrt_hum)}')

    print(f'Loading AI long-form (n={args.n_ai_longform})...')
    ai_longform = load_ai_longform(AI_LONGFORM_PATH, n=args.n_ai_longform, seed=args.seed)
    print(f'  AI long-form: {len(ai_longform)}')

    print(f'Loading human novels (n={args.n_human_novel})...')
    human_novel = load_human_novel(HUMAN_NOVEL_PATH, n=args.n_human_novel, seed=args.seed)
    print(f'  Human novel: {len(human_novel)}')

    print(f'Loading human news (n={args.n_human_news})...')
    human_news = load_human_news(HUMAN_NEWS_PATH, n=args.n_human_news, seed=args.seed)
    print(f'  Human news: {len(human_news)}')

    # Combine
    ai_all = hc3_ai + cudrt_ai + ai_longform
    hum_all = hc3_hum + cudrt_hum + human_novel + human_news
    n = min(len(ai_all), len(hum_all))
    ai_all = ai_all[:n]
    hum_all = hum_all[:n]
    print(f'Combined: {len(ai_all)} AI + {len(hum_all)} human (n={n} per class)')

    # Extract features
    print(f'Extracting features for {2*n} samples...')
    X, y = [], []
    for i, t in enumerate(ai_all):
        vec, _ = nm.extract_feature_vector(t)
        X.append(vec); y.append(1)
        if (i + 1) % 100 == 0: print(f'  AI {i+1}/{n}')
    for i, t in enumerate(hum_all):
        vec, _ = nm.extract_feature_vector(t)
        X.append(vec); y.append(0)
        if (i + 1) % 100 == 0: print(f'  Human {i+1}/{n}')

    # 80/20 split
    rng = random.Random(args.seed)
    idx_ai = list(range(n))
    idx_h = list(range(n, 2 * n))
    rng.shuffle(idx_ai); rng.shuffle(idx_h)
    split = int(0.8 * n)
    train_idx = idx_ai[:split] + idx_h[:split]
    holdout_idx = idx_ai[split:] + idx_h[split:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_holdout = [X[i] for i in holdout_idx]
    y_holdout = [y[i] for i in holdout_idx]
    print(f'Train: {len(X_train)}  Holdout: {len(X_holdout)}')

    X_train_std, X_holdout_std, means, scales = standardize(X_train, X_holdout)

    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression(C=args.c, penalty='l2', solver='lbfgs', max_iter=1000)
    lr_model.fit(X_train_std, y_train)
    w = list(lr_model.coef_[0])
    b = float(lr_model.intercept_[0])

    def predict(x):
        z = b + sum(x[i] * w[i] for i in range(len(w)))
        return 1.0 / (1.0 + math.exp(-z)) if z > -500 else 0.0
    p_holdout = [predict(x) for x in X_holdout_std]
    correct = sum(1 for p, yy in zip(p_holdout, y_holdout) if (p >= 0.5) == (yy == 1))
    acc = correct / len(p_holdout)
    print(f'\nHoldout accuracy: {acc:.3f}')

    names = list(nm.LR_FEATURE_NAMES)
    ranked = sorted(zip(names, w), key=lambda x: -abs(x[1]))
    print('\nTop features by |coef|:')
    for nm_, wi in ranked[:10]:
        sign = '↑AI' if wi > 0 else '↓AI'
        print(f'  {nm_:<30} {wi:+.3f}  {sign}')

    output = {
        'version': '4.1.0-multisource',
        'features': list(nm.LR_FEATURE_NAMES),
        'mean': means,
        'scale': scales,
        'coef': w,
        'intercept': b,
        'trained_on': f'HC3 (n={args.n_hc3}) + CUDRT Rewrite (n={args.n_cudrt})',
        'n_train': len(X_train),
        'n_holdout': len(X_holdout),
        'holdout_accuracy': acc,
        'backend': 'sklearn',
    }
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
