#!/usr/bin/env python3
"""Train LR ensemble scorer on HC3-Chinese balanced sample.

Loads HC3 all.jsonl, extracts 18 continuous features per text via
`ngram_model.extract_feature_vector`, fits an L2 logistic regression
to distinguish AI from human answers, and saves coefficients + scaler
stats to scripts/lr_coef_cn.json.

Usage:
    python scripts/train_lr_scorer.py
    python scripts/train_lr_scorer.py --n 400 --seed 123

sklearn is preferred; pure-numpy gradient descent fallback runs when
sklearn is not installed.
"""
import argparse
import json
import math
import os
import random
import sys
from statistics import mean, stdev

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE = '/Users/mac/claudeclaw/humanize'
DEFAULT_DATA = os.path.join(WORKSPACE, 'data/hc3_chinese_all.jsonl')
DEFAULT_OUT = os.path.join(SCRIPT_DIR, 'lr_coef_cn.json')

sys.path.insert(0, SCRIPT_DIR)
import ngram_model as nm


def load_hc3_balanced(path, n=300, seed=42, min_cn_chars=100):
    rng = random.Random(seed)
    ai_texts, human_texts = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            for a in row.get('chatgpt_answers', []) or []:
                if a and sum(1 for c in a if '\u4e00' <= c <= '\u9fff') >= min_cn_chars:
                    ai_texts.append(a)
            for h in row.get('human_answers', []) or []:
                if h and sum(1 for c in h if '\u4e00' <= c <= '\u9fff') >= min_cn_chars:
                    human_texts.append(h)
    rng.shuffle(ai_texts)
    rng.shuffle(human_texts)
    return ai_texts[:n], human_texts[:n]


def standardize(X_train, X_holdout):
    """Standardize features: (X - mean) / scale. Return (X_train_std, X_holdout_std, mean, scale)."""
    n_feat = len(X_train[0])
    means = [mean(x[f] for x in X_train) for f in range(n_feat)]
    scales = []
    for f in range(n_feat):
        s = stdev([x[f] for x in X_train]) or 1.0
        scales.append(s)

    def scale_vec(x):
        return [(x[f] - means[f]) / scales[f] for f in range(n_feat)]

    return [scale_vec(x) for x in X_train], [scale_vec(x) for x in X_holdout], means, scales


def fit_lr_sklearn(X, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=1000)
    lr.fit(X, y)
    return list(lr.coef_[0]), float(lr.intercept_[0])


def fit_lr_numpy(X, y, C=1.0, lr=0.05, max_iter=2000):
    """Pure-numpy L2 LR via batch gradient descent.

    loss = -mean(y log p + (1-y) log(1-p)) + 0.5/(n*C) * ||w||^2
    Adam would be faster, but plain GD converges fine on 500-sample 18-feature setup.
    """
    import numpy as np
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for it in range(max_iter):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = X.T @ (p - y) / n + (w / (n * C))
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return list(w), float(b)


def predict_proba(X, w, b):
    result = []
    for x in X:
        z = sum(x[i] * w[i] for i in range(len(w))) + b
        p = 1.0 / (1.0 + math.exp(-z)) if z > -500 else 0.0
        result.append(p)
    return result


def eval_binary(probs, y_true, threshold=0.5):
    correct = sum(1 for p, y in zip(probs, y_true) if (p >= threshold) == (y == 1))
    return correct / len(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--n', type=int, default=300, help='samples per class')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--c', type=float, default=1.0, help='L2 inverse regularization')
    args = ap.parse_args()

    print(f'Loading HC3 from {args.data}...')
    ai_texts, human_texts = load_hc3_balanced(args.data, n=args.n, seed=args.seed)
    print(f'Got {len(ai_texts)} AI + {len(human_texts)} human')

    # Extract features
    print(f'Extracting features for {2 * args.n} samples...')
    X = []
    y = []
    for i, t in enumerate(ai_texts):
        vec, _ = nm.extract_feature_vector(t)
        X.append(vec)
        y.append(1)  # 1 = AI
        if (i + 1) % 50 == 0:
            print(f'  AI {i+1}/{len(ai_texts)}')
    for i, t in enumerate(human_texts):
        vec, _ = nm.extract_feature_vector(t)
        X.append(vec)
        y.append(0)  # 0 = human
        if (i + 1) % 50 == 0:
            print(f'  Human {i+1}/{len(human_texts)}')

    # 80/20 split, stratified by shuffling each class separately
    rng = random.Random(args.seed)
    idx_ai = list(range(args.n))
    idx_h = list(range(args.n, 2 * args.n))
    rng.shuffle(idx_ai)
    rng.shuffle(idx_h)
    split_ai = int(0.8 * args.n)
    split_h = int(0.8 * args.n)
    train_idx = idx_ai[:split_ai] + idx_h[:split_h]
    holdout_idx = idx_ai[split_ai:] + idx_h[split_h:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_holdout = [X[i] for i in holdout_idx]
    y_holdout = [y[i] for i in holdout_idx]

    print(f'Train: {len(X_train)}  Holdout: {len(X_holdout)}')

    # Standardize
    X_train_std, X_holdout_std, means, scales = standardize(X_train, X_holdout)

    # Fit
    try:
        w, b = fit_lr_sklearn(X_train_std, y_train, C=args.c)
        backend = 'sklearn'
    except ImportError:
        print('sklearn not available, using pure-numpy fallback')
        w, b = fit_lr_numpy(X_train_std, y_train, C=args.c)
        backend = 'numpy-gd'

    # Eval on holdout
    p_holdout = predict_proba(X_holdout_std, w, b)
    acc = eval_binary(p_holdout, y_holdout)

    # Show top features by |weight|
    names = list(nm.LR_FEATURE_NAMES)
    ranked = sorted(zip(names, w), key=lambda x: -abs(x[1]))
    print(f'\nHoldout accuracy: {acc:.3f}  (backend: {backend})')
    print(f'\nTop features by |coef| (standardized):')
    for name, wi in ranked:
        sign = '↑AI' if wi > 0 else '↓AI'
        print(f'  {name:<30} {wi:+.3f}  {sign}')

    # AI-vs-human probability distribution on holdout
    ai_probs = [p for p, yi in zip(p_holdout, y_holdout) if yi == 1]
    h_probs = [p for p, yi in zip(p_holdout, y_holdout) if yi == 0]
    print(f'\nHoldout probs:')
    print(f'  AI    mean={mean(ai_probs):.3f} median={sorted(ai_probs)[len(ai_probs)//2]:.3f} n={len(ai_probs)}')
    print(f'  Human mean={mean(h_probs):.3f} median={sorted(h_probs)[len(h_probs)//2]:.3f} n={len(h_probs)}')

    # Save
    output = {
        'version': '3.4.0-dev',
        'features': list(nm.LR_FEATURE_NAMES),
        'mean': means,
        'scale': scales,
        'coef': w,
        'intercept': b,
        'trained_on': os.path.basename(args.data),
        'n_train': len(X_train),
        'n_holdout': len(X_holdout),
        'holdout_accuracy': acc,
        'backend': backend,
    }
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
