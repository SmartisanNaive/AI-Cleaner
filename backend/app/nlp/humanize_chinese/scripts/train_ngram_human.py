#!/usr/bin/env python3
"""B-2: Train a character-level unigram/bigram/trigram frequency table on the
HC3 human-answer corpus. Output format matches `ngram_freq_cn.json` so it can
be consumed by `ngram_model._trigram_log_prob` via a freq_table parameter.

Methodology note (important): we split the HC3 human corpus 80/20 into train/test
with a fixed seed, and train on the 80% split ONLY. This keeps our HC3 benchmark
honest — the 20% held-out human texts + all HC3 ChatGPT texts provide a clean
test surface for calibrating the Binoculars ratio indicator in B-3.

Output: scripts/ngram_freq_cn_human.json (primary) and, next to it, a companion
scripts/ngram_freq_cn_human_holdout.json describing the withheld indices so
benchmark scripts can filter them out.
"""
import json
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(os.path.dirname(ROOT), '..', 'data', 'hc3_human_corpus.txt')
OUT_FREQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ngram_freq_cn_human.json')
OUT_HOLDOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ngram_freq_cn_human_holdout.json')

HOLDOUT_FRAC = 0.20
SEED = 42


def is_chinese(c):
    return '\u4e00' <= c <= '\u9fff'


def main():
    if not os.path.exists(CORPUS):
        print(f'error: corpus not found at {CORPUS}')
        print('run data/extract_hc3_human.py first (B-1 step)')
        sys.exit(1)

    with open(CORPUS, encoding='utf-8') as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]

    n_total = len(lines)
    rng = random.Random(SEED)
    indices = list(range(n_total))
    rng.shuffle(indices)
    n_test = int(n_total * HOLDOUT_FRAC)
    test_idx = set(indices[:n_test])
    train_lines = [lines[i] for i in range(n_total) if i not in test_idx]

    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    total_chars = 0

    for line in train_lines:
        chars = [c for c in line if is_chinese(c)]
        total_chars += len(chars)
        for c in chars:
            unigrams[c] += 1
        for i in range(len(chars) - 1):
            bigrams[chars[i] + chars[i + 1]] += 1
        for i in range(len(chars) - 2):
            trigrams[chars[i] + chars[i + 1] + chars[i + 2]] += 1

    print(f'Corpus lines: {n_total:,} ({n_total - n_test:,} train + {n_test:,} holdout)')
    print(f'Train Chinese chars: {total_chars:,}')
    print(f'Unique unigrams: {len(unigrams):,}')
    print(f'Unique bigrams: {len(bigrams):,}')
    print(f'Unique trigrams: {len(trigrams):,}')

    freq = {
        'unigrams': dict(unigrams),
        'bigrams': dict(bigrams),
        'trigrams': dict(trigrams),
        'meta': {
            'source': 'HC3-Chinese human_answers (80% train split)',
            'seed': SEED,
            'total_chars': total_chars,
            'unique_chars': len(unigrams),
        },
    }
    with open(OUT_FREQ, 'w', encoding='utf-8') as f:
        json.dump(freq, f, ensure_ascii=False, separators=(',', ':'))
    print(f'Wrote {OUT_FREQ} ({os.path.getsize(OUT_FREQ):,} bytes)')

    with open(OUT_HOLDOUT, 'w', encoding='utf-8') as f:
        json.dump({
            'seed': SEED,
            'holdout_frac': HOLDOUT_FRAC,
            'holdout_line_indices': sorted(test_idx),
            'corpus_path': CORPUS,
        }, f, ensure_ascii=False, separators=(',', ':'))
    print(f'Wrote {OUT_HOLDOUT} ({os.path.getsize(OUT_HOLDOUT):,} bytes)')


if __name__ == '__main__':
    main()
