#!/usr/bin/env python3
"""Train a Chinese news corpus ngram (THUCNews-derived).
Format matches ngram_freq_cn_human.json / ngram_freq_cn_wiki.json.

Used for news-register Binoculars-style divergence features.
"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(os.path.dirname(ROOT), '..', 'data', 'thucnews_corpus.txt')
OUT_FREQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ngram_freq_cn_news.json')


def is_chinese(c):
    return '\u4e00' <= c <= '\u9fff'


def main():
    if not os.path.exists(CORPUS):
        print(f'error: corpus not found at {CORPUS}')
        sys.exit(1)

    with open(CORPUS, encoding='utf-8') as f:
        text = f.read()

    chars = [c for c in text if is_chinese(c)]
    total = len(chars)

    unigrams = Counter(chars)
    bigrams = Counter(chars[i] + chars[i + 1] for i in range(total - 1))
    trigrams = Counter(chars[i] + chars[i + 1] + chars[i + 2] for i in range(total - 2))

    print(f'Chinese chars: {total:,}')
    print(f'Unigrams: {len(unigrams):,}; bigrams: {len(bigrams):,}; trigrams: {len(trigrams):,}')

    freq = {
        'unigrams': dict(unigrams),
        'bigrams': dict(bigrams),
        'trigrams': dict(trigrams),
        'meta': {
            'source': 'THUCNews subset (news register)',
            'total_chars': total,
        },
    }
    with open(OUT_FREQ, 'w', encoding='utf-8') as f:
        json.dump(freq, f, ensure_ascii=False, separators=(',', ':'))
    print(f'Wrote {OUT_FREQ} ({os.path.getsize(OUT_FREQ):,} bytes)')


if __name__ == '__main__':
    main()
