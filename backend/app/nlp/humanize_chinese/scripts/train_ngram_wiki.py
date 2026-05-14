#!/usr/bin/env python3
"""Train a character-level unigram/bigram/trigram frequency table on the
Wikipedia academic corpus. Output format matches `ngram_freq_cn_human.json`
so it can be consumed by `ngram_model._trigram_log_prob` via a freq_table.

Used for Wikipedia-vs-Human Binoculars-style feature — AI text is closer to
Wikipedia register (encyclopedic, formal) than casual human Q&A.
"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(os.path.dirname(ROOT), '..', 'data', 'wiki_academic_corpus.txt')
OUT_FREQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ngram_freq_cn_wiki.json')


def is_chinese(c):
    return '\u4e00' <= c <= '\u9fff'


def main():
    if not os.path.exists(CORPUS):
        print(f'error: corpus not found at {CORPUS}')
        sys.exit(1)

    with open(CORPUS, encoding='utf-8') as f:
        text = f.read()

    chars = [c for c in text if is_chinese(c)]
    total_chars = len(chars)

    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    for c in chars:
        unigrams[c] += 1
    for i in range(len(chars) - 1):
        bigrams[chars[i] + chars[i + 1]] += 1
    for i in range(len(chars) - 2):
        trigrams[chars[i] + chars[i + 1] + chars[i + 2]] += 1

    print(f'Total Chinese chars: {total_chars:,}')
    print(f'Unique unigrams: {len(unigrams):,}')
    print(f'Unique bigrams: {len(bigrams):,}')
    print(f'Unique trigrams: {len(trigrams):,}')

    freq = {
        'unigrams': dict(unigrams),
        'bigrams': dict(bigrams),
        'trigrams': dict(trigrams),
        'meta': {
            'source': 'Chinese Wikipedia academic corpus',
            'total_chars': total_chars,
            'unique_chars': len(unigrams),
        },
    }
    with open(OUT_FREQ, 'w', encoding='utf-8') as f:
        json.dump(freq, f, ensure_ascii=False, separators=(',', ':'))
    print(f'Wrote {OUT_FREQ} ({os.path.getsize(OUT_FREQ):,} bytes)')


if __name__ == '__main__':
    main()
