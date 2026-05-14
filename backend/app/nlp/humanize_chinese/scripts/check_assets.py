#!/usr/bin/env python3
"""Report local data asset status for humanize-chinese.

This command never downloads data. Optional ngram tables are local-only because
they are large and depend on corpora that users may need to prepare themselves.
"""
from __future__ import annotations

import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)


CORE_ASSETS = (
    ('scripts/ngram_freq_cn.json', 50_000, 'primary char 3-gram; detect/rewrite baseline perplexity'),
    ('scripts/patterns_cn.json', 10_000, 'rule patterns and rewrite replacements'),
    ('scripts/lr_coef_cn.json', 500, 'general LR scorer'),
    ('scripts/lr_coef_academic.json', 500, 'academic LR scorer'),
    ('scripts/lr_coef_longform.json', 500, 'longform LR scorer'),
    ('scripts/ngram_freq_cn_human_holdout.json', 1_000, 'HC3 human holdout split metadata'),
)

OPTIONAL_ASSETS = (
    (
        'scripts/ngram_freq_cn_human.json',
        5_000_000,
        'enables bino_lp_diff and the binoculars component of best-of-n secondary signal',
        'python scripts/train_ngram_human.py',
    ),
    (
        'scripts/ngram_freq_cn_wiki.json',
        2_000_000,
        'enables wiki_vs_human/wiki_vs_primary LR features; also required by news divergence',
        'python scripts/train_ngram_wiki.py',
    ),
    (
        'scripts/ngram_freq_cn_news.json',
        2_000_000,
        'enables news_vs_human LR feature',
        'python scripts/train_ngram_news.py',
    ),
)


def _asset_status(rel_path: str, min_bytes: int) -> tuple[str, int]:
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        return 'MISSING', 0
    size = os.path.getsize(path)
    if size < min_bytes:
        return 'SMALL', size
    return 'OK', size


def _is_git_ignored(rel_path: str) -> bool:
    try:
        proc = subprocess.run(
            ['git', 'check-ignore', '-q', rel_path],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return proc.returncode == 0


def _fmt_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f'{size / (1024 * 1024):.1f}MB'
    if size >= 1024:
        return f'{size / 1024:.1f}KB'
    return f'{size}B'


def _print_group(title: str, rows):
    print(title)
    for rel_path, min_bytes, desc, *rest in rows:
        status, size = _asset_status(rel_path, min_bytes)
        ignored = ' gitignored' if _is_git_ignored(rel_path) else ''
        print(f'  {status:7s} {_fmt_size(size):>8s} {rel_path}{ignored}')
        print(f'           {desc}')
        if status != 'OK' and rest:
            print(f'           rebuild: {rest[0]}')


def main() -> int:
    print('humanize-chinese asset doctor')
    print()
    _print_group('Core assets (checked in; required for normal operation):', CORE_ASSETS)
    print()
    _print_group('Optional local ngram assets (not checked in):', OPTIONAL_ASSETS)
    print()

    missing_optional = [
        rel_path for rel_path, min_bytes, _, _ in OPTIONAL_ASSETS
        if _asset_status(rel_path, min_bytes)[0] != 'OK'
    ]
    if missing_optional:
        print('Impact:')
        print('  Fresh clone still runs offline, but LR scores and best-of-n ranking may differ from')
        print('  full-asset benchmark numbers. Missing optional ngram features are treated as 0.0.')
        print('  No data is downloaded automatically; prepare local corpora and run the rebuild commands above.')
    else:
        print('Impact:')
        print('  Optional ngram features are available; full-asset LR and secondary signals can run locally.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
