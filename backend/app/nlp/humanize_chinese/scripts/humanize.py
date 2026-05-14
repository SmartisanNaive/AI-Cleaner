#!/usr/bin/env python3
"""Unified CLI entrypoint for humanize-chinese.

Usage:
  humanize detect   <file> [options]    AI detection score (0-100)
  humanize rewrite  <file> [options]    Humanize (去 AI 味改写)
  humanize academic <file> [options]    Academic paper AIGC 降重
  humanize style    <file> --style S    8 种写作风格转换
  humanize compare  <file> [options]    改写前后对比
  humanize doctor                       Check local data asset status

  humanize --list                       List available subcommands
  humanize <sub> --help                 Per-subcommand help (forwards to underlying script)

Under the hood each subcommand calls the corresponding scripts/*_cn.py via
subprocess, forwarding all remaining args. Exit code is propagated.

This is a thin dispatcher — the individual scripts remain the canonical
implementations and can still be invoked directly.
"""
from __future__ import annotations

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SUBCOMMANDS = {
    'detect':   ('detect_cn.py',   'AI 痕迹检测 (0-100)'),
    'rewrite':  ('humanize_cn.py', '通用去 AI 味改写'),
    'academic': ('academic_cn.py', '学术论文 AIGC 降重（11 维度）'),
    'style':    ('style_cn.py',    '8 种风格转换（含小说/小红书/知乎/微博等）'),
    'compare':  ('compare_cn.py',  '改写前后对比'),
    'doctor':   ('check_assets.py', '本地数据资产状态检查'),
}

ALIASES = {
    'humanize': 'rewrite',
    'rewrite_cn': 'rewrite',
    'acad':     'academic',
    'paper':    'academic',
    'detct':    'detect',
    'cmp':      'compare',
}

USAGE = """humanize — Chinese AI-text humanization toolkit

Usage:
  humanize <subcommand> [args]

Subcommands:
  detect     AI 痕迹检测 (0-100)
  rewrite    通用去 AI 味改写
  academic   学术论文 AIGC 降重（11 维度）
  style      8 种风格转换（含小说/小红书/知乎/微博等）
  compare    改写前后对比
  doctor     本地数据资产状态检查

Examples:
  humanize detect 论文.txt
  humanize rewrite text.txt -o clean.txt --quick
  humanize academic 论文.txt -o 改后.txt --compare
  humanize style text.txt --style xiaohongshu -o xhs.txt
  humanize compare text.txt -a
  humanize doctor

Per-subcommand help:
  humanize detect --help
  humanize academic --help
"""


def print_usage(stream=sys.stdout):
    stream.write(USAGE)


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])

    if not argv or argv[0] in ('-h', '--help', 'help'):
        print_usage()
        return 0

    if argv[0] in ('--list', 'list'):
        for name, (_, desc) in SUBCOMMANDS.items():
            print(f'  {name:9s} {desc}')
        return 0

    sub = argv[0]
    sub = ALIASES.get(sub, sub)

    if sub not in SUBCOMMANDS:
        sys.stderr.write(f'error: unknown subcommand "{argv[0]}"\n\n')
        print_usage(sys.stderr)
        return 2

    script_name, _ = SUBCOMMANDS[sub]
    target = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(target):
        sys.stderr.write(f'error: missing backing script {target}\n')
        return 3

    cmd = [sys.executable, target, *argv[1:]]
    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
