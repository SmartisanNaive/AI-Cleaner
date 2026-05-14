#!/usr/bin/env python3
"""Shared text helpers for paragraph-preserving Chinese rewrites."""

import re


_PARAGRAPH_SPLIT_RE = re.compile(r'\n[ \t\r]*\n+')


def split_paragraphs(text):
    """Split text into paragraphs using blank lines, including CRLF blanks."""
    return _PARAGRAPH_SPLIT_RE.split(text)


def join_paragraphs(parts):
    """Join paragraph parts with the repository's canonical blank line."""
    return '\n\n'.join(parts)
