from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "ai_cleaner.sqlite3"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_ANTHROPIC_MODEL = "claude-4-6-sonnet"
OPENAI_CHAT_COMPLETIONS_PATH = "/chat/completions"
ANTHROPIC_MESSAGES_PATH = "/v1/messages"

SUPPORTED_PLATFORMS = ("weipu", "paperyy", "paperpass", "zhuque")
SUPPORTED_PROVIDERS = ("openai", "anthropic")
SUPPORTED_NLP_STYLES = ("academic", "general", "long_blog")
