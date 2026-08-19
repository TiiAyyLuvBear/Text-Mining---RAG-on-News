from __future__ import annotations

import re


def tokenize_vietnamese(text: str) -> list[str]:
    """Tokenize Vietnamese text, using underthesea when it is installed."""
    value = text.strip().lower()
    if not value:
        return []
    try:
        from underthesea import word_tokenize

        return [token for token in word_tokenize(value, format="text").split() if token]
    except ImportError:
        return re.findall(r"[\wÀ-ỹ]+", value, flags=re.UNICODE)


def tokenizer_name() -> str:
    try:
        import underthesea  # noqa: F401

        return "underthesea.word_tokenize"
    except ImportError:
        return "unicode_regex_fallback"
