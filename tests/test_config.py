"""Unit-тесты конфигурации: инварианты значений и промпта."""

from __future__ import annotations

import config


def test_prompt_has_context_placeholder() -> None:
    assert "{context}" in config.RAG_SYSTEM_PROMPT


def test_prompt_requests_numbered_refs() -> None:
    """Промпт должен просить ссылаться по номеру [N], а не выдумывать страницы."""
    assert "[1]" in config.RAG_SYSTEM_PROMPT


def test_retrieval_defaults_sane() -> None:
    assert config.DEFAULT_TOP_K >= 1
    assert 0.0 < config.MIN_ANSWER_SCORE <= 1.0


def test_chunking_defaults_sane() -> None:
    assert config.DEFAULT_CHUNK_SIZE > 0
    assert 0 <= config.DEFAULT_CHUNK_OVERLAP < config.DEFAULT_CHUNK_SIZE
