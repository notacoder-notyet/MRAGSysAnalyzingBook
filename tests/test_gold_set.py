"""Unit-тесты gold-набора: разбор ссылок и извлечение JSON из ответа LLM."""

from __future__ import annotations

from gold_set import GoldItem, extract_json_block


def test_parse_refs() -> None:
    assert GoldItem.parse_refs("6:3|6:4") == [(6, 3), (6, 4)]
    assert GoldItem.parse_refs(" 12 : 8 ") == [(12, 8)]


def test_parse_refs_empty_values() -> None:
    assert GoldItem.parse_refs("") == []
    assert GoldItem.parse_refs("nan") == []
    assert GoldItem.parse_refs("None") == []


def test_extract_json_block_plain() -> None:
    assert extract_json_block('проза до {"a": 1} и после') == {"a": 1}


def test_extract_json_block_fenced() -> None:
    assert extract_json_block('```json\n{"b": 2}\n```') == {"b": 2}


def test_extract_json_block_none() -> None:
    assert extract_json_block("нет json") is None
    assert extract_json_block("") is None
