"""Unit-тесты парсинга: номер урока из имени файла и символьный чанкинг."""

from __future__ import annotations

import pytest

from pars_pdf import chunk_text, extract_lesson_number


def test_extract_lesson_number() -> None:
    assert extract_lesson_number("lesson_6.pdf") == 6
    assert extract_lesson_number("Lesson-12.pdf") == 12
    assert extract_lesson_number("lesson 3.pdf") == 3
    assert extract_lesson_number("no_number.pdf") is None


def test_chunk_text_basic() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=4, chunk_overlap=1)
    assert chunks[0] == "abcd"
    assert chunks[1] == "defg"  # шаг = chunk_size - overlap = 3


def test_chunk_text_empty() -> None:
    assert chunk_text("   ") == []


def test_chunk_text_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=4, chunk_overlap=4)


def test_chunk_text_invalid_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0, chunk_overlap=0)
