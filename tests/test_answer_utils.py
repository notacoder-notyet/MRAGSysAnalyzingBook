"""Unit-тесты чистых утилит ответа (answer_utils) — без тяжёлых зависимостей."""

from __future__ import annotations

from answer_utils import dedupe_sources, sanitize_answer, substitute_refs

SOURCES = [
    {"lesson": 12, "page": 8, "score": 0.857},
    {"lesson": 39, "page": 14, "score": 0.852},
    {"lesson": 6, "page": 9, "score": 0.851},
    {"lesson": 1, "page": 26, "score": 0.849},
    {"lesson": 39, "page": 17, "score": 0.843},
    {"lesson": 10, "page": 9, "score": 0.842},
    {"lesson": 6, "page": 6, "score": 0.842},
]


def test_dedupe_keeps_one_page_per_lesson() -> None:
    """Один урок — одна (лучшая) страница, порядок — по убыванию score."""
    out = dedupe_sources(SOURCES)
    assert [s["lesson"] for s in out] == [12, 39, 6, 1, 10]
    # Оставлены именно самые релевантные страницы уроков
    assert {s["lesson"]: s["page"] for s in out}[39] == 14
    assert {s["lesson"]: s["page"] for s in out}[6] == 9


def test_dedupe_sorts_by_score() -> None:
    scores = [s["score"] for s in dedupe_sources(SOURCES)]
    assert scores == sorted(scores, reverse=True)


def test_dedupe_empty() -> None:
    assert dedupe_sources([]) == []


def test_sanitize_removes_latex_footnotes_and_markdown() -> None:
    raw = r"\[ G = \sqrt{G_x^2} \] **Итог:** 【5†L1-L4】 (Урок 6, Страница 3)"
    out = sanitize_answer(raw)
    assert "\\" not in out
    assert "【" not in out
    assert "**" not in out
    assert "sqrt(G_x^2)" in out
    assert "(Урок 6, Страница 3)" in out


def test_sanitize_normalizes_unicode() -> None:
    assert sanitize_answer("a\u202fb\u2011c") == "a b-c"


def test_sanitize_empty() -> None:
    assert sanitize_answer("") == ""


def test_substitute_refs_maps_to_chunk_metadata() -> None:
    chunks = [
        {"metadata": {"lesson": 6, "page": 3}},
        {"metadata": {"lesson": 9, "page": 1}},
    ]
    out = substitute_refs("Смотри [1] и [2].", chunks)
    assert out == "Смотри (Урок 6, Страница 3) и (Урок 9, Страница 1)."


def test_substitute_refs_multiple_numbers_in_bracket() -> None:
    chunks = [
        {"metadata": {"lesson": 6, "page": 3}},
        {"metadata": {"lesson": 9, "page": 1}},
    ]
    assert substitute_refs("[1, 2]", chunks) == "(Урок 6, Страница 3) (Урок 9, Страница 1)"


def test_substitute_refs_drops_out_of_range() -> None:
    chunks = [{"metadata": {"lesson": 6, "page": 3}}]
    assert substitute_refs("См [3].", chunks) == "См ."


def test_substitute_refs_without_chunks_is_noop() -> None:
    assert substitute_refs("[1]", []) == "[1]"
