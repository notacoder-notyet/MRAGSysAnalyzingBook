"""Unit-тесты Pydantic-схем: очистка ответа и дедуп источников при отдаче."""

from __future__ import annotations

from webapp.schemas import MessageOut, SourceOut

CREATED_AT = "2026-01-01T00:00:00"


def test_assistant_content_is_sanitized() -> None:
    msg = MessageOut(
        id=1,
        role="assistant",
        content=r"\[ x \] итог 【1†L1】",
        created_at=CREATED_AT,
    )
    content = msg.model_dump()["content"]
    assert "\\" not in content
    assert "【" not in content


def test_user_content_is_untouched() -> None:
    original = r"Что такое \[x\]?"
    msg = MessageOut(id=2, role="user", content=original, created_at=CREATED_AT)
    assert msg.model_dump()["content"] == original


def test_sources_are_deduped_and_sorted() -> None:
    msg = MessageOut(
        id=3,
        role="assistant",
        content="ok",
        created_at=CREATED_AT,
        sources=[
            SourceOut(lesson=6, page=9, score=0.85),
            SourceOut(lesson=6, page=6, score=0.84),
            SourceOut(lesson=9, page=1, score=0.90),
        ],
    )
    lessons = [s["lesson"] for s in msg.model_dump()["sources"]]
    assert lessons == [9, 6]  # по score desc, один урок — одна ссылка
