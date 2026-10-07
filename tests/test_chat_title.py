"""
Интеграционный тест: имя нового чата = первый вопрос.

Модуль тяжёлый (импортирует приложение → torch), поэтому в лёгком CI
пропускается через importorskip. Локально проверяет поведение эндпоинта.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")  # приложение тянет sentence-transformers

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from webapp import rag_service  # noqa: E402
from webapp.database import Base, get_db  # noqa: E402
from webapp.main import app  # noqa: E402
from webapp.models import User  # noqa: E402
from webapp.security import create_access_token  # noqa: E402

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base.metadata.create_all(engine)


def _override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client(monkeypatch):
    """TestClient с изолированной БД и заглушкой RAG (без реального LLM)."""
    app.dependency_overrides[get_db] = _override_get_db

    # Чистим состояние между тестами (пересоздаём схему)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    db = TestingSession()
    user = User(username="tester", password_hash="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.id, user.username)
    db.close()

    monkeypatch.setattr(
        rag_service,
        "ask",
        lambda question, top_k=None, filter_lesson=None: {"answer": "Ответ", "sources": []},
    )

    c = TestClient(app)
    c.headers.update({"Authorization": f"Bearer {token}"})
    yield c
    app.dependency_overrides.clear()


def test_new_chat_named_by_first_question(client: TestClient) -> None:
    created = client.post("/api/chat", json={"title": "Новый чат"}).json()
    chat_id = created["id"]
    assert created["title"] == "Новый чат"

    resp = client.post(
        "/api/chat/ask", json={"question": "Что такое градиент?", "chat_id": chat_id}
    )
    assert resp.status_code == 200

    detail = client.get(f"/api/chat/{chat_id}").json()
    assert detail["title"] == "Что такое градиент?"


def test_title_not_changed_on_second_question(client: TestClient) -> None:
    created = client.post("/api/chat", json={"title": "Новый чат"}).json()
    chat_id = created["id"]

    client.post("/api/chat/ask", json={"question": "Первый вопрос", "chat_id": chat_id})
    client.post("/api/chat/ask", json={"question": "Второй вопрос", "chat_id": chat_id})

    detail = client.get(f"/api/chat/{chat_id}").json()
    assert detail["title"] == "Первый вопрос"


def test_ask_without_chat_id_creates_named_chat(client: TestClient) -> None:
    resp = client.post("/api/chat/ask", json={"question": "Что такое переобучение?"})
    assert resp.status_code == 200
    chat_id = resp.json()["chat_id"]

    detail = client.get(f"/api/chat/{chat_id}").json()
    assert detail["title"] == "Что такое переобучение?"
