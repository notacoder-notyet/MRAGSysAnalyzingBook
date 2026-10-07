"""Pydantic-схемы (валидация запросов и сериализация ответов)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from config import (
    CHAT_TITLE_MAX_LENGTH,
    PASSWORD_MIN_LENGTH,
    USERNAME_MAX_LENGTH,
    USERNAME_MIN_LENGTH,
)

# ============================================================
# AUTH
# ============================================================


class RegisterRequest(BaseModel):
    """Регистрация нового пользователя."""

    username: str = Field(min_length=USERNAME_MIN_LENGTH, max_length=USERNAME_MAX_LENGTH)
    password: str = Field(min_length=PASSWORD_MIN_LENGTH)

    @field_validator("username")
    @classmethod
    def username_allowed_chars(cls, value: str) -> str:
        """Разрешаем только буквы, цифры, дефис и подчёркивание."""
        cleaned = value.strip()
        if not cleaned.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Имя может содержать только буквы, цифры, - и _")
        return cleaned


class LoginRequest(BaseModel):
    """Вход существующего пользователя."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """Ответ с токеном доступа."""

    access_token: str
    token_type: str = "bearer"
    username: str


class UserOut(BaseModel):
    """Публичные данные пользователя."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    created_at: datetime


# ============================================================
# CHAT
# ============================================================


class ChatCreate(BaseModel):
    """Создание нового чата."""

    title: str = Field(default="Новый чат", max_length=CHAT_TITLE_MAX_LENGTH)


class ChatRename(BaseModel):
    """Переименование чата."""

    title: str = Field(min_length=1, max_length=CHAT_TITLE_MAX_LENGTH)


class SourceOut(BaseModel):
    """Источник ответа: урок и страница."""

    model_config = ConfigDict(from_attributes=True)

    lesson: int
    page: int
    score: float


class MessageOut(BaseModel):
    """Сообщение чата с источниками."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    created_at: datetime
    sources: list[SourceOut] = []

    @field_serializer("content")
    def _clean_assistant_content(self, value: str) -> str:
        """
        Чистит текст ответа ассистента при отдаче (LaTeX, сноски 【...】).

        Нужно, чтобы и ранее сохранённые ответы показывались «вменяемо»:
        санитайзер применяется при генерации, но старые сообщения уже лежат
        в базе в исходном виде. Вопросы пользователя не трогаем.
        """
        if self.role != "assistant":
            return value
        from webapp.rag_service import _sanitize_answer

        return _sanitize_answer(value)

    @field_serializer("sources")
    def _dedupe_sources_field(self, value: list[SourceOut]) -> list[SourceOut]:
        """
        Убирает дубликаты источников при отдаче: по одной странице на урок.

        Так и старые ответы (источники которых сохранены как есть) показывают
        тот же компактный список, что и свежие.
        """
        if not value:
            return value
        from webapp.rag_service import _dedupe_sources

        rows = [{"lesson": s.lesson, "page": s.page, "score": s.score} for s in value]
        return [SourceOut(**row) for row in _dedupe_sources(rows)]


class ChatOut(BaseModel):
    """Чат без сообщений (для списка истории)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ChatDetail(BaseModel):
    """Чат с полной историей сообщений."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut] = []


class AskRequest(BaseModel):
    """Вопрос пользователя к RAG-системе."""

    question: str = Field(min_length=2, max_length=1000)
    chat_id: int | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    filter_lesson: int | None = Field(default=None, ge=1)


class AskResponse(BaseModel):
    """Ответ RAG-системы."""

    chat_id: int
    message_id: int
    question: str
    answer: str
    sources: list[SourceOut] = []


# ============================================================
# LESSONS / PDF
# ============================================================


class LessonOut(BaseModel):
    """Информация об уроке для навигации по PDF."""

    lesson: int
    pdf_name: str
    pages: int
