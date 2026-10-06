"""Подключение к базе данных (SQLAlchemy + SQLite)."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATABASE_URL

# check_same_thread=False нужен, т.к. FastAPI обслуживает запросы в разных потоках
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Базовый класс для ORM-моделей."""


def get_db() -> Iterator[Session]:
    """
    FastAPI-зависимость: выдаёт сессию БД и закрывает её после запроса.

    Yields:
        Активная сессия SQLAlchemy.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Создаёт таблицы, если их ещё нет."""
    # Импорт моделей нужен, чтобы SQLAlchemy увидел их до create_all
    from webapp import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
