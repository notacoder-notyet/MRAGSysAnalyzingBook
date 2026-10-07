"""
Точка входа веб-приложения: FastAPI + SQLAlchemy + статика.

Запуск:
    python -m webapp.main
    # или
    uvicorn webapp.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import STATIC_DIR, WEBAPP_HOST, WEBAPP_PORT
from webapp.database import init_db
from webapp.routers import auth, chat, lessons


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Управляет жизненным циклом приложения: создаёт таблицы при старте.

    Args:
        app: Экземпляр FastAPI.

    Yields:
        Управление приложению после инициализации.
    """
    init_db()
    yield


app = FastAPI(
    title="MRAG — RAG по учебникам ML",
    description="Вопросы к 59 лекциям с цитированием урока и страницы",
    version="1.0.0",
    lifespan=lifespan,
)

# Разрешаем запросы с локального фронтенда при разработке
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(lessons.router)


@app.get("/api/health", tags=["system"])
def health() -> dict[str, str]:
    """Проверка живости сервиса."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Отдаёт главную страницу приложения."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/static/{path:path}", include_in_schema=False)
def static_file(path: str) -> FileResponse:
    """
    Отдаёт статику без кэширования.

    Нужно для разработки: иначе браузер держит старый app.js и правки
    не видны, что уже приводило к «пустым» формам без обработчиков.

    Args:
        path: Путь внутри папки static.

    Returns:
        Запрошенный файл с заголовками no-cache.
    """
    target = STATIC_DIR / path
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")

    return FileResponse(
        target,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


# Вся статика отдаётся через маршрут /static/{path} выше (с no-cache),
# поэтому отдельный StaticFiles-монт не нужен


def main() -> None:
    """Запуск dev-сервера uvicorn."""
    import uvicorn

    uvicorn.run("webapp.main:app", host=WEBAPP_HOST, port=WEBAPP_PORT, reload=False)


if __name__ == "__main__":
    main()
