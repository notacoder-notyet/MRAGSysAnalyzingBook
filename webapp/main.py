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
from fastapi.responses import FileResponse, HTMLResponse

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
def index() -> HTMLResponse:
    """
    Отдаёт главную страницу с cache-busting версией статики.

    К ссылкам на style.css и app.js добавляется `?v=<mtime>`. Если файл
    изменился, версия меняется, и браузер обязан запросить свежую копию.
    Это решает проблему «браузер держит старый app.js», из-за которой
    правки не видны и формы уходят в нативный submit.

    Returns:
        HTML страницы с подставленной версией статики и запретом кэша.
    """
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    # Версия = самый свежий mtime среди файлов фронтенда
    version = int(
        max((STATIC_DIR / name).stat().st_mtime for name in ("index.html", "style.css", "app.js"))
    )

    html = html.replace("/static/style.css", f"/static/style.css?v={version}")
    html = html.replace("/static/app.js", f"/static/app.js?v={version}")

    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


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
