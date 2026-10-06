"""Роутер уроков: список презентаций и отдача PDF-файла для просмотра."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from config import PAGES_CSV_PATH, RAW_PDF_DIR
from pars_pdf import extract_lesson_number
from webapp.schemas import LessonOut

router = APIRouter(prefix="/api/lessons", tags=["lessons"])


@lru_cache(maxsize=1)
def _build_registry() -> dict[int, dict[str, object]]:
    """
    Строит реестр уроков: номер → {pdf_path, pages}.

    Кешируется, т.к. состав папки и число страниц в рамках работы
    приложения не меняются. Число страниц берём из pdf_pages.csv
    (уже посчитан при парсинге), иначе — из самого PDF.

    Returns:
        Словарь {номер урока: {"pdf_path": Path, "pages": int}}.
    """
    page_counts: dict[int, int] = {}
    if PAGES_CSV_PATH.exists():
        pages_df = pd.read_csv(PAGES_CSV_PATH)
        if {"lesson", "page"}.issubset(pages_df.columns):
            counts = pages_df.groupby("lesson")["page"].max()
            page_counts = {int(k): int(v) for k, v in counts.items()}

    registry: dict[int, dict[str, object]] = {}
    folder = Path(RAW_PDF_DIR)
    if not folder.is_dir():
        return registry

    for pdf_path in sorted(folder.glob("*.pdf")):
        lesson = extract_lesson_number(pdf_path.name)
        if lesson is None:
            continue
        pages = page_counts.get(lesson)
        if pages is None:
            pages = _count_pdf_pages(pdf_path)
        registry[lesson] = {"pdf_path": pdf_path, "pages": pages}
    return registry


def _count_pdf_pages(pdf_path: Path) -> int:
    """Считает страницы PDF напрямую (резерв, если нет pdf_pages.csv)."""
    import pdfplumber

    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:  # noqa: BLE001 — битый PDF не должен ронять список уроков
        return 0


@router.get("", response_model=list[LessonOut])
def list_lessons() -> list[LessonOut]:
    """
    Возвращает список доступных уроков с числом страниц.

    Returns:
        Отсортированный по номеру урока список.
    """
    registry = _build_registry()
    return [
        LessonOut(lesson=num, pdf_name=Path(str(data["pdf_path"])).name, pages=int(data["pages"]))
        for num, data in sorted(registry.items())
    ]


@router.get("/{lesson}/pdf")
def get_lesson_pdf(lesson: int) -> FileResponse:
    """
    Отдаёт PDF-файл урока для просмотра в браузере (PDF.js).

    Args:
        lesson: Номер урока.

    Returns:
        Файл презентации с заголовком inline — чтобы открывался во вьювере,
        а не скачивался.

    Raises:
        HTTPException: 404, если урок не найден.
    """
    registry = _build_registry()
    entry = registry.get(lesson)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Урок не найден")

    pdf_path = Path(str(entry["pdf_path"]))
    if not pdf_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Файл презентации недоступен"
        )

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
        headers={"Content-Disposition": f'inline; filename="{pdf_path.name}"'},
    )
