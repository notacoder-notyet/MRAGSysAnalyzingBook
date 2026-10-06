"""
Модуль парсинга PDF и подготовки данных для RAG.

Содержит:
- постраничное извлечение текста с метаданными (имя файла, урок, страница);
- извлечение таблиц;
- лёгкую EDA по датасету страниц;
- базовый чанкинг с сохранением метаданных для citation.

Эмбеддинги чанков — в соседнем модуле `embeddings.py` (см. baseline в README.md).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pdfplumber

from config import (
    CHUNKS_CSV_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    PAGES_CSV_PATH,
    RAW_PDF_DIR,
)


def extract_lesson_number(pdf_name: str) -> int | None:
    """
    Достаёт номер урока из имени PDF (например, lesson_6.pdf -> 6).

    Args:
        pdf_name: Имя файла PDF.

    Returns:
        Номер урока или None, если распознать не удалось.
    """
    # Ищем шаблон lesson_<число> в имени файла
    match = re.search(r"lesson[_\s-]?(\d+)", pdf_name, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def extract_text_from_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """
    Извлекает текст из PDF постранично вместе с названием файла и номером страницы.

    Args:
        pdf_path: Путь к PDF файлу.

    Returns:
        Список словарей вида
        {"pdf_name": str, "lesson": int | None, "page": int, "text": str}
        для каждой страницы (нумерация с 1).
    """
    pdf_name = os.path.basename(pdf_path)
    lesson = extract_lesson_number(pdf_name)
    pages: list[dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        # Идём по страницам и сохраняем текст с citation-метаданными
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            pages.append(
                {
                    "pdf_name": pdf_name,
                    "lesson": lesson,
                    "page": page_number,
                    "text": page_text,
                }
            )

    return pages


def extract_tables_from_pdf(pdf_path: str) -> list[pd.DataFrame]:
    """
    Извлекает таблицы из PDF и преобразует их в DataFrame.

    Args:
        pdf_path: Путь к PDF файлу.

    Returns:
        Список таблиц в виде pandas DataFrame.
    """
    tables: list[list[list[Any]]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # extract_tables может вернуть None — учитываем это
            page_tables = page.extract_tables() or []
            tables.extend(page_tables)

    # Первая строка таблицы используется как заголовок колонок
    return [
        pd.DataFrame(table[1:], columns=table[0])
        for table in tables
        if table and len(table) > 1
    ]


def save_to_csv(data: pd.DataFrame, output_path: str) -> None:
    """
    Сохраняет DataFrame в CSV без индекса.

    Args:
        data: Данные для сохранения.
        output_path: Путь к выходному CSV.
    """
    data.to_csv(output_path, index=False)


def build_pages_dataframe(pages: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Собирает DataFrame страниц и добавляет простые метрики длины текста.

    Args:
        pages: Результат extract_text_from_pdf.

    Returns:
        DataFrame со столбцами pdf_name, lesson, page, text, char_count, word_count.
    """
    pages_df = pd.DataFrame(pages)

    # Метрики для EDA: символы и слова (по пробелам)
    pages_df["char_count"] = pages_df["text"].fillna("").str.len()
    pages_df["word_count"] = (
        pages_df["text"].fillna("").str.split().str.len().fillna(0).astype(int)
    )
    return pages_df


def summarize_pages_dataset(pages_df: pd.DataFrame) -> dict[str, Any]:
    """
    Считает краткую статистику по датасету страниц (лёгкая EDA).

    Args:
        pages_df: DataFrame со страницами и метриками длины.

    Returns:
        Словарь со сводными метриками датасета.
    """
    # Гарантируем наличие метрик, даже если DataFrame пришёл «сырым»
    if "char_count" not in pages_df.columns or "word_count" not in pages_df.columns:
        pages_df = build_pages_dataframe(pages_df.to_dict(orient="records"))

    summary: dict[str, Any] = {
        "num_pages": int(len(pages_df)),
        "num_pdfs": int(pages_df["pdf_name"].nunique()) if "pdf_name" in pages_df else 0,
        "empty_pages": int((pages_df["char_count"] == 0).sum()),
        "total_chars": int(pages_df["char_count"].sum()),
        "total_words": int(pages_df["word_count"].sum()),
        "avg_chars_per_page": float(pages_df["char_count"].mean()) if len(pages_df) else 0.0,
        "avg_words_per_page": float(pages_df["word_count"].mean()) if len(pages_df) else 0.0,
        "median_chars_per_page": float(pages_df["char_count"].median()) if len(pages_df) else 0.0,
        "min_chars": int(pages_df["char_count"].min()) if len(pages_df) else 0,
        "max_chars": int(pages_df["char_count"].max()) if len(pages_df) else 0,
    }
    return summary


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Делит текст на чанки по символам с перекрытием.

    Args:
        text: Исходный текст страницы.
        chunk_size: Максимальная длина чанка в символах.
        chunk_overlap: Число символов перекрытия между соседними чанками.

    Returns:
        Список текстовых чанков.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap должен быть в диапазоне [0, chunk_size)")

    # Шаг окна = размер чанка минус overlap, чтобы соседние куски пересекались
    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0

    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start += step

    return chunks


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Чанкит постраничный датасет, сохраняя метаданные для citation.

    Args:
        pages: Список страниц из extract_text_from_pdf.
        chunk_size: Максимальная длина чанка в символах.
        chunk_overlap: Перекрытие соседних чанков в символах.

    Returns:
        Список чанков вида
        {chunk_id, pdf_name, lesson, page, chunk_index, text}.
    """
    result: list[dict[str, Any]] = []
    global_id = 0

    for page in pages:
        page_chunks = chunk_text(
            text=str(page.get("text", "")),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Каждый чанк наследует pdf_name / lesson / page исходной страницы
        for chunk_index, chunk in enumerate(page_chunks):
            result.append(
                {
                    "chunk_id": global_id,
                    "pdf_name": page.get("pdf_name"),
                    "lesson": page.get("lesson"),
                    "page": page.get("page"),
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )
            global_id += 1

    return result


def find_pdfs_in_dir(directory: str | Path = RAW_PDF_DIR) -> list[Path]:
    """
    Находит все PDF-файлы в директории (без рекурсии).

    Args:
        directory: Папка с PDF (по умолчанию data/raw).

    Returns:
        Отсортированный список путей к PDF.
    """
    folder = Path(directory)
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.pdf"))


def batch_extract_pages(
    pdf_paths: Sequence[str | Path] | None = None,
    directory: str | Path = RAW_PDF_DIR,
) -> list[dict[str, Any]]:
    """
    Извлекает страницы со всех PDF и склеивает в один список.

    Каждая страница наследует citation-метаданные своего файла
    (pdf_name, lesson, page), поэтому источник не теряется.

    Args:
        pdf_paths: Явный список файлов. Если None — берём все PDF из directory.
        directory: Папка с PDF, если pdf_paths не задан.

    Returns:
        Список словарей страниц по всем файлам.
    """
    if pdf_paths is None:
        pdf_paths = find_pdfs_in_dir(directory)

    pages: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        pages.extend(extract_text_from_pdf(str(pdf_path)))
    return pages


def batch_extract_chunks(
    pdf_paths: Sequence[str | Path] | None = None,
    directory: str | Path = RAW_PDF_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Чанкит все PDF из папки в единый список с уникальными chunk_id.

    Args:
        pdf_paths: Явный список файлов. Если None — берём все PDF из directory.
        directory: Папка с PDF, если pdf_paths не задан.
        chunk_size: Максимальная длина чанка в символах.
        chunk_overlap: Перекрытие соседних чанков в символах.

    Returns:
        Список чанков по всем файлам.
    """
    pages = batch_extract_pages(pdf_paths=pdf_paths, directory=directory)
    return chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def save_pages_and_chunks(
    pages: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    pages_path: str | Path = PAGES_CSV_PATH,
    chunks_path: str | Path = CHUNKS_CSV_PATH,
) -> None:
    """
    Сохраняет страницы и чанки в CSV.

    Args:
        pages: Список страниц.
        chunks: Список чанков.
        pages_path: Куда сохранить страницы.
        chunks_path: Куда сохранить чанки.
    """
    save_to_csv(build_pages_dataframe(pages), str(pages_path))
    save_to_csv(pd.DataFrame(chunks), str(chunks_path))


if __name__ == "__main__":
    pdfs = find_pdfs_in_dir()
    print(f"PDF в папке: {len(pdfs)}")

    # 1) Парсинг всех PDF по страницам
    pages = batch_extract_pages(pdfs)
    pages_df = build_pages_dataframe(pages)
    print(f"Всего страниц: {len(pages_df)}")
    print(pages_df.head())

    # 2) Лёгкая EDA
    summary = summarize_pages_dataset(pages_df)
    print("\nСводка по датасету:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # 3) Чанкинг всех PDF
    chunks = batch_extract_chunks(pdfs)
    print(f"\nЧанков получено: {len(chunks)}")

    # 4) Сохранение
    save_pages_and_chunks(pages, chunks)
    print(f"сохранено: {PAGES_CSV_PATH}, {CHUNKS_CSV_PATH}")

    # 5) Таблицы из первого файла (для примера)
    if pdfs:
        tables = extract_tables_from_pdf(str(pdfs[0]))
        for i, table in enumerate(tables):
            print(f"\nТаблица {i + 1} из {pdfs[0].name}:")
            print(table.head())
