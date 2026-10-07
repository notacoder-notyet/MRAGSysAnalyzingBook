"""
Эмбеддинги чанков для baseline RAG.

Считает локальные sentence-transformers векторы и сохраняет их
вместе с citation-метаданными для последующей загрузки в векторную БД.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_META_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_NORMALIZE,
)


def load_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformer:
    """
    Загружает модель эмбеддингов sentence-transformers.

    Args:
        model_name: Имя модели на Hugging Face / локальный путь.

    Returns:
        Инициализированная SentenceTransformer.
    """
    # Первый запуск скачает веса; дальше берёт из кэша
    return SentenceTransformer(model_name)


def model_prefixes(model_name: str | None = None) -> tuple[str, str]:
    """
    Возвращает префиксы (query, passage) для модели эмбеддингов.

    Часть моделей обучена с обязательными префиксами, и без них качество
    поиска заметно падает:
      - E5 (intfloat/multilingual-e5-*): "query: " для вопроса,
        "passage: " для документа. Это не опционально — так модель обучалась.
      - BGE-m3 префиксов не требует.

    Args:
        model_name: Имя модели (None — взять из конфига).

    Returns:
        Кортеж (префикс запроса, префикс документа).
    """
    name = (model_name or DEFAULT_EMBEDDING_MODEL).lower()
    if "e5" in name:
        return "query: ", "passage: "
    return "", ""


def embed_texts(
    texts: Sequence[str],
    model: SentenceTransformer | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    normalize: bool = EMBEDDING_NORMALIZE,
    show_progress: bool = True,
    role: str = "passage",
) -> np.ndarray:
    """
    Считает эмбеддинги для списка текстов.

    Args:
        texts: Тексты для кодирования.
        model: Уже загруженная модель (если None — загрузится model_name).
        model_name: Имя модели, если model не передан.
        batch_size: Размер батча encode.
        normalize: L2-нормализация (удобно для cosine через dot product).
        show_progress: Показывать прогресс encode.
        role: "query" для вопроса или "passage" для документа. Для E5-моделей
              от этого зависит обязательный префикс — перепутать нельзя,
              иначе качество поиска падает.

    Returns:
        Матрица shape (n_texts, dim) типа float32.
    """
    if model is None:
        model = load_embedding_model(model_name)

    query_prefix, passage_prefix = model_prefixes(model_name)
    prefix = query_prefix if role == "query" else passage_prefix

    # Пустые строки заменяем на пробел, чтобы encode не падал на ""
    safe_texts = [
        prefix + t if (t or "").strip() else " " for t in texts
    ]

    vectors = model.encode(
        safe_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_chunks(
    chunks: list[dict[str, Any]],
    model: SentenceTransformer | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    text_key: str = "text",
    batch_size: int = EMBEDDING_BATCH_SIZE,
    normalize: bool = EMBEDDING_NORMALIZE,
    show_progress: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Эмбеддит чанки и возвращает матрицу векторов + таблицу метаданных.

    Args:
        chunks: Список чанков (результат chunk_pages).
        model: Уже загруженная модель.
        model_name: Имя модели, если model не передан.
        text_key: Ключ с текстом чанка.
        batch_size: Размер батча encode.
        normalize: L2-нормализация векторов.
        show_progress: Показывать прогресс encode.

    Returns:
        Кортеж (embeddings[n, dim], meta_df) где meta_df содержит
        citation-поля без тяжёлого дублирования векторов в CSV.
    """
    if not chunks:
        empty_meta = pd.DataFrame(
            columns=["chunk_id", "pdf_name", "lesson", "page", "chunk_index", "text"]
        )
        return np.zeros((0, 0), dtype=np.float32), empty_meta

    texts = [str(chunk.get(text_key, "")) for chunk in chunks]
    embeddings = embed_texts(
        texts=texts,
        model=model,
        model_name=model_name,
        batch_size=batch_size,
        normalize=normalize,
        show_progress=show_progress,
    )

    # Метаданные храним отдельно от .npy — так проще грузить в Chroma/Qdrant
    meta_df = pd.DataFrame(chunks)
    preferred_cols = ["chunk_id", "pdf_name", "lesson", "page", "chunk_index", "text"]
    ordered = [c for c in preferred_cols if c in meta_df.columns]
    other = [c for c in meta_df.columns if c not in ordered]
    meta_df = meta_df[ordered + other]
    return embeddings, meta_df


def save_embeddings(
    embeddings: np.ndarray,
    meta_df: pd.DataFrame,
    embeddings_path: str | Path,
    meta_path: str | Path,
) -> None:
    """
    Сохраняет матрицу эмбеддингов (.npy) и метаданные чанков (.csv).

    Args:
        embeddings: Матрица (n, dim).
        meta_df: Метаданные чанков в том же порядке строк, что и embeddings.
        embeddings_path: Путь к .npy файлу.
        meta_path: Путь к CSV с метаданными.
    """
    embeddings_path = Path(embeddings_path)
    meta_path = Path(meta_path)

    # Создаём родительские директории при необходимости
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    if len(meta_df) != len(embeddings):
        raise ValueError(
            f"Число строк meta ({len(meta_df)}) != числу векторов ({len(embeddings)})"
        )

    np.save(embeddings_path, embeddings)
    meta_df.to_csv(meta_path, index=False)


def load_embeddings(
    embeddings_path: str | Path,
    meta_path: str | Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Загружает эмбеддинги и метаданные с диска.

    Args:
        embeddings_path: Путь к .npy.
        meta_path: Путь к CSV метаданных.

    Returns:
        Кортеж (embeddings, meta_df).
    """
    embeddings = np.load(embeddings_path)
    meta_df = pd.read_csv(meta_path)
    if len(meta_df) != len(embeddings):
        raise ValueError(
            f"Число строк meta ({len(meta_df)}) != числу векторов ({len(embeddings)})"
        )
    return embeddings.astype(np.float32), meta_df


def cosine_topk(
    query_vector: np.ndarray,
    embeddings: np.ndarray,
    meta_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """
    Простой baseline-retriever: top-k по cosine (для L2-нормированных векторов = dot).

    Args:
        query_vector: Вектор запроса shape (dim,) или (1, dim).
        embeddings: Матрица чанков (n, dim).
        meta_df: Метаданные чанков.
        top_k: Сколько ближайших чанков вернуть.

    Returns:
        DataFrame top-k чанков с колонкой score (убывание).
    """
    if embeddings.size == 0:
        return meta_df.head(0).assign(score=pd.Series(dtype=float))

    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    # Для normalize_embeddings=True cosine similarity = скалярное произведение
    scores = embeddings @ q
    k = min(top_k, len(scores))
    top_idx = np.argpartition(-scores, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    result = meta_df.iloc[top_idx].copy()
    result.insert(0, "score", scores[top_idx])
    return result.reset_index(drop=True)


if __name__ == "__main__":
    from pars_pdf import batch_extract_chunks, find_pdfs_in_dir

    pdfs = find_pdfs_in_dir()
    print(f"PDF в папке: {len(pdfs)}")

    # Чанки со всех PDF сразу; размеры берутся из config.py
    chunks = batch_extract_chunks(pdfs)
    print(f"Чанков: {len(chunks)}")

    model = load_embedding_model(DEFAULT_EMBEDDING_MODEL)
    embeddings, meta_df = embed_chunks(chunks, model=model)
    save_embeddings(embeddings, meta_df, str(CHUNK_EMBEDDINGS_PATH), str(CHUNK_META_PATH))

    print(f"embeddings shape: {embeddings.shape}")
    print(f"сохранено: {CHUNK_EMBEDDINGS_PATH}, {CHUNK_META_PATH}")
    print(f"Уроков в индексе: {meta_df['lesson'].nunique()}")

    # Быстрая проверка retriever на одном вопросе
    query = "Что такое функция потерь?"
    q_vec = embed_texts([query], model=model, show_progress=False, role="query")[0]
    hits = cosine_topk(q_vec, embeddings, meta_df, top_k=3)
    print("\nTop-3 по запросу:", query)
    print(hits[["score", "lesson", "page", "text"]])
