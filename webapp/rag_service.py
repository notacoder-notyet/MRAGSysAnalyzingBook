"""
Сервис RAG: единый экземпляр пайплайна на всё приложение.

Модель эмбеддингов и клиент Chroma инициализируются один раз (лениво) —
повторная загрузка на каждый HTTP-запрос заняла бы секунды.
"""

from __future__ import annotations

import threading
from typing import Any

from answer_utils import dedupe_sources, sanitize_answer, substitute_refs
from config import DEFAULT_TOP_K, MIN_ANSWER_SCORE
from rag_pipeline import RAGConfig, RAGPipeline, create_rag_pipeline

_pipeline: RAGPipeline | None = None
_lock = threading.Lock()


def get_pipeline() -> RAGPipeline:
    """
    Возвращает общий экземпляр RAG-пайплайна, создавая его при первом вызове.

    Блокировка нужна, т.к. FastAPI может обработать несколько первых
    запросов параллельно и без неё модель загрузилась бы несколько раз.

    Returns:
        Готовый к работе RAGPipeline.
    """
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:  # двойная проверка: второй поток ждёт первый
                _pipeline = create_rag_pipeline(rag_config=RAGConfig(top_k=DEFAULT_TOP_K))
    return _pipeline


def ask(
    question: str, top_k: int | None = None, filter_lesson: int | None = None
) -> dict[str, Any]:
    """
    Задаёт вопрос RAG-пайплайну и нормализует результат для API.

    Args:
        question: Текст вопроса.
        top_k: Сколько чанков поднять (None — значение из конфига).
        filter_lesson: Ограничить поиск одним уроком.

    Returns:
        Словарь {answer, sources} где sources — список
        {lesson, page, score}, отфильтрованный по MIN_ANSWER_SCORE.
    """
    result = get_pipeline().ask(question, top_k=top_k, filter_lesson=filter_lesson)

    # Оставляем только релевантные источники: слабые совпадения
    # вводят пользователя в заблуждение ложной ссылкой
    sources = [
        {
            "lesson": chunk["metadata"].get("lesson"),
            "page": chunk["metadata"].get("page"),
            "score": round(float(chunk["score"]), 4),
        }
        for chunk in result.retrieved_chunks
        if float(chunk["score"]) >= MIN_ANSWER_SCORE and chunk["metadata"].get("lesson") is not None
    ]

    # 1) [N] → реальные «(Урок N, Страница M)» из метаданных чанков
    #    (модель не может выдумать номер: он подставляется нами);
    # 2) очистка текста от LaTeX / сносок / Markdown.
    answer = sanitize_answer(substitute_refs(result.answer, result.retrieved_chunks))
    return {"answer": answer, "sources": dedupe_sources(sources)}
