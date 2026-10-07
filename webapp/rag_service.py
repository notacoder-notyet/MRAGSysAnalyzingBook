"""
Сервис RAG: единый экземпляр пайплайна на всё приложение.

Модель эмбеддингов и клиент Chroma инициализируются один раз (лениво) —
повторная загрузка на каждый HTTP-запрос заняла бы секунды.
"""

from __future__ import annotations

import re
import threading
from typing import Any

from config import DEFAULT_TOP_K, MIN_ANSWER_SCORE
from rag_pipeline import RAGConfig, RAGPipeline, create_rag_pipeline

_pipeline: RAGPipeline | None = None
_lock = threading.Lock()

# Служебные метки-сноски, которые некоторые модели вставляют после фактов
# (напр. 【5†L1-L4】). В интерфейсе они выглядят как «мусор».
_CITATION_META = re.compile(r"【[^】]*】")

# Читаемые замены частым LaTeX-командам: модель игнорирует запрет на LaTeX,
# а пользователю нужен обычный текст, поэтому подчищаем на выходе.
_LATEX_SYMBOLS = {
    "nabla": "∇",
    "partial": "∂",
    "times": "×",
    "cdot": "·",
    "pm": "±",
    "approx": "≈",
    "le": "≤",
    "ge": "≥",
    "to": "→",
    "infty": "∞",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "omega": "ω",
    "Delta": "Δ",
    "Sigma": "Σ",
}


def _sanitize_answer(text: str) -> str:
    """
    Приводит ответ модели к «вменяемому» виду.

    Модель может проигнорировать запрет на LaTeX и разметку, поэтому ответ
    дополнительно очищается: убираются сноски вида 【...】, LaTeX-делимитеры
    и команды, нормализуются «экзотические» пробелы и дефисы.

    Args:
        text: Сырой ответ модели.

    Returns:
        Очищенный текст, пригодный для показа пользователю.
    """
    if not text:
        return text

    # 1. Сноски-метки вида 【5†L1-L4】
    text = _CITATION_META.sub("", text)

    # 2. «Экзотические» пробелы (узкий неразрывный и т.п.) и дефисы
    text = (
        text.replace("\u202f", " ")
        .replace("\u00a0", " ")
        .replace("\u2011", "-")
    )

    # 3. LaTeX-делимитеры формул: \( \) \[ \] и $
    text = re.sub(r"\\[\[\]()]", "", text)
    text = text.replace("$$", "").replace("$", "")

    # 4. Команды с аргументами в фигурных скобках → читаемый вид.
    #    Их обрабатываем ДО команд без аргументов, иначе «\text» распадётся
    #    на «text{...}».
    text = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\text(?:rm|bf|it|sf|tt)?\s*\{([^{}]*)\}", r"\1", text)

    # 5. Команды без аргументов (пробелы, скобки, многоточия)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\qquad", " ").replace("\\quad", " ")
    text = text.replace("\\dots", "…").replace("\\ldots", "…")
    text = re.sub(r"\\[,;:!\s]", " ", text)
    text = text.replace("\\\\", "\n")

    # 6. Остальные \команды → символ или просто слово без backslash
    text = re.sub(
        r"\\([A-Za-z]+)",
        lambda m: _LATEX_SYMBOLS.get(m.group(1), m.group(1)),
        text,
    )
    text = text.replace("\\", "")

    # 7. Снимаем остатки Markdown, если модель его всё же использовала
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*]\s+", "• ", text)

    # 8. Схлопываем лишние пробелы и пустые строки
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
                _pipeline = create_rag_pipeline(
                    rag_config=RAGConfig(top_k=DEFAULT_TOP_K)
                )
    return _pipeline


def ask(question: str, top_k: int | None = None, filter_lesson: int | None = None) -> dict[str, Any]:
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
    result = get_pipeline().ask(
        question, top_k=top_k, filter_lesson=filter_lesson
    )

    # Оставляем только релевантные источники: слабые совпадения
    # вводят пользователя в заблуждение ложной ссылкой
    sources = [
        {
            "lesson": chunk["metadata"].get("lesson"),
            "page": chunk["metadata"].get("page"),
            "score": round(float(chunk["score"]), 4),
        }
        for chunk in result.retrieved_chunks
        if float(chunk["score"]) >= MIN_ANSWER_SCORE
        and chunk["metadata"].get("lesson") is not None
    ]

    return {"answer": _sanitize_answer(result.answer), "sources": _dedupe_sources(sources)}


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Оставляет по одной (самой релевантной) странице на урок.

    Без этого один урок занимает несколько ссылок (например, «Урок 39 · стр. 14»
    и «Урок 39 · стр. 17»), и список источников выглядит как набор дубликатов.

    Args:
        sources: Список источников {lesson, page, score} в любом порядке.

    Returns:
        Отсортированный по убыванию score список без повторов уроков.
    """
    ordered = sorted(sources, key=lambda src: src["score"], reverse=True)
    seen_lessons: set[int] = set()
    unique: list[dict[str, Any]] = []
    for src in ordered:
        if src["lesson"] in seen_lessons:
            continue
        seen_lessons.add(src["lesson"])
        unique.append(src)
    return unique
