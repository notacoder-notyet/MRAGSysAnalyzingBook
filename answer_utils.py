"""
Утилиты обработки ответа модели: очистка текста, дедуп источников, ссылки.

Модуль намеренно **без тяжёлых зависимостей** (только стандартная библиотека).
Это позволяет импортировать его в лёгких контекстах (Pydantic-схемы API,
unit-тесты, CI) без загрузки torch / sentence-transformers / chromadb.

Зачем вынесено отдельно:
- ``webapp/rag_service.py`` зависит от тяжёлого RAG-пайплайна, поэтому
  тестировать из него чистые функции дорого;
- ``webapp/schemas.py`` нужно чистить ответ при сериализации, но тянуть туда
  RAG недопустимо.
"""

from __future__ import annotations

import re
from typing import Any

# Служебные метки-сноски, которые некоторые модели вставляют после фактов
# (напр. 【5†L1-L4】). В интерфейсе они выглядят как «мусор».
_CITATION_META = re.compile(r"【[^】]*】")

# Ссылки-номера на фрагменты контекста: [1], [2] или [1, 2].
_REF_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")

# Читаемые замены частым LaTeX-командам: модель может проигнорировать запрет
# на LaTeX, а пользователю нужен обычный текст, поэтому подчищаем на выходе.
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


def sanitize_answer(text: str) -> str:
    """
    Приводит ответ модели к «вменяемому» виду.

    Убирает сноски вида 【...】, LaTeX-делимитеры и команды, остатки Markdown,
    нормализует «экзотические» пробелы и дефисы.

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
    text = text.replace("\u202f", " ").replace("\u00a0", " ").replace("\u2011", "-")

    # 3. LaTeX-делимитеры формул: \( \) \[ \] и $
    text = re.sub(r"\\[\[\]()]", "", text)
    text = text.replace("$$", "").replace("$", "")

    # 4. Команды с аргументами в фигурных скобках → читаемый вид.
    #    Их обрабатываем ДО команд без аргументов, иначе «\\text» распадётся
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


def dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def substitute_refs(answer: str, chunks: list[dict[str, Any]]) -> str:
    """
    Подставляет реальные «(Урок N, Страница M)» вместо ссылок-номеров [1], [2].

    LLM плохо и нестабильно копирует числа, поэтому в промпте просим ссылаться
    на фрагменты по их номеру в контексте. Номер → метаданные берём из самих
    чанков, поэтому ссылка в ответе **не может быть выдумана** (как Perplexity).
    Неверные номера (вне диапазона) просто отбрасываются.

    Args:
        answer: Текст ответа модели со ссылками вида [1], [1, 2].
        chunks: Чанки контекста в том же порядке, что и в промпте.

    Returns:
        Ответ, в котором ссылки заменены на текст «(Урок N, Страница M)».
        Дальше фронтенд делает эти упоминания кликабельными.
    """
    if not answer or not chunks:
        return answer

    refs: list[dict[str, Any]] = [
        {
            "lesson": (chunk.get("metadata") or {}).get("lesson"),
            "page": (chunk.get("metadata") or {}).get("page"),
        }
        for chunk in chunks
    ]

    def _replace(match: re.Match[str]) -> str:
        parts: list[str] = []
        for number in re.findall(r"\d+", match.group(1)):
            index = int(number) - 1
            if not 0 <= index < len(refs):
                continue
            meta = refs[index]
            if meta["lesson"] is not None and meta["page"] is not None:
                parts.append(f"(Урок {meta['lesson']}, Страница {meta['page']})")
        return " ".join(parts)

    return _REF_RE.sub(_replace, answer)
