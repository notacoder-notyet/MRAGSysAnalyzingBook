"""
Очистка текста страниц PDF и метрика «мусорности».

Проблема: слайды-презентации содержат диаграммы, таблицы, вертикальные подписи
и декоративные символы. pdfplumber читает текст по позиции и на таких страницах
перемешивает символы (`Н И а м п е р е а т...`), а также тащит математические
Unicode-глифы (𝕐, 𝑥, 𝑎) и эмодзи.

Модуль решает три задачи:
1. `normalize_unicode` — NFKC-нормализация (𝕐→Y, 𝑥→x, 𝑤1→w1) и удаление эмодзи;
2. `clean_page_text` — построчная фильтрация «мусорных» строк;
3. `garble_ratio` — метрика доли мусорных токенов (общая для парсера и отчёта).

Модуль без тяжёлых зависимостей (стандартная библиотека) — тестируется в CI.
"""

from __future__ import annotations

import re
import unicodedata

# Гласные (латиница + кириллица) — для определения «обрывков» без гласных.
VOWELS = frozenset("аеёиоуыэюяAEIOUYaeiouy")

# Односложные слова, которые НЕ являются мусором: предлоги/союзы/артикли.
# Без этого метрика считает мусором обычный русский текст («в», «и», «с»).
VALID_SINGLE_LETTER_WORDS = frozenset("а в и к о с у я э a i".split())

# Эмодзи, стрелки, пиктограммы, модификаторы — в тексте для поиска бесполезны.
_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff\u2600-\u27bf\u2b00-\u2bff\ufe00-\ufe0f\u20e3\u2190-\u21ff\u2500-\u25ff]"
)

# Всё, кроме букв (оставляем кириллицу и латиницу).
_NON_LETTERS_RE = re.compile(r"[^А-Яа-яЁёA-Za-z]")


def normalize_unicode(text: str) -> str:
    """
    Нормализует Unicode-текст страницы.

    NFKC сводит математические глифы к ASCII (𝕐→Y, 𝑥→x, 𝑤→w, 𝟏→1) и
    композирует диакритику; затем удаляются эмодзи и декоративные символы.

    Args:
        text: Сырой текст.

    Returns:
        Очищенный текст.
    """
    if not text:
        return text
    normalized = unicodedata.normalize("NFKC", text)
    return _EMOJI_RE.sub("", normalized)


def _letters(token: str) -> str:
    """Возвращает только буквы токена."""
    return _NON_LETTERS_RE.sub("", token)


def is_junk_token(token: str) -> bool:
    """
    Считает токен «мусорным» — признак плохо извлечённой диаграммы/таблицы.

    Мусор:
    - одиночная буква, не являющаяся осмысленным односложным словом;
    - обрывок из 2–3 букв без гласных (типично при чтении по колонкам).

    Args:
        token: Токен (слово).

    Returns:
        True, если токен похож на мусор.
    """
    letters = _letters(token)
    if not letters:
        return False
    if len(letters) == 1:
        return letters.lower() not in VALID_SINGLE_LETTER_WORDS
    if len(letters) <= 3 and not (set(letters) & VOWELS):
        return True
    return False


def garble_ratio(text: str) -> float:
    """
    Доля мусорных токенов от 0.0 до 1.0 — предиктор плохого retrieval.

    Args:
        text: Текст страницы.

    Returns:
        Доля мусорных токенов (0.0, если токенов нет).
    """
    tokens = [t for t in re.split(r"\s+", text or "") if t]
    if not tokens:
        return 0.0
    junk = sum(1 for token in tokens if is_junk_token(token))
    return junk / len(tokens)


def clean_page_text(text: str) -> str:
    """
    Полная очистка текста страницы: Unicode + фильтрация мусорных строк.

    Строка отбрасывается, если она почти целиком состоит из мусорных токенов
    (перемешанные подписи диаграммы). Содержательные строки (заголовки, абзацы)
    сохраняются — на тестовом корпусе сохраняется ~98% символов.

    Args:
        text: Сырой текст страницы.

    Returns:
        Очищенный текст.
    """
    normalized = normalize_unicode(text or "")
    kept: list[str] = []

    for line in normalized.split("\n"):
        tokens = line.split()
        if not tokens:
            continue
        letters = [_letters(t) for t in tokens]
        lengths = [len(x) for x in letters if x]
        junk = sum(1 for x in letters if is_junk_token(x))
        junk_share = junk / len(tokens)
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0

        # Почти целиком мусорная строка
        if junk_share >= 0.6:
            continue
        # Длинная строка из сплошных коротких обрывков (перемешанная диаграмма)
        if len(tokens) >= 3 and avg_len < 3.0 and junk_share >= 0.34:
            continue
        kept.append(line.strip())

    result = "\n".join(kept)
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()
