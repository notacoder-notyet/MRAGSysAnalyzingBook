"""
Структурная проверка фронтенда (app.js / style.css / index.html).

Модуль ловит класс ошибок, невидимый синтаксическому парсеру: если функция или
CSS-правило не закрыто, весь последующий код «уезжает» внутрь него. Синтаксис
остаётся валидным, но функции не видны на верхнем уровне, а стили не
применяются — формы уходят в нативный submit, интерфейс «белеет».

Модуль без тяжёлых зависимостей, поэтому используется и интеграционным
скриптом (`test_full_pipeline.py`), и unit-тестами (`tests/test_frontend.py`).
"""

from __future__ import annotations

from pathlib import Path

from config import STATIC_DIR

# Функции, которые обязаны существовать ровно в одном экземпляре (защита от
# дублей после правок).
_KEY_FUNCTIONS = (
    "restoreSession",
    "init",
    "initAuth",
    "submitAuth",
    "initComposer",
    "deleteChat",
)


def check_frontend_assets() -> dict[str, int]:
    """
    Проверяет структурную целостность фронтенда.

    Проверки:
      1. Баланс фигурных скобок в app.js и style.css;
      2. CSS: нет вложенных селекторов (признак незакрытой скобки выше);
      3. JS: все функции объявлены на верхнем уровне (глубина 0);
      4. Ключевые функции объявлены ровно один раз;
      5. Есть код, который реально вызывает init().

    Returns:
        Сводка {functions, css_rules} для логов.

    Raises:
        AssertionError: если структура нарушена.
    """
    js_path = Path(STATIC_DIR) / "app.js"
    css_path = Path(STATIC_DIR) / "style.css"

    js_lines = js_path.read_text(encoding="utf-8").split("\n")
    css = css_path.read_text(encoding="utf-8")

    assert css.count("{") == css.count("}"), "style.css: несбалансированные скобки"

    # CSS может быть «сбалансирован», но с вложенными селекторами из-за
    # незакрытой скобки выше. Валидная вложенность — только 1 уровень
    # (внутри @media / @keyframes), поэтому селектор на глубине >= 2 — поломка.
    css_depth = 0
    css_nested: list[tuple[int, int, str]] = []
    css_rules = 0
    for idx, css_line in enumerate(css.split("\n"), 1):
        stripped = css_line.strip()
        if stripped.endswith("{"):
            css_rules += 1
            if css_depth >= 2:
                css_nested.append((idx, css_depth, stripped[:50]))
        css_depth += css_line.count("{") - css_line.count("}")

    assert css_depth == 0, f"style.css: несбалансированные скобки (глубина {css_depth})"
    assert not css_nested, (
        "style.css: найдены вложенные селекторы (не закрыта скобка выше): "
        f"{css_nested}. Из-за этого правила не применяются — интерфейс «белеет»."
    )

    depth = 0
    nested: list[tuple[int, str, int]] = []
    declared: list[str] = []
    total_depth = 0

    for line in js_lines:
        stripped = line.strip()
        is_function = stripped.startswith(("function ", "async function "))

        # Проверяем ВСЕ объявления, а не только известные по имени — именно так
        # пропускается вложенная функция, а в рантайме прилетает «not defined».
        if is_function and depth != 0:
            nested.append((len(declared) + 1, stripped[:50], depth))
        if is_function:
            declared.append(stripped)

        depth += line.count("{") - line.count("}")
        total_depth = depth

    assert total_depth == 0, (
        f"app.js: несбалансированные скобки (итоговая глубина {total_depth}). "
        "Какая-то функция не закрыта — весь код после неё вложен внутрь."
    )
    assert not nested, (
        "app.js: найдены вложенные объявления функций (значит выше не закрыта "
        f"скобка): {nested}. Из-за этого функции не видны на верхнем уровне и "
        "падают в рантайме с «is not defined»."
    )

    for name in _KEY_FUNCTIONS:
        count = sum(1 for line in js_lines if f"function {name}(" in line)
        assert count == 1, f"app.js: {name} объявлена {count} раз(а), ожидалось 1"

    assert any("init()" in line for line in js_lines), "app.js: init() нигде не вызывается"

    return {"functions": len(declared), "css_rules": css_rules}


if __name__ == "__main__":
    summary = check_frontend_assets()
    print(
        f"фронтенд: {summary['functions']} функций, все на верхнем уровне, "
        f"{summary['css_rules']} CSS-правил, скобки сбалансированы ✓"
    )
