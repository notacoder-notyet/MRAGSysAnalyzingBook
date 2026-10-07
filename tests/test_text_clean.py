"""Unit-тесты очистки текста страниц (text_clean)."""

from __future__ import annotations

from text_clean import clean_page_text, garble_ratio, is_junk_token, normalize_unicode


def test_normalize_unicode_maps_math_glyphs() -> None:
    # Математические глифы сводятся к ASCII (NFKC)
    assert normalize_unicode("𝕐 𝑥 𝑤1 𝑎(𝑥)") == "Y x w1 a(x)"


def test_normalize_unicode_strips_emoji() -> None:
    cleaned = normalize_unicode("Внимание 👩\u200d🏫 ✅ 🧩 текст")
    assert "👩" not in cleaned
    assert "🧩" not in cleaned
    assert "текст" in cleaned


def test_junk_token_detection() -> None:
    # Односложные слова — НЕ мусор
    assert not is_junk_token("в")
    assert not is_junk_token("и")
    assert not is_junk_token("a")
    # Одиночные согласные и обрывки без гласных — мусор
    assert is_junk_token("м")
    assert is_junk_token("вд")
    assert is_junk_token("нн")
    # Нормальные слова — не мусор
    assert not is_junk_token("вектор")
    assert not is_junk_token("имеет")


def test_garble_ratio_low_for_clean_text() -> None:
    text = "Функция потерь измеряет ошибку модели на обучающей выборке"
    assert garble_ratio(text) < 0.1


def test_garble_ratio_high_for_scrambled_text() -> None:
    text = "Н И а м п е р е а т в д л л е и н н"
    assert garble_ratio(text) > 0.5


def test_clean_page_text_drops_scrambled_keeps_prose() -> None:
    raw = (
        "Понятие вектора\n"
        "Н И а м п е р е а т в д л л е и н н н у ы\n"
        "Вектор — это направленный отрезок, соединяющий начало и конец.\n"
        "о н к\n"
    )
    cleaned = clean_page_text(raw)
    assert "Понятие вектора" in cleaned
    assert "направленный отрезок" in cleaned
    assert "м п е р е а т" not in cleaned


def test_clean_page_text_empty() -> None:
    assert clean_page_text("") == ""
