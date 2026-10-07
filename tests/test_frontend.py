"""Структурные проверки фронтенда (app.js / style.css / index.html)."""

from __future__ import annotations

from pathlib import Path

from config import STATIC_DIR
from frontend_check import check_frontend_assets


def test_frontend_structure_is_valid() -> None:
    """app.js и style.css не имеют незакрытых скобок и вложенных правил."""
    summary = check_frontend_assets()
    assert summary["functions"] > 0
    assert summary["css_rules"] > 0


def test_index_references_assets_and_pdf() -> None:
    html = (Path(STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    assert "/static/app.js" in html
    assert "/static/style.css" in html
    assert 'id="pdf-panel"' in html
    assert 'id="pdf-fullscreen"' in html


def test_pdfjs_vendor_is_local() -> None:
    """PDF.js подключён локально — без внешнего CDN."""
    vendor = Path(STATIC_DIR) / "vendor"
    assert (vendor / "pdf.min.js").is_file()
    assert (vendor / "pdf.worker.min.js").is_file()
