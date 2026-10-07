"""
Полный интеграционный тест всех компонентов пайплайна на всей коллекции PDF.

Проверяет сквозной путь: PDF из data/raw → страницы → чанки → эмбеддинги →
VectorStore → LLM-клиент → RAG-ответ с citation.
"""

from config import DEFAULT_TOP_K
from embeddings import embed_chunks, embed_texts, load_embedding_model
from llm import create_llm_client
from pars_pdf import batch_extract_chunks, batch_extract_pages, find_pdfs_in_dir
from rag_pipeline import RAGConfig, create_rag_pipeline
from vector_store import create_vector_store


def check_frontend_assets() -> None:
    """
    Проверяет структурную целостность фронтенда.

    Ловит класс ошибок, невидимый парсеру: если функция не закрыта, все
    последующие функции и регистрация обработчиков оказываются вложенными
    внутрь неё. Синтаксис при этом валиден, но `init` объявляется внутри
    другой функции и обработчики не навешиваются — формы уходят в нативный
    submit (страница перезагружается, поля очищаются, пароль попадает в URL).

    Проверяем:
      1. Баланс фигурных скобок в app.js и style.css;
      2. Ключевые функции объявлены на верхнем уровне (глубина 0);
      3. Каждая из них объявлена ровно один раз;
      4. Есть код, который реально вызывает init().
    """
    from pathlib import Path

    from config import STATIC_DIR

    js_path = Path(STATIC_DIR) / "app.js"
    css_path = Path(STATIC_DIR) / "style.css"

    js_lines = js_path.read_text(encoding="utf-8").split("\n")
    css = css_path.read_text(encoding="utf-8")

    assert css.count("{") == css.count("}"), "style.css: несбалансированные скобки"

    # CSS может быть «сбалансирован», но при этом селекторы вложены друг в
    # друга из-за незакрытой скобки выше. Тогда браузер игнорирует правила, и
    # интерфейс «белеет»: пропадают рамки, обрезается PDF, не виден индикатор
    # ожидания. Валидная вложенность только на 1 уровне — внутри @media /
    # @keyframes, поэтому селектор на глубине >= 2 — признак поломки.
    css_depth = 0
    css_nested: list[tuple[int, int, str]] = []
    for idx, css_line in enumerate(css.split("\n"), 1):
        stripped_css = css_line.strip()
        if stripped_css.endswith("{") and css_depth >= 2:
            css_nested.append((idx, css_depth, stripped_css[:50]))
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

        # Важно: проверяем ВСЕ объявления, а не только известные по имени —
        # именно так пропускается вложенная функция (initComposer), и потом
        # в рантайме прилетает «X is not defined».
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

    # Ключевые функции объявлены по одному разу (защита от дублей после правок)
    for name in ("restoreSession", "init", "initAuth", "submitAuth", "initComposer", "deleteChat"):
        count = sum(1 for line in js_lines if f"function {name}(" in line)
        assert count == 1, f"app.js: {name} объявлена {count} раз(а), ожидалось 1"

    assert any("init()" in line for line in js_lines), "app.js: init() нигде не вызывается"

    print(
        f"  фронтенд: {len(declared)} функций, все на верхнем уровне, "
        "скобки сбалансированы ✓"
    )


def main() -> int:
    """
    Прогоняет все этапы пайплайна на всех PDF из data/raw.

    Returns:
        0 — все этапы отработали, 1 — папка с PDF пуста.
    """
    print("=== 0. Frontend assets ===")
    check_frontend_assets()

    print("\n=== 1. Discovery ===")
    pdfs = find_pdfs_in_dir()
    if not pdfs:
        print("PDF не найдены в data/raw. Запустите: python download_pdfs.py")
        return 1
    print(f"PDF: {len(pdfs)}")

    print("\n=== 2. Parsing ===")
    pages = batch_extract_pages(pdfs)
    lessons = {p.get("lesson") for p in pages}
    print(f"Pages: {len(pages)}, lessons: {len(lessons)}")

    print("\n=== 3. Chunking ===")
    chunks = batch_extract_chunks(pdfs)
    chunk_ids = [c["chunk_id"] for c in chunks]
    print(f"Chunks: {len(chunks)}, unique chunk_id: {len(set(chunk_ids))}")

    print("\n=== 4. Embeddings ===")
    model = load_embedding_model()
    embeddings, meta_df = embed_chunks(chunks, model=model, show_progress=False)
    print(f"Embeddings: {embeddings.shape}, dtype: {embeddings.dtype}")
    print(f"Lessons in meta: {meta_df['lesson'].nunique()}")

    print("\n=== 5. Vector Store ===")
    store = create_vector_store()
    print(f"Store: {type(store).__name__}, count: {store.count()}")

    # Проверяем фильтрацию по метаданным: поиск в пределах одного урока
    query_vec = embed_texts(
        ["Что такое функция потерь?"], model=model, show_progress=False, role="query"
    )[0]
    filtered = store.search(query_vec, top_k=5, filter_dict={"lesson": 6})
    lessons_in_result = {h["metadata"].get("lesson") for h in filtered}
    print(f"Фильтр lesson=6 -> уроков в выдаче: {lessons_in_result}")
    assert lessons_in_result <= {6}, "фильтр по уроку не сработал"

    print("\n=== 6. LLM Client ===")
    llm = create_llm_client()
    print(f"LLM: {type(llm).__name__}")

    print("\n=== 7. RAG Pipeline ===")
    pipeline = create_rag_pipeline(rag_config=RAGConfig(top_k=DEFAULT_TOP_K))
    result = pipeline.ask("Что такое функция потерь?")
    print(f"Question: {result.query}")
    print(f"Sources: {result.sources}")
    print(f"Chunks retrieved: {len(result.retrieved_chunks)}")

    print("\n✅ ALL COMPONENTS WORKING!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())