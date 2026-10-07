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

    depth = 0
    top_level_functions: dict[str, int] = {}
    total_depth = 0

    for line in js_lines:
        stripped = line.strip()
        # Глубина на момент объявления функции
        if stripped.startswith(("function ", "async function ")):
            for name in ("restoreSession", "init", "initAuth", "submitAuth", "api"):
                if f"function {name}(" in stripped:
                    top_level_functions.setdefault(name, depth)
        depth += line.count("{") - line.count("}")
        total_depth = depth

    assert total_depth == 0, (
        f"app.js: несбалансированные скобки (итоговая глубина {total_depth})"
    )

    for name, func_depth in top_level_functions.items():
        assert func_depth == 0, (
            f"app.js: {name}() объявлена на глубине {func_depth}, а не на верхнем "
            "уровне — значит какая-то функция не закрыта, обработчики не навесятся"
        )

    for name in ("restoreSession", "init", "initAuth", "submitAuth"):
        count = sum(1 for line in js_lines if f"function {name}(" in line)
        assert count == 1, f"app.js: {name} объявлена {count} раз(а), ожидалось 1"

    assert any("init()" in line for line in js_lines), "app.js: init() нигде не вызывается"

    print("  фронтенд: скобки сбалансированы, ключевые функции на верхнем уровне ✓")


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