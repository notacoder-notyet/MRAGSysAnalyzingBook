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


def main() -> int:
    """
    Прогоняет все этапы пайплайна на всех PDF из data/raw.

    Returns:
        0 — все этапы отработали, 1 — папка с PDF пуста.
    """
    print("=== 1. Discovery ===")
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