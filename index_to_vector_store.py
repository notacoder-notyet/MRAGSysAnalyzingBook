"""
Индексация существующих эмбеддингов (из .npy + .csv) в векторное хранилище.

Использование:
    python index_to_vector_store.py           # Chroma (default из config)
    python index_to_vector_store.py --qdrant  # Qdrant (если запущен на localhost:6333)
"""

import argparse

import pandas as pd

from config import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_META_PATH,
    INDEX_BATCH_SIZE,
    VECTOR_STORE_CONFIG_PATH,
)
from embeddings import load_embeddings
from vector_store import create_vector_store, load_config, save_config


def main():
    parser = argparse.ArgumentParser(description="Индексация чанков в векторное хранилище")
    parser.add_argument(
        "--qdrant", action="store_true", help="Использовать Qdrant вместо Chroma"
    )
    parser.add_argument(
        "--embeddings",
        default=str(CHUNK_EMBEDDINGS_PATH),
        help="Путь к .npy файлу",
    )
    parser.add_argument(
        "--meta", default=str(CHUNK_META_PATH), help="Путь к CSV метаданным"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Очистить коллекцию перед индексацией"
    )
    args = parser.parse_args()

    # Подготовка конфига
    config = load_config(VECTOR_STORE_CONFIG_PATH)
    if args.qdrant:
        config["type"] = "qdrant"
        save_config(config, VECTOR_STORE_CONFIG_PATH)
        print("Переключено на Qdrant")

    # Создаём хранилище
    store = create_vector_store(config)
    print(f"Используется: {type(store).__name__}")

    if args.clear:
        store.clear()
        print("Коллекция очищена")

    # Загружаем эмбеддинги и метаданные
    print(f"Загрузка: {args.embeddings} + {args.meta}")
    embeddings, meta_df = load_embeddings(args.embeddings, args.meta)
    print(f"Загружено: {embeddings.shape[0]} чанков, dim={embeddings.shape[1]}")

    # Подготовка метаданных для VectorStore
    # Chroma/Qdrant хранят text отдельно, остальное в metadata.
    # ID делаем составным (lesson_page_chunkindex), а не по порядку в CSV:
    # так индекс переживает пересортировку строк и не конфликтует
    # с уроками, где несколько PDF (lesson_39_1 и lesson_39_2).
    metadatas: list[dict[str, object]] = []
    ids: list[str] = []
    seen_ids: set[str] = set()

    for _, row in meta_df.iterrows():
        lesson_value = row["lesson"]
        lesson = int(lesson_value) if pd.notna(lesson_value) else None
        page = int(row["page"])
        chunk_index = int(row["chunk_index"])
        doc_id = f"{lesson}_{page}_{chunk_index}"

        # Гарантируем уникальность: при коллизии добавляем порядковый номер
        if doc_id in seen_ids:
            suffix = 1
            while f"{doc_id}_dup{suffix}" in seen_ids:
                suffix += 1
            doc_id = f"{doc_id}_dup{suffix}"
        seen_ids.add(doc_id)

        metadatas.append(
            {
                "chunk_id": int(row["chunk_id"]),
                "pdf_name": str(row["pdf_name"]),
                "lesson": lesson,
                "page": page,
                "chunk_index": chunk_index,
            }
        )
        ids.append(doc_id)

    print(f"Уникальных ID: {len(set(ids))} из {len(ids)}")

    # Тексты чанков передаём отдельно (documents), а не в metadata:
    # Chroma хранит их в поле documents, Qdrant — в payload["text"].
    # Без этого поиск вернёт пустой текст и LLM нечего будет цитировать.
    documents = meta_df["text"].fillna("").astype(str).tolist()

    # Индексация батчами (на случай большого объёма)
    batch_size = INDEX_BATCH_SIZE
    for i in range(0, len(embeddings), batch_size):
        batch_emb = embeddings[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_docs = documents[i : i + batch_size]
        store.add(batch_emb, batch_meta, batch_ids, batch_docs)
        print(f"  Индексировано {min(i + batch_size, len(embeddings))} / {len(embeddings)}")

    print(f"Готово! Всего в хранилище: {store.count()} векторов")

    # Быстрый тест поиска
    from embeddings import load_embedding_model, embed_texts

    model = load_embedding_model()
    query = "Что такое функция потерь?"
    q_vec = embed_texts([query], model=model, show_progress=False)[0]

    hits = store.search(q_vec, top_k=3)
    print(f"\nТестовый поиск: '{query}'")
    for h in hits:
        text_preview = (h.get("text") or "").strip().replace("\n", " ")[:90]
        print(
            f"  score={h['score']:.4f} | lesson={h['metadata'].get('lesson')} | "
            f"page={h['metadata'].get('page')}"
        )
        print(f"    текст: {text_preview or '<ПУСТО — проверьте documents>'}")


if __name__ == "__main__":
    main()