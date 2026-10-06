"""
Абстракция векторного хранилища с поддержкой Chroma (default) и Qdrant.

Позволяет переключать бэкенд одной строчкой в конфиге / коде.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_STORE_TYPE,
    QDRANT_DEFAULT_URL,
    QDRANT_DEFAULT_VECTOR_SIZE,
    VECTOR_STORE_CONFIG_PATH,
)


class VectorStore(abc.ABC):
    """Базовый интерфейс векторного хранилища."""

    @abc.abstractmethod
    def add(
        self,
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """
        Добавить векторы с метаданными и текстами.

        Args:
            embeddings: Матрица векторов (n, dim).
            metadatas: Метаданные чанков (lesson, page, chunk_id, ...).
            ids: Строковые ID записей.
            documents: Тексты чанков. Без них поиск вернёт пустой текст,
                       и LLM нечего будет цитировать.
        """
        ...

    @abc.abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Семантический поиск top-k.

        Returns:
            Список словарей с ключами: id, score, metadata (включает text)
        """
        ...

    @abc.abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Удалить векторы по ID."""
        ...

    @abc.abstractmethod
    def count(self) -> int:
        """Количество векторов в коллекции."""
        ...

    @abc.abstractmethod
    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Получить записи по списку ID."""
        ...

    @abc.abstractmethod
    def clear(self) -> None:
        """Полностью очистить коллекцию."""
        ...


class ChromaStore(VectorStore):
    """ChromaDB реализация (локальная, файл-based)."""

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        import chromadb
        from chromadb.config import Settings

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine distance
        )

    def add(
        self,
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """
        Добавляет векторы в коллекцию Chroma.

        Важно: тексты передаются через `documents`, а не через metadata —
        Chroma хранит их в отдельном поле и возвращает в `search()`. Если
        не передать documents, поиск вернёт пустой текст.

        Args:
            embeddings: Матрица векторов (n, dim).
            metadatas: Метаданные чанков.
            ids: Строковые ID.
            documents: Тексты чанков (идут в поле documents).
        """
        if ids is None:
            ids = [str(i) for i in range(len(embeddings))]

        if documents is None:
            # Запасной путь: текст мог быть упакован в метаданные
            documents = [str(meta.get("text", "")) for meta in metadatas]

        # Chroma ожидает списки, не numpy массивы
        self.collection.add(
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=documents,
            ids=ids,
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        where = filter_dict or None
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances", "documents"],
        )

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, (doc_id, metadata, distance, document) in enumerate(
                zip(
                    results["ids"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                    results["documents"][0],
                )
            ):
                # Chroma возвращает distance (cosine), переводим в similarity
                score = 1.0 - distance
                hits.append(
                    {
                        "id": doc_id,
                        "score": float(score),
                        "metadata": metadata or {},
                        "text": document or "",
                    }
                )
        return hits

    def delete(self, ids: list[str]) -> None:
        self.collection.delete(ids=ids)

    def count(self) -> int:
        return self.collection.count()

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        results = self.collection.get(ids=ids, include=["metadatas", "documents"])
        out = []
        for doc_id, metadata, document in zip(
            results["ids"], results["metadatas"], results["documents"]
        ):
            out.append(
                {"id": doc_id, "metadata": metadata or {}, "text": document or ""}
            )
        return out

    def clear(self) -> None:
        # Удаляем и пересоздаем коллекцию
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )


class QdrantStore(VectorStore):
    """Qdrant реализация (можно локально через Docker или облако)."""

    def __init__(
        self,
        url: str = QDRANT_DEFAULT_URL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = QDRANT_DEFAULT_VECTOR_SIZE,
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Создаём коллекцию если нет
        collections = self.client.get_collections().collections
        names = {c.name for c in collections}
        if collection_name not in names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def add(
        self,
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """
        Добавляет векторы в коллекцию Qdrant.

        Текст чанка кладётся в payload под ключом "text" — оттуда его
        достаёт `search()`.

        Args:
            embeddings: Матрица векторов (n, dim).
            metadatas: Метаданные чанков.
            ids: Строковые ID.
            documents: Тексты чанков.
        """
        from qdrant_client.models import PointStruct

        if ids is None:
            ids = [str(i) for i in range(len(embeddings))]

        if documents is None:
            # Запасной путь: текст мог быть упакован в метаданные
            documents = [str(meta.get("text", "")) for meta in metadatas]

        points = [
            PointStruct(
                id=idx,
                vector=vec.tolist(),
                payload={**meta, "text": doc},
            )
            for idx, vec, meta, doc in zip(ids, embeddings, metadatas, documents)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if filter_dict:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_dict.items()
            ]
            qdrant_filter = Filter(must=conditions)

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {
                "id": str(hit.id),
                "score": float(hit.score),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                "text": hit.payload.get("text", ""),
            }
            for hit in hits
        ]

    def delete(self, ids: list[str]) -> None:
        from qdrant_client.models import PointIdsList

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )

    def count(self) -> int:
        info = self.client.get_collection(collection_name=self.collection_name)
        return info.points_count

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=True,
        )
        return [
            {
                "id": str(p.id),
                "metadata": {k: v for k, v in p.payload.items() if k != "text"},
                "text": p.payload.get("text", ""),
            }
            for p in points
        ]

    def clear(self) -> None:
        self.client.delete_collection(collection_name=self.collection_name)
        self.__init__(url=self.client._host, collection_name=self.collection_name, vector_size=self.vector_size)


def create_vector_store(config: dict[str, Any] | None = None) -> VectorStore:
    """
    Фабрика векторного хранилища.

    Args:
        config: Словарь конфигурации. Если None — читает из vector_store.yaml
                или использует дефолты (Chroma).

    Пример config:
    {
        "type": "chroma",
        "chroma": {"persist_dir": "data/chroma", "collection": "lessons"}
    }
    или
    {
        "type": "qdrant",
        "qdrant": {"url": "http://localhost:6333", "collection": "lessons", "vector_size": 384}
    }
    """
    if config is None:
        config = load_config()

    store_type = config.get("type", DEFAULT_VECTOR_STORE_TYPE).lower()

    if store_type == "chroma":
        chroma_cfg = config.get("chroma", {})
        return ChromaStore(
            persist_dir=chroma_cfg.get("persist_dir", str(CHROMA_PERSIST_DIR)),
            collection_name=chroma_cfg.get("collection", DEFAULT_COLLECTION_NAME),
        )
    elif store_type == "qdrant":
        qdrant_cfg = config.get("qdrant", {})
        return QdrantStore(
            url=qdrant_cfg.get("url", QDRANT_DEFAULT_URL),
            collection_name=qdrant_cfg.get("collection", DEFAULT_COLLECTION_NAME),
            vector_size=qdrant_cfg.get("vector_size", QDRANT_DEFAULT_VECTOR_SIZE),
        )
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")


def load_config(config_path: str | Path = VECTOR_STORE_CONFIG_PATH) -> dict[str, Any]:
    """Загрузить конфиг из YAML файла."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config: dict[str, Any], config_path: str | Path = VECTOR_STORE_CONFIG_PATH) -> None:
    """Сохранить конфиг в YAML файл."""
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


# Дефолтный конфиг для создания файла (значения — из config.py)
DEFAULT_CONFIG = {
    "type": DEFAULT_VECTOR_STORE_TYPE,
    "chroma": {
        "persist_dir": str(CHROMA_PERSIST_DIR),
        "collection": DEFAULT_COLLECTION_NAME,
    },
    "qdrant": {
        "url": QDRANT_DEFAULT_URL,
        "collection": DEFAULT_COLLECTION_NAME,
        "vector_size": QDRANT_DEFAULT_VECTOR_SIZE,
    },
}


if __name__ == "__main__":
    # Демо: создаём дефолтный конфиг файл
    save_config(DEFAULT_CONFIG)
    print(f"Создан {VECTOR_STORE_CONFIG_PATH} с дефолтными настройками ({DEFAULT_VECTOR_STORE_TYPE})")

    # Быстрый тест Chroma
    store = create_vector_store()
    print(f"Store type: {type(store).__name__}")
    print(f"Initial count: {store.count()}")