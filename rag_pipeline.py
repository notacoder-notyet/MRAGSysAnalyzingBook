"""
Полный RAG пайплайн: Retrieval + Generation с citation.

Пайплайн:
1. Пользователь задаёт вопрос
2. Embedding модели → вектор вопроса
3. VectorStore поиск top-k (с опциональным фильтром по lesson)
4. Формирование контекста из найденных чанков
5. Промпт с инструкцией по citation
6. LLM генерация ответа
7. Возврат ответа + источники (lesson, page)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FILTER_LESSON,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_TOP_K,
    RAG_SYSTEM_PROMPT,
)
from embeddings import load_embedding_model, embed_texts
from vector_store import VectorStore, create_vector_store
from llm import LLMClient, create_llm_client, LLMResponse


@dataclass
class RAGConfig:
    """Конфигурация RAG пайплайна (значения по умолчанию — из config.py)."""

    top_k: int = DEFAULT_TOP_K
    filter_lesson: int | None = DEFAULT_FILTER_LESSON
    temperature: float = DEFAULT_LLM_TEMPERATURE
    max_tokens: int | None = DEFAULT_LLM_MAX_TOKENS
    embedding_model: str = DEFAULT_EMBEDDING_MODEL


@dataclass
class RAGResult:
    """Результат RAG пайплайна."""
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    query: str = ""
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    llm_response: LLMResponse | None = None
    config: RAGConfig | None = None


class RAGPipeline:
    """Основной RAG пайплайн."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        llm_client: LLMClient | None = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        config: RAGConfig | None = None,
    ):
        self.vector_store = vector_store or create_vector_store()
        self.llm_client = llm_client or create_llm_client()
        self.embedding_model = load_embedding_model(embedding_model_name)
        self.config = config or RAGConfig()
        self.config.embedding_model = embedding_model_name

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_lesson: int | None = None,
    ) -> list[dict[str, Any]]:
        """Поиск релевантных чанков."""
        q_vec = embed_texts(
            [query], model=self.embedding_model, show_progress=False, role="query"
        )[0]
        k = top_k or self.config.top_k
        flt = filter_lesson if filter_lesson is not None else self.config.filter_lesson
        filter_dict = {"lesson": flt} if flt is not None else None
        return self.vector_store.search(q_vec, top_k=k, filter_dict=filter_dict)

    def build_context(self, chunks: list[dict[str, Any]]) -> str:
        """Формирует контекст из чанков для промпта."""
        if not chunks:
            return "Нет релевантных фрагментов."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            lesson = meta.get("lesson", "?")
            page = meta.get("page", "?")
            text = chunk.get("text", "").strip()
            parts.append(f"[Источник {i}: Урок {lesson}, Страница {page}]\n{text}")
        return "\n\n---\n\n".join(parts)

    def build_prompt(self, query: str, context: str) -> list[dict[str, str]]:
        """Строит сообщения для LLM."""
        system = RAG_SYSTEM_PROMPT.format(context=context)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]

    def extract_sources(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Извлекает уникальные источники (урок, страница) из чанков."""
        seen = set()
        sources = []
        for chunk in chunks:
            meta = chunk["metadata"]
            lesson = meta.get("lesson")
            page = meta.get("page")
            if lesson is not None and page is not None:
                key = (lesson, page)
                if key not in seen:
                    seen.add(key)
                    sources.append({"lesson": lesson, "page": page})
        return sources

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        filter_lesson: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> RAGResult:
        """Полный цикл: вопрос → поиск → генерация → ответ с источниками."""
        chunks = self.retrieve(query, top_k=top_k, filter_lesson=filter_lesson)
        context = self.build_context(chunks)
        messages = self.build_prompt(query, context)
        llm_resp = self.llm_client.chat(
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
        )
        sources = self.extract_sources(chunks)
        return RAGResult(
            answer=llm_resp.text.strip(),
            sources=sources,
            query=query,
            retrieved_chunks=chunks,
            llm_response=llm_resp,
            config=self.config,
        )

    def ask_stream(
        self,
        query: str,
        top_k: int | None = None,
        filter_lesson: int | None = None,
    ):
        """Streaming версия (если LLM поддерживает). Пока не реализовано."""
        raise NotImplementedError("Streaming пока не поддерживается")


def create_rag_pipeline(
    vector_store_config: dict[str, Any] | None = None,
    llm_config: dict[str, Any] | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    rag_config: RAGConfig | None = None,
) -> RAGPipeline:
    """Фабрика для создания готового RAG пайплайна."""
    vs = create_vector_store(vector_store_config)
    llm = create_llm_client(llm_config)
    return RAGPipeline(
        vector_store=vs,
        llm_client=llm,
        embedding_model_name=embedding_model,
        config=rag_config,
    )


if __name__ == "__main__":
    print("Инициализация RAG пайплайна...")
    try:
        pipeline = create_rag_pipeline()
        print(f"VectorStore: {type(pipeline.vector_store).__name__}")
        print(f"LLM: {type(pipeline.llm_client).__name__}")
        print(f"Embedding model: {pipeline.config.embedding_model}")
        print(f"Векторов в хранилище: {pipeline.vector_store.count()}")

        print("\n--- Тест Retrieval ---")
        chunks = pipeline.retrieve("функция потерь", top_k=3)
        for i, c in enumerate(chunks, 1):
            meta = c["metadata"]
            print(f"  {i}. score={c['score']:.4f} | Урок {meta.get('lesson')} | стр. {meta.get('page')}")

        print("\n--- Тест полного пайплайна (требует LLM ключ) ---")
        result = pipeline.ask("Что такое функция потерь?")
        print(f"Ответ: {result.answer}")
        print(f"Источники: {result.sources}")

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()