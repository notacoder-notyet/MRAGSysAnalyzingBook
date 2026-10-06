# MRAGSysAnalyzingBook — Полное руководство разработчика

> Пошаговое описание архитектуры, компонентов и решений проекта.

---

## Оглавление

1. [Общая архитектура](#1-общая-архитектура)
2. [Этап 1: Парсинг PDF (`pars_pdf.py`)](#2-этап-1-парсинг-pdf-pars_pdfpy)
3. [Этап 2: Эмбеддинги (`embeddings.py`)](#3-этап-2-эмбеддинги-embeddingspy)
4. [Этап 3: Векторное хранилище (`vector_store.py`)](#4-этап-3-векторное-хранилище-vector_storepy)
5. [Этап 4: Индексация (`index_to_vector_store.py`)](#5-этап-4-индексация-index_to_vector_storepy)
6. [Этап 5: LLM абстракция (`llm.py`)](#6-этап-5-llm-абстракция-llmpy)
7. [Этап 6: RAG пайплайн (`rag_pipeline.py`)](#7-этап-6-rag-пайплайн-rag_pipelinepy)
8. [Веб-приложение (`webapp/`)](#8-веб-приложение-webapp)
9. [Оценка качества (`gold_set.py`)](#9-оценка-качества-gold_setpy)
10. [Конфигурация и запуск](#10-конфигурация-и-запуск)
11. [Полезные ссылки в коде](#11-полезные-ссылки-в-коде)
12. [Чек-лист качества кода](#12-чек-лист-качества-кода)

---

## 1. Общая архитектура

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────────┐
│   PDF файл  │───▶│  Парсинг     │───▶│  Чанкинг     │───▶│   Эмбеддинги   │
│  (lesson_6) │    │  (страницы)  │    │  (500/100)   │    │  (384-dim)     │
└─────────────┘    └──────────────┘    └──────────────┘    └───────┬────────┘
                                                                   │
                    ┌──────────────────────────────────────────────┘
                    ▼
           ┌──────────────────┐    ┌──────────────┐    ┌────────────────┐
           │  Vector Store    │◀───│  Индексация  │    │   Retrieval    │
           │  (Chroma/Qdrant) │    │  (.npy/CSV)  │    │  (top-k + filter)│
           └────────┬─────────┘    └──────────────┘    └───────┬────────┘
                    │                                        │
                    ▼                                        ▼
           ┌──────────────────┐    ┌──────────────┐    ┌────────────────┐
           │   LLM Client     │◀───│   Prompt     │    │  Generation    │
           │ (OpenAI/Ollama)  │    │  (citation)  │    │  + sources     │
           └──────────────────┘    └──────────────┘    └────────────────┘
```

### Ключевые принципы

| Принцип | Реализация |
|---------|------------|
| **Citation-first** | Каждый чанк несёт `lesson`, `page`, `pdf_name`, `chunk_id` |
| **Modularity** | Каждый этап — отдельный модуль с чётким интерфейсом |
| **Swappable backends** | VectorStore (Chroma/Qdrant), LLM (OpenAI/Ollama/Mock) через фабрики |
| **Local-first** | Эмбеддинги считаются локально (sentence-transformers), никаких внешних API на этапе индексации |
---

## 2. Этап 1: Парсинг PDF (`pars_pdf.py`)

### 2.1 `extract_lesson_number(pdf_name: str) -> int | None`

**Что делает:** Извлекает номер урока из имени файла (например, `lesson_6.pdf` → `6`).

**Почему используется:** Для citation-метаданных нужен `lesson` номер. Регулярное выражение гибко парсит варианты имен.

**Код:** строки 23–37

**Зависимости:** `re` (stdlib)

---

### 2.2 `extract_text_from_pdf(pdf_path: str) -> list[dict[str, Any]]`

**Что делает:** Постранично извлекает текст из PDF через `pdfplumber`. Возвращает список словарей:
```python
{
    "pdf_name": "lesson_6.pdf",
    "lesson": 6,
    "page": 1,
    "text": "Текст страницы..."
}
```

**Почему `pdfplumber`:**
- Лучше всего извлекает текст из сложных PDF (колонки, таблицы, заголовки)
- Даёт доступ к номеру страницы
- Не требует OCR для цифровых PDF

**Код:** строки 40–69

**Возвращает:** `list[dict]` — по одному элементу на страницу (нумерация с 1)

---

### 2.3 `extract_tables_from_pdf(pdf_path: str) -> list[pd.DataFrame]`

**Что делает:** Извлекает все таблицы из PDF, конвертирует в `pandas.DataFrame` (первая строка = заголовки).

**Почему:** Таблицы в учебниках содержат структурированные данные (гиперпараметры, формулы, сравнения). Сохраняем как CSV для последующего использования.

**Код:** строки 72–95

**Возвращает:** `list[pd.DataFrame]`

---

### 2.4 `build_pages_dataframe(pages: list[dict]) -> pd.DataFrame`

**Что делает:** Строит DataFrame со страницами + метрики: `char_count`, `word_count`.

**Зачем:** EDA (разведочный анализ) — понять распределение длин страниц, найти пустые, подобрать `CHUNK_SIZE`.

**Код:** строки 109–136

---

### 2.5 `summarize_pages_dataset(pages_df: pd.DataFrame) -> dict[str, Any]`

**Что делает:** Сводка по датасету: всего страниц/символов/слов, min/max/mean длины, % пустых страниц.

**Код:** строки 138–155

---

### 2.6 `chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]`

**Что делает:** Делит текст на чанки по символам с перекрытием (sliding window).

**Алгоритм:**
```
step = chunk_size - chunk_overlap
start = 0
while start < len(text):
    end = min(start + chunk_size, len(text))
    chunks.append(text[start:end])
    start += step
```

**Почему символы, а не токены:**
- Проще, быстрее, не требует токенизатора
- Достаточно для baseline (позже можно заменить на токен-чанкинг) #TODO 

**Валидация:** проверяет `chunk_size > 0`, `0 <= overlap < chunk_size`

**Код:** строки 158–195

---

### 2.7 `chunk_pages(pages: list[dict], chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict[str, Any]]`

**Что делает:** Применяет `chunk_text` к каждой странице, сохраняя citation-метаданные:
```python
{
    "chunk_id": 0,          # глобальный ID
    "pdf_name": "...",
    "lesson": 6,
    "page": 1,
    "chunk_index": 0,       # индекс внутри страницы
    "text": "..."
}
```

**Важно:** `chunk_id` — глобальный уникальный ID (инкрементится), `chunk_index` — локальный на странице.

**Код:** строки 198–239

---

## 3. Этап 2: Эмбеддинги (`embeddings.py`)

### 3.1 Константы

```python
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

**Почему эта модель:**
- Multilingual (RU/EN)
- Лёгкая: 384 dim, ~120 MB, быстро на CPU
- Хорошее качество для baseline
- Upgrade: `BAAI/bge-m3` (1024 dim, лучше качество, медленнее) #TODO переключаемо, для запуска на нормальном железе

---

### 3.2 `load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer`

**Что делает:** Загружает модель (с кэшированием весов в `~/.cache/huggingface`).

**Код:** строки 21–34

---

### 3.3 `embed_texts(texts: Sequence[str], model: SentenceTransformer | None = None, model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 32, normalize: bool = True, show_progress: bool = True) -> np.ndarray`

**Что делает:** Базовая функция эмбеддинга списка текстов.

**Параметры:**
- `normalize=True` — L2-нормализация (cosine similarity = dot product)
- `batch_size=32` — баланс памяти/скорости
- Пустые строки заменяются на `" "` (чтобы не падал encode)

**Возвращает:** `np.ndarray` shape `(n_texts, dim)` dtype `float32`

**Код:** строки 37–72

---

### 3.4 `embed_chunks(chunks: list[dict], model: SentenceTransformer | None = None, model_name: str = DEFAULT_EMBEDDING_MODEL, text_key: str = "text", batch_size: int = 32, normalize: bool = True, show_progress: bool = True) -> tuple[np.ndarray, pd.DataFrame]`

**Что делает:** Основная функция — эмбеддит чанки, возвращает `(embeddings, meta_df)`.

**meta_df колонки:** `chunk_id`, `pdf_name`, `lesson`, `page`, `chunk_index`, `text` (без векторов — они в `.npy`)

**Код:** строки 75–127

---

### 3.5 `save_embeddings(embeddings: np.ndarray, meta_df: pd.DataFrame, embeddings_path: str, meta_path: str) -> None`

**Что делает:** Сохраняет `.npy` (векторы) + `.csv` (метаданные). Создаёт директории. Проверяет соответствие длин.

**Код:** строки 129–153

---

### 3.6 `load_embeddings(embeddings_path: str, meta_path: str) -> tuple[np.ndarray, pd.DataFrame]`

**Что делает:** Загружает обратно, проверяет согласованность.

**Код:** строки 156–176

---

### 3.7 `cosine_topk(query_vector: np.ndarray, embeddings: np.ndarray, meta_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame`

**Что делает:** Baseline retriever — топ-k по косинусному сходству (для нормализованных векторов = dot product).

**Алгоритм:**
```python
scores = embeddings @ query_vector  # (n,) — cosine similarity
top_idx = argpartition(-scores, k)[:k]
top_idx = top_idx[argsort(-scores[top_idx])]  # сортировка по убыванию
```

**Возвращает:** DataFrame с колонкой `score` + метаданными чанков.

**Код:** строки 179–209
---

## 4. Этап 3: Векторное хранилище (`vector_store.py`)

### 4.1 Абстракция `VectorStore` (ABC)

**Интерфейс:**
```python
add(embeddings, metadatas, ids) -> None
search(query_vector, top_k, filter_dict) -> list[dict]
delete(ids) -> None
count() -> int
get_by_ids(ids) -> list[dict]
clear() -> None
```

**Зачем абстракция:** Переключение Chroma ↔ Qdrant одной строчкой в конфиге, без изменения бизнес-логики.

**Код:** строки 18–64

---

### 4.2 `ChromaStore(VectorStore)`

**Реализация:** Локальная ChromaDB (PersistentClient), cosine space.

**Особенности:**
- Файловая БД в `data/chroma/`
- Хранит: embeddings + metadatas + documents (text)
- Фильтрация через `where` (metadata filtering)

**Код:** строки 67–165

---

### 4.3 `QdrantStore(VectorStore)`

**Реализация:** Qdrant клиент (HTTP/gRPC), можно локально в Docker или облако.

**Особенности:**
- Payload = metadata + text
- Фильтрация через `Filter` + `FieldCondition`
- Требует запущенный Qdrant (`localhost:6333`)

**Код:** строки 168–280

---

### 4.4 `create_vector_store(config: dict | None = None) -> VectorStore`

**Фабрика:** Читает `vector_store.yaml` или принимает dict. Возвращает экземпляр хранилища.

**Пример конфига:**
```yaml
type: chroma
chroma:
  persist_dir: data/chroma
  collection: lessons
qdrant:
  url: http://localhost:6333
  collection: lessons
  vector_size: 384
```

**Код:** строки 283–321
---

## 5. Этап 4: Индексация (`index_to_vector_store.py`)

### Назначение
Скрипт для загрузки существующих `.npy` + `.csv` в векторное хранилище.

### Аргументы CLI
```bash
python index_to_vector_store.py           # Chroma (default)
python index_to_vector_store.py --qdrant  # Qdrant
python index_to_vector_store.py --clear   # очистить перед загрузкой
```

### Логика
1. Загружает конфиг (переключает на Qdrant если `--qdrant`)
2. Читает `chunk_embeddings.npy` + `chunk_meta.csv`
3. Подготавливает метаданные для VectorStore (text отдельно, остальное в metadata)
4. Индексирует батчами по 100
5. Тестовый поиск для проверки

**Код:** весь файл

---

## 6. Этап 5: LLM абстракция (`llm.py`)

### 6.1 `LLMResponse` (dataclass)
```python
text: str
model: str
usage: dict[str, int] | None
raw: Any
```

---

### 6.2 `LLMClient` (ABC)
```python
chat(messages, temperature, max_tokens) -> LLMResponse
complete(prompt, temperature, max_tokens) -> LLMResponse
```

---

### 6.3 `OpenAIClient(LLMClient)`

**Что делает:** Обёртка над `openai.OpenAI` (совместим с vLLM, LM Studio, Azure).

**Параметры:** `model`, `api_key`, `base_url`, `timeout`

**Код:** строки 53–109

---

### 6.4 `OllamaClient(LLMClient)`

**Что делает:** HTTP клиент для локального Ollama (`/api/chat`, `/api/generate`).

**Параметры:** `model`, `base_url` (default `http://localhost:11434`), `timeout`

**Код:** строки 112–193

---

### 6.5 `MockLLMClient(LLMClient)`

**Что делает:** Заглушка для тестирования retrieval без LLM. Возвращает mock-ответ с префиксом `[MOCK]`.

**Код:** строки 249–281

---

### 6.6 `create_llm_client(config: dict | None = None) -> LLMClient`

**Фабрика:** Авто-определение бэкенда:
1. `config.type` если задан
2. `OPENAI_API_KEY` env → OpenAI
3. `OLLAMA_BASE_URL` env → Ollama
4. Иначе → MockLLMClient

**Код:** строки 196–246

---

### 6.7 `OpenRouterClient(OpenAIClient)`

**Что делает:** клиент OpenRouter — OpenAI-совместимый API, поэтому
наследуется от `OpenAIClient`, меняя только `base_url` и заголовки атрибуции.

**Зачем:** на OpenRouter есть бесплатные модели (`:free`), что важно для пет-проекта.

**Код:** `llm.py`, класс `OpenRouterClient`

---

### 6.8 `FallbackLLMClient` — перебор моделей при rate-limit

**Что делает:** держит список LLM-клиентов и при ошибке переключается на следующий.

**Зачем:** бесплатные модели жёстко лимитированы (429). Без этого каждый второй
вопрос падал бы с ошибкой.

**Логика `_is_retryable`:**

| Код | Повторяем? | Почему |
|-----|-----------|--------|
| 429 | ✅ | rate-limit — главная причина |
| 500/502/503/504/529 | ✅ | перегрузка провайдера |
| 404 | ✅ | «модель больше не бесплатна» — список меняется |
| 400/401/402 | ❌ | наша ошибка: запрос, ключ или баланс |

**Sticky-поведение:** после успеха клиент «прилипает» к удачной модели и не
тратит лимиты на перебор при следующих запросах.

---

## 7. Этап 6: RAG пайплайн (`rag_pipeline.py`)

### 7.1 `RAGConfig` (dataclass)

```python
top_k: int = DEFAULT_TOP_K              # 8
filter_lesson: int | None = None
temperature: float = DEFAULT_LLM_TEMPERATURE
max_tokens: int | None = DEFAULT_LLM_MAX_TOKENS
embedding_model: str = DEFAULT_EMBEDDING_MODEL
```

Все значения по умолчанию берутся из `config.py` — единого источника истины.

---

### 7.2 `RAGResult` (dataclass)

```python
answer: str                       # текст ответа
sources: list[dict]               # [{"lesson": 6, "page": 3}, ...]
query: str
retrieved_chunks: list[dict]      # сырые чанки со score
llm_response: LLMResponse | None
config: RAGConfig | None
```

---

### 7.3 `RAGPipeline.retrieve(query, top_k, filter_lesson) -> list[dict]`

1. Эмбеддит вопрос через `embed_texts` (та же модель, что при индексации — критично!)
2. Ищет в VectorStore с опциональным фильтром по уроку
3. Возвращает чанки со score и метаданными

---

### 7.4 `RAGPipeline.build_context(chunks) -> str`

Собирает контекст с явными заголовками источников:

```
[Источник 1: Урок 6, Страница 3]
ФУНКЦИЯ ПОТЕРЬ. Функция потерь = Линейка ошибки...

---
[Источник 2: Урок 24, Страница 16]
Продвинутые функции потерь: Triplet Loss и Focal Loss...
```

Заголовки нужны, чтобы модель видела, откуда взят текст, и могла цитировать.

---

### 7.5 `RAG_SYSTEM_PROMPT` (в `config.py`)

**Ключевые правила промпта:**
1. Отвечать **только** по переданным чанкам
2. Если данных нет — честно сказать «в материалах нет ответа»
3. В конце указать источники в формате `(Урок X, Страница Y)`
4. Не выдумывать факты, не использовать внешние знания
5. Отвечать на русском

---

### 7.6 `RAGPipeline.ask(...) -> RAGResult`

Главный метод: `retrieve → build_context → build_prompt → llm.chat → extract_sources`.

`extract_sources` дедуплицирует пары `(lesson, page)`, чтобы в ответе не было
повторов одной и той же страницы.

---

### 7.7 `create_rag_pipeline(...)` — фабрика

Собирает пайплайн из конфигов VectorStore и LLM.

---

## 8. Веб-приложение (`webapp/`)

### 8.1 Слой данных — `database.py` + `models.py`

**Что делает:** SQLAlchemy поверх SQLite. `init_db()` создаёт таблицы при старте.

**Схема:**

```
User ──< Chat ──< Message ──< Source
                                  └─ lesson, page, score
```

**Почему SQLite:** файловая БД без сервера — для пет-проекта достаточно,
а SQLAlchemy позволяет позже переехать на PostgreSQL сменой одной строки.

**Каскады:** `cascade="all, delete-orphan"` — удаление чата чистит его сообщения,
удаление сообщения — его источники.

**`get_db()`** — FastAPI-зависимость: выдаёт сессию и гарантированно закрывает её
после запроса (иначе соединения утекали бы).

---

### 8.2 Валидация — `schemas.py`

**Что делает:** Pydantic-модели для запросов и ответов.

**Зачем:** ограничения (длина пароля, формат логина) проверяются до бизнес-логики,
а `ConfigDict(from_attributes=True)` позволяет отдавать ORM-объекты напрямую.

**Валидатор логина:** только буквы, цифры, `-` и `_` — защита от подстановки в URL.

---

### 8.3 Безопасность — `security.py`

| Функция | Что делает |
|---------|-----------|
| `hash_password` | bcrypt-хеш с солью внутри |
| `verify_password` | сверка пароля с хешем |
| `create_access_token` | выпуск JWT (HS256, 7 дней) |
| `decode_access_token` | проверка подписи и срока |
| `get_current_user` | FastAPI-зависимость: достаёт пользователя из токена |

**Важная деталь:** bcrypt работает только с первыми 72 байтами пароля —
обрезаем явно, иначе длинный пароль вызывает ошибку.

**Защита от чужого доступа:** все операции с чатами идут через `_get_owned_chat()`,
который проверяет `user_id`. Без этого любой авторизованный пользователь
мог бы читать чужие диалоги по id.

---

### 8.4 Сервис RAG — `rag_service.py`

**Что делает:** держит единственный экземпляр `RAGPipeline` на всё приложение.

**Зачем:** загрузка sentence-transformers занимает секунды. Если создавать пайплайн
на каждый HTTP-запрос, ответ будет приходить через 5–10 секунд вместо секунды.

**Двойная проверка под `threading.Lock`:** FastAPI может принять два запроса
одновременно, и без блокировки модель загрузилась бы дважды.

**Фильтр `MIN_ANSWER_SCORE`:** источники со score ниже 0.25 не показываются —
иначе пользователь получил бы ссылку на нерелевантную страницу.

---

### 8.5 Роутеры — `routers/`

**`auth.py`** — регистрация, вход, профиль. Проверяет уникальность логина (409).

**`chat.py`** — CRUD чатов и главный эндпоинт `POST /api/chat/ask`:
1. Находит или создаёт чат (заголовок = первые 60 символов вопроса)
2. Сохраняет вопрос пользователя
3. Вызывает RAG
4. Сохраняет ответ и источники
5. Поднимает чат наверх списка (`updated_at`)

**`lessons.py`** — список уроков и отдача PDF. Реестр уроков кешируется
(`@lru_cache`), т.к. состав папки в рамках работы не меняется.

---

### 8.6 Фронтенд — `static/`

| Файл | Роль |
|------|------|
| `index.html` | Разметка: модалка авторизации, чат, панель PDF, презентация |
| `style.css` | Тема VS Code Dark+ (CSS-переменные для акцентов) |
| `app.js` | Логика: auth, чаты, PDF.js, аккордеон презентации |

**Почему без сборки:** ванильный JS + CDN для PDF.js — не нужен Node.js,
проект запускается одной командой.

**Связка «ответ → страница»:**
```
POST /api/chat/ask → sources
   → чипы «Урок 6 · стр. 3»
   → openPdfPage(6, 3)
   → PDF.js грузит /api/lessons/6/pdf
   → рендер canvas + навигация ‹ ›
```

---

## 9. Оценка качества (`gold_set.py`, `retrieval_report.py`)

### 9.1 Зачем нужен gold-набор

Чтобы измерять улучшения, а не «казаться лучше». Без него любая правка
(смена модели, размера чанка) проверяется на глаз.

### 9.2 Ключевое решение: единица истины — `(урок, страница)`

На странице бывает до 7 чанков. Если размечать по чанкам, Hit@k падает
на 30–50% просто из-за разбиения текста, а не из-за качества поиска.

Формат `6:3|6:5` разрешает любой из вариантов-страниц.

### 9.3 Защита от перекоса выборки

- `LESSON_QUARANTINE = {39}` — урок 39 это 11% корпуса (38 МБ конспекта),
  при случайном семпле он занял бы треть набора
- `PER_LESSON_SHARE = 0.08` — не более 8% вопросов на один урок
- `sample(frac=1, random_state=seed)` — перемешивание страниц, иначе
  `groupby().head()` всегда брал бы первые страницы лекций

### 9.4 Защита от завышения метрик

Промпт генерации явно требует:
- перефразировать, а не копировать текст чанка
- не упоминать номера страниц и уроков в тексте вопроса

Без этого вопрос дословно совпал бы с чанком и дал бы Hit@1 ≈ 100%
вне зависимости от качества модели.

### 9.5 Команды

```bash
python gold_set.py generate --n 25   # LLM генерирует черновик
python gold_set.py validate          # проверка структуры
python gold_set.py approve           # отметить verified после ревью
python gold_set.py eval              # Hit@k + MRR
python retrieval_report.py           # диагностика: какой провал и почему
```

### 9.6 `retrieval_report.py` — диагностика, а не просто метрики

Считает `garble_ratio` — долю «мусорных» токенов на странице. Это оказалось
главным предиктором провала: **все 20 провальных вопросов (100%) указывали
на страницы с мусорным текстом** (средняя мусорность 0.291 против 0.200).

Вывод, который это дало: узкое место — парсинг таблиц и схем, а не модель
эмбеддингов. Приоритеты в `TODO.md` пересобраны по этому факту.

---

## 10. Конфигурация и запуск

### 10.1 Файлы конфигурации

| Файл | Назначение |
|------|-----------|
| `config.py` | **Единый источник констант** (пути, модели, параметры, промпт) |
| `vector_store.yaml` | Тип векторной БД и её параметры |
| `.env` | Секреты: `OPENROUTER_API_KEY`, `JWT_SECRET` (в `.gitignore`) |
| `.env.example` | Шаблон для `.env` — без секретов, попадает в git |

### 10.2 Полный цикл

```bash
# 1. Зависимости
pip install -r requirements.txt

# 2. Секреты
cp .env.example .env          # вписать OPENROUTER_API_KEY

# 3. Данные
python download_pdfs.py       # 59 PDF с Яндекс Диска → data/raw/
python pars_pdf.py            # 973 страницы → 1034 чанка
python embeddings.py          # (1034, 384)
python index_to_vector_store.py --clear

# 4. Оценка (опционально)
python gold_set.py eval
python retrieval_report.py

# 5. Приложение
python -m webapp.main         # http://localhost:8000

# 6. Тесты
python test_full_pipeline.py
```

### 10.3 Как переключать компоненты

**Векторная БД:** `vector_store.yaml` → `type: chroma | qdrant`

**LLM:** определяется по env автоматически —
`OPENROUTER_API_KEY` → OpenRouter, `OPENAI_API_KEY` → OpenAI,
`OLLAMA_BASE_URL` → Ollama, ничего нет → Mock.

**Модель эмбеддингов:** `DEFAULT_EMBEDDING_MODEL` в `config.py`.
При смене обязательно поменять `EMBEDDING_DIM` и переиндексировать
(`index_to_vector_store.py --clear`), иначе размерности не совпадут.

### 10.4 Частые грабли

| Симптом | Причина |
|---------|---------|
| LLM отвечает «нет ответа», хотя чанки найдены | Тексты не попали в `documents` при индексации |
| `401 User not found` от OpenRouter | Ключ provisioning вместо inference |
| Пустые ответы при рабочем поиске | Модель-роутер отдала запрос классификатору модерации |
| Ошибка размерности при поиске | Сменили модель эмбеддингов без переиндексации |

---

## 11. Полезные ссылки в коде

| Компонент | Файл |
|-----------|------|
| Единые константы | `config.py` |
| Парсинг и чанкинг | `pars_pdf.py` — `extract_text_from_pdf`, `batch_extract_chunks` |
| Эмбеддинги | `embeddings.py` — `embed_chunks`, `cosine_topk` |
| Векторное хранилище | `vector_store.py` — `VectorStore`, `ChromaStore`, `QdrantStore` |
| Индексация | `index_to_vector_store.py` |
| LLM-клиенты | `llm.py` — `OpenRouterClient`, `FallbackLLMClient` |
| RAG | `rag_pipeline.py` — `RAGPipeline.ask` |
| Веб-API | `webapp/main.py`, `webapp/routers/*` |
| Модели БД | `webapp/models.py` |
| Безопасность | `webapp/security.py` |
| Фронтенд | `webapp/static/app.js` |
| Gold-набор | `gold_set.py` |
| Диагностика поиска | `retrieval_report.py` |
| Скачивание PDF | `download_pdfs.py` |
| Сквозной тест | `test_full_pipeline.py` |

---

## 12. Чек-лист качества кода

Проверено на текущей версии:

- [x] `pyflakes` — неиспользуемых импортов и переменных нет
- [x] Все модули компилируются (`py_compile`)
- [x] JS проходит разбор синтаксиса
- [x] Type hints и docstring у всех публичных функций
- [x] Нет мёртвого кода (удалены `main.py`, `utils.py`, `demo_rag.py`,
      `test_vector_store.py`, `quick_chat()`)
- [x] API покрыт сквозным тестом (13 сценариев, включая 401/409)
- [x] CSS-скобки сбалансированы, JS-синтаксис валиден
- [x] Секреты — только в `.env`, он в `.gitignore`

Что можно улучшить дальше:

- [ ] `pytest` вместо скриптовых тестов
- [ ] `mypy --strict` по всем модулям
- [ ] Logging вместо `print` в скриптах пайплайна
- [ ] Alembic для миграций схемы БД
- [ ] Переменная `JWT_SECRET` обязательна из env (сейчас есть dev-дефолт)

