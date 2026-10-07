# MRAGSysAnalyzingBook

[![CI](https://github.com/notacoder-notyet/MRAGSysAnalyzingBook/actions/workflows/ci.yml/badge.svg)](https://github.com/notacoder-notyet/MRAGSysAnalyzingBook/actions/workflows/ci.yml)

Пет-проект: **Multimodal RAG** по учебным PDF (59 уроков, ~1000 страниц) с
цитированием урока и страницы и полноценным веб-интерфейсом (чат + просмотр
презентаций).

Ключевая идея — **проверяемость ответа**. Система не просто генерирует текст, а
ссылается на конкретный слайд PDF. Чтобы модель не выдумывала номера страниц,
она ссылается на фрагменты по индексу (`[1]`, `[2]`), а реальные
«Урок N, Страница M» подставляются из метаданных найденных чанков
(`answer_utils.substitute_refs`) — поэтому ссылка не может «поплыть».

Ориентир по продукту и архитектуре — [`idea.MD`](idea.MD).

## 🚀 Быстрый запуск

```bash
# 1. Зависимости (полный набор, включая torch)
make install                 # = pip install -r requirements.txt

# 2. Секреты
cp .env.example .env         # вписать OPENROUTER_API_KEY (есть бесплатные модели)

# 3. Данные (один раз; ~15–30 мин из-за эмбеддингов)
make download                # 59 PDF → data/raw/
make parse                   # парсинг + чанкинг
make index                   # эмбеддинги + индексация в Chroma

# 4. Запуск
make run                     # → http://localhost:8000
```

Альтернатива — **Docker** (после шага 3, чтобы в `./data` уже лежал индекс):

```bash
docker compose up --build    # → http://localhost:8000
```

> `make help` покажет все команды (download / parse / index / run / test / lint / docker).

### Что умеет интерфейс

| Возможность | Описание |
|-------------|----------|
| **Чат** | Вопрос → ответ; ссылки-источники `[N]` становятся кликабельными «(Урок X, Страница Y)» |
| **Панель PDF** | Страница-источник открывается рядом; листание по страницам/урокам, полноэкранный режим |
| **Источники** | Чипы под ответом: по одной (самой релевантной) странице на урок — без дублей |
| **Читаемый ответ** | Постобработка убирает LaTeX, Markdown и служебные сноски `【…】` |
| **Авторизация** | Регистрация/вход, JWT, пароли хешируются bcrypt |
| **История чатов** | Хранится на сервере в SQLite, доступна с любого устройства |
| **Презентация проекта** | 7 этапов с раскрывающимися деталями |

Тема оформления — VS Code Dark+: тёмный фон, зелёные акценты и подсветка
синим / оранжевым / розовым.

> **Офлайн-режим.** С облачным OpenRouter интернет нужен. Полностью локально
> система работает через **Ollama** (`OLLAMA_BASE_URL` в `.env`): LLM,
> эмбеддинги и векторное хранилище — всё локальное.

## 🗂️ Структура проекта

```
├── config.py                 # единый источник констант (single source of truth)
├── answer_utils.py           # чистые функции ответа: sanitize / dedupe_sources / substitute_refs
├── frontend_check.py         # структурная проверка app.js / style.css (без тяжёлых зависимостей)
├── text_clean.py             # очистка текста страниц (Unicode/эмодзи/мусорные строки) + garble_ratio
├── pars_pdf.py               # парсинг PDF → страницы → чанки
├── embeddings.py             # эмбеддинги (multilingual-e5-base)
├── vector_store.py           # абстракция Chroma / Qdrant
├── index_to_vector_store.py  # индексация чанков
├── llm.py                    # LLM: OpenAI / OpenRouter / Ollama / Mock + fallback
├── rag_pipeline.py           # retrieve → prompt([N]-ссылки) → generate
├── gold_set.py               # gold-набор и метрики (Hit@k, MRR)
├── retrieval_report.py       # диагностика провалов retrieval
├── download_pdfs.py          # скачивание PDF с Яндекс Диска
├── test_full_pipeline.py     # интеграционный сквозной тест (нужны данные)
├── webapp/                   # FastAPI-приложение (см. «Архитектура», ниже)
├── tests/                    # unit-тесты на pytest (без torch)
├── Dockerfile / docker-compose.yml
├── Makefile                  # удобные команды
└── .github/workflows/ci.yml  # CI: ruff + black + pytest + py_compile + JS
```

## Что уже есть

- ✅ Постраничный парсинг PDF (`pdf_name`, `lesson`, `page`, `text`) — `pars_pdf.py`
- ✅ Скачивание 59 PDF с Яндекс Диска — `download_pdfs.py`
- ✅ Извлечение таблиц через `pdfplumber` → DataFrame/CSV
- ✅ Лёгкая EDA по датасету страниц (`build_pages_dataframe`, `summarize_pages_dataset`)
- ✅ Чанкинг с overlap, размер подобран по EDA — `chunk_pages`
- ✅ Эмбеддинги чанков (`embeddings.py`, `intfloat/multilingual-e5-base`, 768-dim)
- ✅ **Векторное хранилище** — Chroma (default) / Qdrant с абстракцией `VectorStore`
- ✅ **Индексация** — `index_to_vector_store.py` загружает текст + метаданные
- ✅ **LLM абстракция** — OpenAI / OpenRouter / Ollama / Mock через фабрику
- ✅ **RAG пайплайн** — `rag_pipeline.py`: retrieve → prompt → generate → answer + sources
- ✅ **Веб-приложение** — `webapp/`: FastAPI + JWT + SQLite + чат + PDF-вьювер
- ✅ **Оценка качества** — gold-набор (63 вопроса), Hit@k/MRR, диагностика провалов

**Надёжность ответа и инженерное качество**

- ✅ **Цитаты по индексу** — модель ссылается на фрагменты `[N]`, реальные урок/страница подставляются из метаданных (`answer_utils.substitute_refs`) — ссылка не выдумывается
- ✅ **Читаемость ответа** — очистка LaTeX / Markdown / сносок `【…】` (`answer_utils.sanitize_answer`), применяется и к старым сообщениям при отдаче
- ✅ **Дедуп источников** — одна (самая релевантная) страница на урок (`answer_utils.dedupe_sources`)
- ✅ **Unit-тесты** — `tests/` на pytest (33 теста), без тяжёлых ML-зависимостей
- ✅ **CI** — GitHub Actions: `ruff` + `black --check` + `pytest` + `py_compile` + `node --check` (`.github/workflows/ci.yml`)
- ✅ **Docker** — `Dockerfile` + `docker-compose.yml`, запуск одной командой
- ✅ **Makefile** — `make install/dev/parse/index/run/test/lint/docker-up`

## Стек на текущем этапе

**Данные и ML**
- Python, `pdfplumber`, `pandas`, `numpy`
- `sentence-transformers` (+ PyTorch) — локальные эмбеддинги

**Хранение и поиск**
- **`chromadb` / `qdrant-client`** — векторные БД (Chroma по умолчанию)
- **`sqlalchemy`** + SQLite — пользователи, чаты, сообщения

**LLM**
- `openai` / `httpx` — клиенты (OpenRouter, OpenAI, Ollama)

**Веб**
- **`fastapi`** + `uvicorn` — REST API
- **`pyjwt`** + `bcrypt` — авторизация
- Ванильный JS + **PDF.js** — фронтенд без сборки

## ⚙️ Конфигурация (`config.py`)

Все параметры проекта вынесены в **единый файл `config.py`** — источник истины (single source of truth).
Модули импортируют константы оттуда, поэтому изменение параметра в одном месте меняет его везде.

| Группа | Константы |
|--------|-----------|
| **PATHS** | `DATA_DIR`, `CHROMA_PERSIST_DIR`, `CHUNK_EMBEDDINGS_PATH`, `CHUNK_META_PATH`, `VECTOR_STORE_CONFIG_PATH` |
| **EMBEDDING** | `DEFAULT_EMBEDDING_MODEL`, `UPGRADE_EMBEDDING_MODEL`, `EMBEDDING_DIM`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_NORMALIZE` |
| **CHUNKING** | `DEFAULT_CHUNK_SIZE=1000`, `DEFAULT_CHUNK_OVERLAP=150` (подобрано по EDA, см. ниже) |
| **RETRIEVAL** | `DEFAULT_TOP_K=8`, `DEFAULT_FILTER_LESSON` |
| **VECTOR STORE** | `DEFAULT_COLLECTION_NAME`, `DEFAULT_VECTOR_STORE_TYPE`, `QDRANT_DEFAULT_URL`, `QDRANT_DEFAULT_VECTOR_SIZE`, `INDEX_BATCH_SIZE` |
| **LLM** | `DEFAULT_LLM_MODEL`, `DEFAULT_OLLAMA_MODEL`, `DEFAULT_OLLAMA_BASE_URL`, `DEFAULT_LLM_TEMPERATURE`, `DEFAULT_LLM_MAX_TOKENS`, `DEFAULT_LLM_TIMEOUT` |
| **PROMPT** | `RAG_SYSTEM_PROMPT` |
| **YANDEX DISK** | `YANDEX_DISK_PUBLIC_KEY`, `YANDEX_API_BASE`, `YANDEX_PAGE_LIMIT`, `YANDEX_TIMEOUT`, `YANDEX_MAX_RETRIES`, `YANDEX_SKIP_FILENAME_MARKERS` |
| **OPENROUTER** | `OPENROUTER_BASE_URL`, `OPENROUTER_FREE_MODELS`, `OPENROUTER_TIMEOUT`, `OPENROUTER_HTTP_REFERER` |

> ⚠️ **Тексты чанков хранятся в поле `documents`, а не в `metadata`.**
> `ChromaStore.add()` требует `documents=`, иначе `search()` вернёт пустой `text`,
> и LLM нечего будет цитировать (молча — без ошибки). `index_to_vector_store.py`
> передаёт их из колонки `text` в `meta_df`.

Пример использования:
```python
from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K
from embeddings import load_embedding_model
from pars_pdf import chunk_pages

model = load_embedding_model()            # использует DEFAULT_EMBEDDING_MODEL
chunks = chunk_pages(pages)               # использует DEFAULT_CHUNK_SIZE / OVERLAP
pipeline = create_rag_pipeline(rag_config=RAGConfig(top_k=DEFAULT_TOP_K))
```

Чтобы сменить модель эмбеддингов на `BAAI/bge-m3` — достаточно поменять
`DEFAULT_EMBEDDING_MODEL` и `EMBEDDING_DIM` в `config.py`.

---

## Baseline RAG (развёрнуто)

Цель baseline — **минимальный рабочий контур**, с которым потом сравниваем улучшения (таблицы, layout, reranker, другая модель эмбеддингов).

### Что baseline НЕ делает

- Не обучает классификатор / не делит PDF на train/val/test
- Не делает тяжёлый NLP-preprocessing (стемминг, стоп-слова)
- Не требует FastAPI на первом проходе
- Не обязан сразу тянуть `bge-m3` (тяжёлая модель) — для старта берём лёгкую multilingual-модель, `bge-m3` оставляем как upgrade

### Архитектура baseline

```
PDF
 └─ parse pages → {pdf_name, lesson, page, text}
      └─ chunk (size/overlap) → {chunk_id, lesson, page, text, ...}
           └─ embed(text) → vector[d]
                └─ (следующий шаг) vector DB + metadata filter
                     └─ retrieve top-k → LLM prompt + citation
```

### Офлайн-индексация (готово)

| Шаг | Что делаем | Артефакт | Статус |
|---|---|---|---|
| 1. Parse | Текст по страницам + citation-поля | `pdf_pages.csv` | ✅ |
| 2. EDA | Длины, пустые страницы → подбор `CHUNK_SIZE` | сводка + гистограмма | ✅ |
| 3. Chunk | Символьный чанкинг с overlap, метаданные наследуются | `pdf_chunks.csv` | ✅ |
| 4. Embed | Вектор на каждый чанк той же моделью, что и для query | `chunk_embeddings.npy` + `chunk_meta.csv` | ✅ |
| 5. Store | Векторы + `{lesson, page, pdf_name, chunk_id, text}` в **Chroma/Qdrant** | ✅ `data/chroma/` коллекция `lessons` | ✅ |

### Онлайн-запрос (готово)

1. Пользователь задаёт вопрос.
2. Тот же эмбеддер → вектор вопроса (`embed_texts`).
3. Semantic search top-k (например, k=5), опционально filter по `lesson` (`VectorStore.search`).
4. Промпт LLM (`RAG_SYSTEM_PROMPT`):
   - отвечай **только** по переданным чанкам;
   - в конце источники в формате `(Урок X, Страница Y)`;
   - если данных нет — «В предоставленных материалах нет ответа на этот вопрос».
5. Ответ пользователю + список источников (`RAGResult.sources`).

### Eval вместо train/val split

Корпус PDF целиком идёт в индекс. Для качества собираем **gold-набор** 20–50 вопросов:

```text
question | expected_lesson | expected_page | notes
```

Метрики baseline:

- **Retrieval Hit@k** — попала ли нужная страница/урок в top-k
- **Citation present** — модель указала `(Урок, Страница)`
- **Citation correct** — ссылка совпала с gold
- (опционально) качество ответа глазами / LLM-as-judge

Dev-вопросы — для подкрутки `CHUNK_*` / `TOP_K`; holdout — один раз для финальной оценки.

### Параметры baseline (стартовые)

| Параметр | Старт | Комментарий |
|---|---|---|
| `CHUNK_SIZE` | 500 символов | подкрутить после EDA |
| `CHUNK_OVERLAP` | 100 | ~20% overlap |
| Embedding model | `intfloat/multilingual-e5-base` (768-dim) | retrieval-модель для RU/EN |
| Upgrade model | `BAAI/bge-m3` | из idea.MD, тяжелее |
| `TOP_K` | 5 | для retriever |
| Normalize embeddings | `True` | cosine ≈ dot product |

### Критерий «baseline готов»

- [x] PDF → pages → chunks с `lesson`/`page`
- [x] Чанки эмбеддятся локально, векторы сохраняются (`data/chunk_embeddings.npy`, `data/chunk_meta.csv`)
- [ ] Есть ≥20 gold-вопросов
- [ ] Работает retrieve top-k + ответ LLM с citation
- [ ] Посчитан Hit@k на holdout

---

## Этапы

### Этап 1 — Парсинг и подготовка данных

1. Структура проекта
2. Текст + таблицы из PDF
3. Citation-метаданные
4. EDA
5. Чанкинг
6. Документация

### Этап 2 — Индексация (в работе)

1. ✅ Эмбеддинги чанков — **готово** (`data/chunk_embeddings.npy`, `data/chunk_meta.csv`)
2. ✅ **Векторная БД (Chroma / Qdrant) + метаданные** — **готово** (`vector_store.py`, `index_to_vector_store.py`)
3. Усиление парсинга таблиц/layout при необходимости

### Этап 3 — Online RAG

1. ✅ Retriever top-k (через VectorStore)
2. ✅ Промпт с citation (`rag_pipeline.py`)
3. ✅ Энд-ту-энд пайплайн: вопрос → эмбеддинг → retrieve → LLM → ответ с источниками (`RAGPipeline`)
4. ⏳ Eval на gold Q&A
5. ⏳ FastAPI — после рабочего ядра

## Быстрый старт

```bash
# зависимости
pip install -r requirements.txt

# 1) скачать PDF уроков с Яндекс Диска (59 файлов → data/raw/)
python download_pdfs.py --dry-run    # сначала посмотреть, что скачается
python download_pdfs.py              # скачать

# 2) парсинг всех PDF + чанкинг
python pars_pdf.py
# Артефакты: data/pdf_pages.csv, data/pdf_chunks.csv

# 3) эмбеддинги всех чанков (долго, лучше в фоне)
python embeddings.py
# Артефакты: data/chunk_embeddings.npy, data/chunk_meta.csv

# 4) индексация в векторную БД (Chroma по умолчанию)
python index_to_vector_store.py --clear

# или в Qdrant (требует запущенный Qdrant на localhost:6333)
python index_to_vector_store.py --qdrant --clear

# 5) запуск RAG пайплайна (требует OPENAI_API_KEY или Ollama)
python rag_pipeline.py

# проверка всего пайплайна на всей коллекции
python test_full_pipeline.py
```

> ⚠️ Шаги 2–3 на 973 страницах занимают заметное время на CPU.
> Запускайте в фоне: `nohup python embeddings.py > /tmp/emb.log 2>&1 &`

### Скачивание PDF с Яндекс Диска

Источник: публичный диск с папками `Lesson 1` … `Lesson 59`.
Скрипт рекурсивно обходит все папки и сохраняет PDF в `data/raw/lesson_<N>.pdf`.

**Правила именования и фильтрации:**
- Номер урока берётся из имени **папки** (имена самих PDF между уроками повторяются)
- Если в уроке несколько PDF → `lesson_39_1.pdf`, `lesson_39_2.pdf`
- PDF во вложенных папках тоже находятся (например `Lesson13/Презентация/`)
- Служебные файлы отсеиваются: `Домашнее_задание`, `homework`, `Постановка_реальной_бизнес-задачи`
  (маркеры в `config.py` → `YANDEX_SKIP_FILENAME_MARKERS`)

```bash
python download_pdfs.py --dry-run          # список файлов без скачивания
python download_pdfs.py                    # скачать все недостающие
python download_pdfs.py --limit 5          # только первые 5
python download_pdfs.py --overwrite        # перекачать заново
python download_pdfs.py --dest data/raw    # другая папка (по умолчанию data/raw)
```

**Текущее состояние:** 59 PDF (139 MB, 973 страницы) в `data/raw/`.
Урок 18 на диске отсутствует. Урок 39 содержит 2 PDF.

Или нотубук `pars_pdf.ipynb` для исследования (константы → парсинг → EDA → чанкинг → эмбеддинги). **Для продакшена используйте `.py` модули.**

### Текущие артефакты (после `python embeddings.py`)

| Файл | Описание |
|------|----------|
| `data/chunk_embeddings.npy` | Матрица эмбеддингов `(16, 384)` float32 |
| `data/chunk_meta.csv` | Метаданные 16 чанков: `chunk_id`, `pdf_name`, `lesson`, `page`, `chunk_index`, `text` |

Проверка:
```bash
python -c "
import numpy as np, pandas as pd
emb = np.load('data/chunk_embeddings.npy')
meta = pd.read_csv('data/chunk_meta.csv')
print('Embeddings:', emb.shape, emb.dtype)
print('Meta:', meta.shape)
print(meta[['chunk_id','lesson','page','chunk_index']].head())
"
```

Вывод эталонного запуска (встроенный тест retriever'а):
```
Top-3 по запросу: Что такое функция потерь?
      score  lesson  page                                               text
0  0.800766       6     3  ФУНКЦИЯ ПОТЕРЬ\nФункция потерь = Линейка ошибк...
1  0.619655       6     3  это итеративный поиск таких значений весов, пр...
2  0.521229       6     5  ОПТИМИЗАЦИЯ\nКак добраться до дна ямы?\nУ нас ...
```

## 🌐 Архитектура веб-приложения (`webapp/`)

```
webapp/
├── main.py            # FastAPI: CORS, lifespan, монтирование роутеров и статики
├── database.py        # SQLAlchemy engine + SessionLocal + get_db()
├── models.py          # ORM: User, Chat, Message, Source
├── schemas.py         # Pydantic: валидация + очистка ответа и дедуп источников (answer_utils)
├── security.py        # bcrypt (пароли) + PyJWT (токены) + get_current_user
├── rag_service.py     # синглтон RAG-пайплайна: ask() + санитайзер/ссылки [N]
├── routers/
│   ├── auth.py        # /api/auth: register, login, me
│   ├── chat.py        # /api/chat: CRUD чатов + /ask
│   └── lessons.py     # /api/lessons: список уроков + отдача PDF
└── static/
    ├── index.html     # разметка: модалка авторизации, чат, панель PDF
    ├── style.css      # тема VS Code Dark+
    └── app.js         # логика: auth, чаты, PDF.js, презентация
```

### API

| Метод | Путь | Назначение |
|-------|------|-----------|
| POST | `/api/auth/register` | Регистрация, возвращает JWT |
| POST | `/api/auth/login` | Вход, возвращает JWT |
| GET | `/api/auth/me` | Профиль текущего пользователя |
| GET | `/api/chat` | Список чатов пользователя |
| POST | `/api/chat` | Создать чат |
| GET | `/api/chat/{id}` | Чат с историей и источниками |
| PATCH | `/api/chat/{id}` | Переименовать |
| DELETE | `/api/chat/{id}` | Удалить (каскадно с сообщениями) |
| POST | `/api/chat/ask` | **Вопрос к RAG** → ответ + источники |
| GET | `/api/lessons` | Список уроков с числом страниц |
| GET | `/api/lessons/{n}/pdf` | PDF-файл урока (inline, для PDF.js) |
| GET | `/api/health` | Проверка живости |

### Схема данных

```
User ──< Chat ──< Message ──< Source
                                  └─ lesson, page, score
```

- `User` — логин + bcrypt-хеш пароля
- `Chat` — диалог, `updated_at` поднимает свежие наверх в истории
- `Message` — роль (`user`/`assistant`) и текст
- `Source` — цитаты ответа: урок, страница, косинусный score

### Безопасность

- Пароли хешируются **bcrypt** (соль внутри хеша), в открытом виде не хранятся
- **JWT** (HS256, 7 дней) в заголовке `Authorization: Bearer`
- Доступ к чужому чату по id отклоняется: `_get_owned_chat()` проверяет владельца
- Секрет JWT берётся из `JWT_SECRET` в `.env`, `.env` в `.gitignore`

### Как работает связка «ответ → страница PDF»

1. `POST /api/chat/ask` → RAG возвращает ответ и список `sources`
2. Backend подставляет вместо ссылок `[N]` реальные «(Урок X, Страница Y)» из
   метаданных чанков (`answer_utils.substitute_refs`) и чистит текст
   (`sanitize_answer`), источники дедуплицируются по уроку
3. Фронтенд рисует источники чипами `Урок 6 · стр. 3`, а упоминания в тексте
   ответа делает кликабельными (`app.js → linkifyAnswer`)
4. Клик по чипу или ссылке вызывает `openPdfPage(lesson, page)`
5. PDF.js грузит `/api/lessons/{n}/pdf` и рендерит страницу на canvas
   (подгонка по ширине и высоте; есть полноэкранный режим)
6. Навигация: `‹ Назад` / `Вперёд ›` по страницам, `« Урок` / `Урок »` между лекциями

Фильтр `MIN_ANSWER_SCORE` (0.82, откалиброван под e5) отсекает слабые совпадения,
чтобы не показывать пользователю ложную ссылку. Дополнительно источники
дедуплицируются по уроку (`answer_utils.dedupe_sources`).

## Модули

| Файл / функция | Назначение |
|---|---|
| `config.py` | **Единый источник констант** (пути, модели, параметры чанкинга/ретривала/LLM) |
| `download_pdfs.py` / `YandexDiskClient` | Скачивание PDF уроков с публичного Яндекс Диска |
| `pars_pdf.extract_text_from_pdf` | Страницы с citation-полями |
| `pars_pdf.find_pdfs_in_dir` | Список всех PDF из `data/raw` |
| `pars_pdf.batch_extract_pages` / `batch_extract_chunks` | Парсинг и чанкинг всей коллекции |
| `pars_pdf.save_pages_and_chunks` | Сохранение `pdf_pages.csv` + `pdf_chunks.csv` |
| `pars_pdf.chunk_pages` | Чанки с overlap |
| `pars_pdf.build_pages_dataframe` / `summarize_pages_dataset` | EDA |
| `embeddings.embed_chunks` | Векторы чанков |
| `embeddings.save_embeddings` / `load_embeddings` | `.npy` + meta CSV |
| `vector_store.VectorStore` / `ChromaStore` / `QdrantStore` | Абстракция векторного хранилища |
| `vector_store.create_vector_store` | Фабрика хранилища (по конфигу) |
| `index_to_vector_store.py` | Индексация `.npy`/CSV → Chroma/Qdrant |
| `llm.LLMClient` / `OpenAIClient` / `OllamaClient` | Абстракция LLM клиента |
| `llm.create_llm_client` | Фабрика LLM клиента |
| `rag_pipeline.RAGPipeline` | Полный RAG пайплайн (retrieve + generate) |
| `rag_pipeline.create_rag_pipeline` | Фабрика RAG пайплайна |
| `rag_pipeline.RAGConfig` / `RAGResult` | Конфиг и результат пайплайна |

## Пример использования RAG пайплайна

```python
from rag_pipeline import create_rag_pipeline, RAGConfig

# Создание пайплайна (автоматически использует Chroma + MockLLM если нет ключей)
pipeline = create_rag_pipeline(
    rag_config=RAGConfig(top_k=5, temperature=0.1)
)

# Простой вопрос
result = pipeline.ask("Что такое функция потерь?")
print(result.answer)
print("Источники:", result.sources)

# Вопрос с фильтром по уроку
result = pipeline.ask("Что такое градиентный спуск?", filter_lesson=6)
print(result.answer)

# С реальным LLM (нужен OPENAI_API_KEY в env или в конфиге)
pipeline = create_rag_pipeline(
    llm_config={"type": "openai", "openai": {"model": "gpt-4o-mini"}},
    rag_config=RAGConfig(top_k=5)
)
result = pipeline.ask("Как работает backpropagation?")
```

### LLM бэкенды

#### OpenRouter (бесплатно, рекомендуется)

```bash
cp .env.example .env      # вписать OPENROUTER_API_KEY
# ключ бесплатно: https://openrouter.ai/keys
```

```python
# выбирается автоматически, если задан OPENROUTER_API_KEY
pipeline = create_rag_pipeline()
```

**Проверенные бесплатные модели** (`OPENROUTER_FREE_MODELS` в `config.py`):

| Модель | Статус |
|--------|--------|
| `nvidia/nemotron-3-super-120b-a12b:free` | ✅ проверено, хороший русский |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | ✅ проверено |
| `google/gemma-4-31b-it:free` | ⚠️ резерв (бывает 400/429 upstream) |

`FallbackLLMClient` при 429/500/503/404 молча переключается на следующую модель,
а после успеха «прилипает» к ней и не тратит лимит на перебор.

#### ⚠️ Подводные камни OpenRouter

**1. Тип ключа.** Нужен обычный **inference**-ключ. Если взять *provisioning*-ключ
(он создаётся для программного управления ключами), то `/key` и `/credits` будут
отвечать 200, а `/chat/completions` — `401 User not found`. Проверить тип:

```bash
curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  | python -m json.tool | grep -E 'provisioning|management'
# is_provisioning_key: true  ->  ключ не подойдёт для запросов
```

Inference-ключ можно выпустить через provisioning-ключ:
```bash
curl -X POST https://openrouter.ai/api/v1/keys \
  -H "Authorization: Bearer $PROVISIONING_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-rag"}'
```

**2. `openrouter/free` — не использовать.** Этот авто-роутер может выбрать
неподходящую модель из бесплатного пула. На практике он направил запрос в
`nvidia/nemotron-3.5-content-safety` (классификатор модерации), который вместо
ответа вернул строку `User Safety: safe` — то есть пустые ответы при рабочем поиске.
Поэтому список моделей задан явно.

**3. Reasoning-модели.** Некоторые free-модели (например `nemotron-3.5-lightning`)
весь бюджет токенов тратят на внутренние рассуждения и возвращают
`Here's a thinking process: ...` с `finish_reason=length`. Лечится увеличением
`max_tokens` либо выбором обычной (не reasoning) модели.

Явно задать модель:
```python
pipeline = create_rag_pipeline(
    llm_config={"type": "openrouter", "openrouter": {
        "fallback_models": ["nvidia/nemotron-3-super-120b-a12b:free"],
    }}
)
```

#### OpenAI / Ollama

```bash
export OPENAI_API_KEY="sk-..."                          # или
export OLLAMA_BASE_URL="http://localhost:11434"         # локально
```
```python
pipeline = create_rag_pipeline(
    llm_config={"type": "ollama", "ollama": {"model": "llama3"}}
)
```

---

## 📊 EDA и подбор констант (30.09.2026)

Данные выросли с 1 PDF до 59 — константы пересобраны по реальным распределениям.

**Профиль корпуса:** 973 страницы, 593 000 символов, 59 PDF, 58 уроков.

| Параметр | Было | Стало | Обоснование |
|----------|------|-------|-------------|
| `DEFAULT_CHUNK_SIZE` | 500 | **1000** | Медиана страницы 619 симв. — при 500 текст резался на 72% страниц |
| `DEFAULT_CHUNK_OVERLAP` | 100 | **150** | 15% от размера вместо 20% — меньше дублирования при меньших потерях |
| `DEFAULT_TOP_K` | 5 | **8** | Корпус вырос в 60 раз, нужно шире окно поиска |
| `EMBEDDING_BATCH_SIZE` | 32 | **64** | Ускорение на ~1000 чанках, памяти хватает с запасом |
| `DEFAULT_LLM_MAX_TOKENS` | 1024 | **1536** | Больше контекста → простор для ответа с цитированием |

**Симуляция размера чанка на реальных страницах:**

| `chunk_size` | чанков | страниц в 1 чанк | разорванных страниц |
|---|---|---|---|
| 500 (было) | 1800 | 27.5% | 71.7% |
| 800 | 1168 | 81.1% | 18.2% |
| **1000 (выбрано)** | **1034** | **93.0%** | **6.3%** |
| 1200 | 1006 | 95.4% | 3.9% |
| 1500 | 986 | 97.2% | 2.1% |

1000 символов — «колено кривой»: +20% контекста (1200) даёт всего +1.2 п.п. качества.
Результат: **−43% векторов** (1800 → 1034) при радикально меньшем количестве разрывов.

**Побочный эффект:** качество поиска выросло — на запрос «функция потерь»
в выдаче появились релевантные уроки 24 и 39, а не дубликаты одной страницы.

### Переключение векторной БД

```python
# Chroma (по умолчанию)
pipeline = create_rag_pipeline()

# Qdrant
pipeline = create_rag_pipeline(
    vector_store_config={
        "type": "qdrant",
        "qdrant": {"url": "http://localhost:6333", "collection": "lessons"}
    }
)
```

---

## 🔄 Смена модели эмбеддингов (06.10.2026)

**Симптом.** На вопрос «Что такое переобучение?» система отвечала «нет ответа»,
хотя термин встречается в 44 чанках. Диагностика показала:

```
чанк с текстом «Что такое переобучение?» (урок 7, стр. 2)  -> score 0.424, позиция 28
нерелевантный слайд урока 39                               -> score 0.526
```

Чанк, буквально содержащий вопрос, проигрывал чужим страницам.

**Причина.** `paraphrase-multilingual-MiniLM-L12-v2` — модель для **сравнения
парафраз** (симметричная задача «эти два текста про одно?»). Для поиска нужна
**асимметричная** модель «короткий вопрос → документ». Это разные задачи.

**Решение.** Перешли на `intfloat/multilingual-e5-base` (768-dim) + префиксы
`query: ` / `passage: `, которые требует E5.

| Метрика | MiniLM-384 | **E5-base-768** |
|---------|-----------|-----------------|
| score чанка с текстом вопроса | 0.424 (позиция 28) | **0.860 (позиция 1)** |
| Типичный score релевантного | 0.42–0.53 | **0.84–0.88** |

**Ключевые детали внедрения:**

- **Префиксы обязательны.** E5 обучался с ними: без `query:`/`passage:` качество
  заметно падает. Реализовано через `embeddings.model_prefixes()` и параметр
  `role="query"|"passage"` у `embed_texts` — перепутать нельзя, иначе тихая
  деградация.
- **Порог пришлось калибровать заново.** На 16 вопросах (6 релевантных, 10 офтоп):
  ```
  релевантные: 0.840 .. 0.881
  офтоп:       0.730 .. 0.836   («билет на метро» -> 0.836)
  мёртвая зона: 0.836 .. 0.840
  ```
  Классы почти пересекаются, идеального порога нет. `MIN_ANSWER_SCORE = 0.82` —
  компромисс; финальное решение «отвечать или отказаться» принимает LLM по тексту.
- **Переиндексация обязательна** при смене модели: размерность меняется
  (384 → 768), старый индекс несовместим.

**Как менять модель дальше:** `DEFAULT_EMBEDDING_MODEL` + `EMBEDDING_DIM`
в `config.py` → `python embeddings.py` → `python index_to_vector_store.py --clear`.

---

## 📊 Eval: gold-набор и baseline (06.10.2026)

### Gold-набор

Собран LLM-ассистированным способом: `gold_set.py generate` → 63 вопроса с 25 страниц,
покрытие 21 урок. Типы сбалансированы (definition/how/compare), сложность — 21/21/21.
Все вопросы помечены `verified=True` с заметкой «LLM-черновик, авто-принят для baseline».

```bash
python gold_set.py generate --n 25    # черновик через LLM
python gold_set.py validate           # проверка структуры
python gold_set.py approve            # отметить после ревью
python gold_set.py eval               # Hit@k + MRR
```

### Eval: gold-набор и метрики (07.10.2026)

`python retrieval_report.py` на текущем индексе (e5-base, top_k=8):

| Метрика | Значение |
|---------|----------|
| Hit@8 | **55/63 (87.3%)** |

> Ранние метрики (Hit@1 0.365 … MRR 0.491) измерялись на **MiniLM-384**, до
> перехода на `multilingual-e5-base`; оставлены как история.

### 🔍 Очистка текста и «мусорность» страниц

Слайды содержат диаграммы, и pdfplumber перемешивает их подписи
(`Н И а м п е р е а т...`), тащит математические глифы (`𝕐`, `𝑥`) и эмодзи.
Модуль `text_clean.py` нормализует Unicode (NFKC), убирает эмодзи и отбрасывает
«мусорные» строки; он подключён в `pars_pdf.extract_text_from_pdf`.

**Важно:** первая версия метрики `garble_ratio` считала мусором обычные
односложные слова («в», «и», «с») и **завышала** проблему (якобы 77% страниц).
После исправления (исключены валидные односложные слова):

| Показатель | До очистки | После очистки |
|------------|-----------|---------------|
| Средняя «мусорность» страниц | 0.056 | **0.040** |
| Страниц выше порога 0.15 | 91 | **38** |
| Сохранено символов | — | 98.1% |

**Эффект на retrieval:** переиндексация очищенного текста подняла Hit@8
**84.1% → 87.3%** (+2 вопроса). Доля провалов, связанных с «мусорным» текстом,
стала **0/8** — то есть оставшиеся провалы вызваны семантикой/покрытием, а не
парсингом. Полноценный парсинг таблиц (Camelot / OpenCV, см. `idea.MD`) остаётся
опцией на будущее, но перестал быть блокером.

```bash
python retrieval_report.py            # полный отчёт с разбивкой по типам
python retrieval_report.py --top-k 20
```

---

## 🧪 Тесты и качество

Проект разделяет проверки на **unit** (быстрые, без данных — гоняются в CI) и
**integration** (нужен индекс и PDF — локально).

```bash
make test         # unit-тесты (pytest), ~2 c, без torch
make lint         # ruff + black --check + node --check
```

| Уровень | Что покрыто | Файлы |
|---------|-------------|-------|
| Unit | `sanitize_answer`, `dedupe_sources`, `substitute_refs`, bcrypt/JWT, схемы API, чанкинг, gold-парсинг, структура фронтенда | `tests/` |
| Integration | сквозной путь PDF → индекс → RAG-ответ | `test_full_pipeline.py` |

Структурная проверка фронтенда (`frontend_check.py`) ловит класс ошибок, который
не виден синтаксическому парсеру: незакрытая функция или CSS-правило «утаскивает»
весь код внутрь себя — формы уходят в нативный submit, а стили перестают
применяться.

## 🔁 CI/CD

`.github/workflows/ci.yml` при каждом push/PR запускает: `ruff check` (только
реальные ошибки — `E9/F63/F7/F82`), `black --check`, `pytest`, `py_compile` всех
модулей и `node --check` для `app.js`.

## 🐳 Docker

```bash
make index                   # индекс должен лежать в ./data (монтируется томом)
docker compose up --build    # → http://localhost:8000
```

`Dockerfile` — `python:3.12-slim` (CPU-only). `docker-compose.yml` монтирует
`./data` (индекс Chroma + SQLite), читает секреты из `.env` и проверяет
`/api/health` через healthcheck.

## 🗑️ Что можно удалить / оптимизировать

### Кандидаты на удаление (legacy / не используются)

| Файл | Причина | Статус |
|------|---------|--------|
| ~~`main.py`~~ | Битый импорт (`batch_extract_text_from_pdf` не существует) | ✅ **Удалён** |
| ~~`utils.py`~~ | `setup_logging()` нигде не вызывается | ✅ **Удалён** |
| `pars_pdf.ipynb` | Ноутбук для исследования, дублирует `pars_pdf.py` | Можно удалить |
| `test_vector_store.py` | Простой тест, функционал в `test_full_pipeline.py` | Можно удалить |

### Дублирующиеся артефакты (после перехода на VectorStore)

После того как Chroma/Qdrant стал основным хранилищем:
- `data/chunk_embeddings.npy` — **можно удалить** (векторы в Chroma)
- `data/chunk_meta.csv` — **можно удалить** (метаданные в Chroma)
- `data/pdf_pages.csv`, `data/pdf_chunks.csv` — промежуточные, **можно удалить** после проверки

### Оптимизации кода (приоритеты)

| Область | Что улучшить | Приоритет |
|---------|--------------|-----------|
| **Типизация** | Добавить type hints везде, `mypy --strict` | High |
| **Docstrings** | Google/NumPy style везде | High |
| **Error handling** | Try/except + логирование вместо падений | Medium |
| **Chunking** | Токен-чанкинг (tiktoken) вместо символов | Medium |
| **Embeddings** | Кэширование модели, batch оптимизация | Medium |
| **VectorStore** | Connection pooling для Qdrant, health checks | Low |
| **Logging** | Заменить `print` на `logging` модуль | Medium |
| **Config** | Pydantic Settings для валидации конфигов | Medium |
| **Tests** | pytest + fixtures для всех модулей | ✅ сделано (`tests/`, 33 теста) |
| **CI/CD** | GitHub Actions: lint + тесты + проверка JS | ✅ сделано (`.github/workflows/ci.yml`) |
| **Docker** | Dockerfile + compose | ✅ сделано |

### Архитектурные улучшения (следующие этапы)

1. **Reranker** (cross-encoder) после retrieval
2. **Hybrid search** (BM25 + vector) для точных совпадений  
3. **Query expansion** / HyDE для лучшего recall
4. **Streaming** ответов LLM
5. **Async** версии методов для FastAPI
6. **Observability** (Langfuse, MLflow)
7. **Evaluation harness** (Hit@k, citation accuracy)