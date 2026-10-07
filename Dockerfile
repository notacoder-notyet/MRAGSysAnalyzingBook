# Образ приложения MRAG (FastAPI + локальный RAG). CPU-only, без GPU.
#
# Сборка:  docker build -t mrag-web:latest .
# Запуск:  docker compose up --build
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Системные библиотеки для сборки научных пакетов (torch/sentence-transformers).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Сначала зависимости — слой кэшируется между сборками кода.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Затем код приложения.
COPY . .

# Индекс Chroma и SQLite-база живут в /app/data (монтируется томом).
VOLUME ["/app/data"]

EXPOSE 8000

CMD ["python", "-m", "webapp.main"]
