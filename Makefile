# Удобные команды разработки. Подробности — в DEVELOPER_GUIDE.md (§ 10 «Запуск»).
#
# RECIPEPREFIX='>' — рецепты начинаются с '>', чтобы Makefile не зависел от
# символа табуляции (его легко потерять при редактировании).
.RECIPEPREFIX = >

PY ?= python
COMPOSE ?= docker compose

.PHONY: help install dev download parse index run test lint format docker-build docker-up docker-down

help: ## Список команд
> @grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

install: ## Рантайм-зависимости (полный набор, включая torch)
> $(PY) -m pip install -r requirements.txt

dev: ## Зависимости для разработки и тестов (без torch)
> $(PY) -m pip install -r requirements-dev.txt

download: ## Скачать PDF с Яндекс Диска → data/raw/
> $(PY) download_pdfs.py

parse: ## Парсинг PDF и чанкинг
> $(PY) pars_pdf.py

index: ## Эмбеддинги + индексация в Chroma
> $(PY) embeddings.py
> $(PY) index_to_vector_store.py --clear

run: ## Запустить веб-приложение → http://localhost:8000
> $(PY) -m webapp.main

test: ## Unit-тесты
> $(PY) -m pytest

lint: ## Линтер + проверка формата + синтаксис JS
> ruff check .
> black --check .
> node --check webapp/static/app.js

format: ## Автоформатирование (black)
> black .

docker-build: ## Собрать Docker-образ
> docker build -t mrag-web:latest .

docker-up: ## Поднять приложение через compose
> $(COMPOSE) up --build

docker-down: ## Остановить compose
> $(COMPOSE) down
