"""
LLM абстракция с поддержкой нескольких бэкендов:
- OpenAI API (gpt-4o, gpt-4o-mini, etc.)
- Ollama (локальные модели: llama3, mistral, etc.)
- Совместимый с OpenAI API (vLLM, LM Studio, etc.)
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# Ключи храним в .env (в .gitignore), а не в коде или переменных окружения руками.
# .env необязателен: если файла нет, всё работает как раньше.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:  # python-dotenv не установлен — не критично
    pass

from config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OPENROUTER_APP_TITLE,
    OPENROUTER_BASE_URL,
    OPENROUTER_FREE_MODELS,
    OPENROUTER_HTTP_REFERER,
    OPENROUTER_TIMEOUT,
)


@dataclass
class LLMResponse:
    """Результат генерации LLM."""

    text: str
    model: str
    usage: dict[str, int] | None = None
    raw: Any = None


class LLMClient(abc.ABC):
    """Базовый интерфейс LLM клиента."""

    @abc.abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion."""
        ...

    @abc.abstractmethod
    def complete(
        self,
        prompt: str,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Text completion."""
        ...


class OpenAIClient(LLMClient):
    """OpenAI API клиент (включает совместимые API: vLLM, LM Studio, etc.)."""

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_LLM_TIMEOUT,
    ):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=timeout,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            raw=response,
        )

    def complete(
        self,
        prompt: str,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


class OllamaClient(LLMClient):
    """Ollama локальный клиент."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = OLLAMA_TIMEOUT,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or -1,
            },
        }
        response = self._client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model=self.model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
            raw=data,
        )

    def complete(
        self,
        prompt: str,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or -1,
            },
        }
        response = self._client.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return LLMResponse(
            text=data.get("response", ""),
            model=self.model,
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            raw=data,
        )

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class OpenRouterClient(OpenAIClient):
    """
    Клиент OpenRouter — работает через OpenAI-совместимый API.

    Отличия от обычного OpenAI:
      - свой base_url и обязательные заголовки атрибуции трафика;
      - бесплатные модели жёстко rate-limited, поэтому по умолчанию
        включён резервный список моделей (см. OPENROUTER_FREE_MODELS).
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = OPENROUTER_TIMEOUT,
        fallback_models: tuple[str, ...] = OPENROUTER_FREE_MODELS,
    ) -> None:
        """
        Args:
            model: ID модели на OpenRouter. None — берётся первая из fallback.
            api_key: Ключ OpenRouter (или env OPENROUTER_API_KEY).
            timeout: Таймаут запроса, секунды.
            fallback_models: Модели для переключения при rate-limit.
        """
        primary = model or (fallback_models[0] if fallback_models else "auto")
        super().__init__(
            model=primary,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
            timeout=timeout,
        )
        self.fallback_models = fallback_models

        # OpenRouter просит атрибуцию трафика — без заголовков бывает 403
        self.client.default_headers.update(
            {
                "HTTP-Referer": OPENROUTER_HTTP_REFERER,
                "X-Title": OPENROUTER_APP_TITLE,
            }
        )


class FallbackLLMClient(LLMClient):
    """
    Обёртка, которая перебирает несколько LLM-клиентов по кругу.

    Нужна для бесплатных моделей: при исчерпании дневного лимита OpenRouter
    отдаёт 429, и вместо падения мы молча уходим на следующую модель.

    Каждый успешный ответ запоминается (sticky) — пока работает текущая
    модель, не тратим лимит на перебор.
    """

    def __init__(self, clients: list[LLMClient], max_attempts: int | None = None) -> None:
        """
        Args:
            clients: Список клиентов в порядке предпочтения.
            max_attempts: Сколько попыток суммарно (None — все клиенты).
        """
        if not clients:
            raise ValueError("FallbackLLMClient требует хотя бы одного клиента")
        self._clients = clients
        self._max_attempts = max_attempts or len(clients)
        self._preferred = 0  # индекс клиента, на котором остановились
        self.last_model: str = ""

    @property
    def model(self) -> str:
        """Модель текущего активного клиента."""
        return getattr(self._clients[self._preferred], "model", "unknown")

    def _is_retryable(self, error: Exception) -> bool:
        """
        Определяет, стоит ли переключаться на другую модель.

        Повторяем при:
          - 429 — rate-limit (главная причина у бесплатных моделей);
          - 500/502/503/504/529 — перегрузка провайдера;
          - 404 — Free Models Router вернул «модель больше не бесплатна»
            (список free-моделей у OpenRouter меняется на ходу);
          - сетевые обрывы и таймауты.
        Не повторяем при 400/401/402 — это наша ошибка в запросе, ключе или балансе.
        """
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in {404, 429, 500, 502, 503, 504, 529}
        # Сетевые обрывы и таймауты тоже лечим сменой модели
        return isinstance(error, (httpx.RequestError, TimeoutError))

    def _try(self, method: str, *args: Any, **kwargs: Any) -> LLMResponse:
        """
        Выполняет вызов, перебирая клиентов при retryable-ошибке.

        Args:
            method: Имя метода LLMClient ("chat" или "complete").
            *args: Позиционные аргументы метода.
            **kwargs: Именованные аргументы метода.

        Returns:
            Первый успешный LLMResponse.

        Raises:
            RuntimeError: Если не сработал ни один клиент.
        """
        attempts = 0
        errors: list[str] = []

        for offset in range(len(self._clients)):
            index = (self._preferred + offset) % len(self._clients)
            client = self._clients[index]
            attempts += 1

            try:
                response: LLMResponse = getattr(client, method)(*args, **kwargs)
                self._preferred = index  # sticky: остаёмся на удачной модели
                self.last_model = response.model
                return response

            except Exception as error:  # noqa: BLE001 — решаем по _is_retryable
                errors.append(f"{getattr(client, 'model', '?')}: {error}")
                if not self._is_retryable(error) or attempts >= self._max_attempts:
                    break

        raise RuntimeError(
            f"Все LLM-бэкенды не ответили ({len(errors)} попыток):\n  " + "\n  ".join(errors)
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat completion с перебором моделей при rate-limit."""
        return self._try("chat", messages, temperature=temperature, max_tokens=max_tokens, **kwargs)

    def complete(
        self,
        prompt: str,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Text completion с перебором моделей при rate-limit."""
        return self._try(
            "complete", prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    def close(self) -> None:
        """Закрывает все вложенные клиенты, которые это умеют."""
        for client in self._clients:
            closer = getattr(client, "close", None)
            if callable(closer):
                closer()


def create_llm_client(config: dict[str, Any] | None = None) -> LLMClient:
    """
    Фабрика LLM клиента.

    Args:
        config: Словарь конфигурации. Примеры:
        {
            "type": "openai",
            "openai": {"model": "gpt-4o-mini", "api_key": "...", "base_url": "..."}
        }
        {
            "type": "ollama",
            "ollama": {"model": "llama3", "base_url": "http://localhost:11434"}
        }

    Если config не передан, пробует определить из env переменных:
    - OPENROUTER_API_KEY → OpenRouterClient с цепочкой бесплатных моделей
    - OPENAI_API_KEY → OpenAIClient
    - OLLAMA_BASE_URL → OllamaClient
    - по умолчанию → возвращает MockClient для тестов retrieval
    """
    if config is None:
        config = {}

    llm_type = config.get("type")
    if not llm_type:
        # OpenRouter проверяем первым: это основной бесплатный бэкенд
        if os.getenv("OPENROUTER_API_KEY"):
            llm_type = "openrouter"
        elif os.getenv("OPENAI_API_KEY"):
            llm_type = "openai"
        elif os.getenv("OLLAMA_BASE_URL"):
            llm_type = "ollama"
        else:
            # Нет ключей - вернём mock для тестирования retrieval
            return MockLLMClient()

    if llm_type == "openrouter":
        or_cfg = config.get("openrouter", {})
        api_key = or_cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return MockLLMClient()

        models = tuple(or_cfg.get("fallback_models") or OPENROUTER_FREE_MODELS)
        clients = [OpenRouterClient(model=m, api_key=api_key) for m in models]

        # Одна модель — без лишней обёртки; несколько — с фолбэк-цепочкой
        if len(clients) == 1:
            return clients[0]
        return FallbackLLMClient(clients)

    if llm_type == "openai":
        openai_cfg = config.get("openai", {})
        api_key = openai_cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return MockLLMClient()
        return OpenAIClient(
            model=openai_cfg.get("model", DEFAULT_LLM_MODEL),
            api_key=api_key,
            base_url=openai_cfg.get("base_url"),
        )
    elif llm_type == "ollama":
        ollama_cfg = config.get("ollama", {})
        return OllamaClient(
            model=ollama_cfg.get("model", DEFAULT_OLLAMA_MODEL),
            base_url=ollama_cfg.get("base_url", DEFAULT_OLLAMA_BASE_URL),
        )
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


class MockLLMClient(LLMClient):
    """Mock клиент для тестирования retrieval без LLM."""

    def __init__(self):
        self.model = "mock"

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        # Извлекаем контекст из system prompt для демо-ответа
        context = ""
        for msg in messages:
            if msg["role"] == "system":
                context = msg["content"]
                break
        return LLMResponse(
            text=f"[MOCK] Ответ на основе {len(messages)} сообщений. Контекст: {context[:200]}...",
            model="mock",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    def complete(
        self,
        prompt: str,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        return self.chat([{"role": "user", "content": prompt}], temperature, max_tokens)


if __name__ == "__main__":
    print("Тест LLM клиента...")
    try:
        client = create_llm_client({"type": "openai", "openai": {"model": DEFAULT_LLM_MODEL}})
        print(f"Создан клиент: {type(client).__name__}, модель: {client.model}")
    except Exception as e:
        print(f"Ошибка (ожидаемо без API ключа): {e}")

    try:
        client = create_llm_client({"type": "ollama", "ollama": {"model": DEFAULT_OLLAMA_MODEL}})
        print(f"Создан Ollama клиент: {type(client).__name__}, модель: {client.model}")
    except Exception as e:
        print(f"Ollama недоступен (ожидаемо): {e}")
