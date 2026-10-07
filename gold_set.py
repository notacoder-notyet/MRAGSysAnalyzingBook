"""
Gold-набор для оценки качества RAG-пайплайна.

Единица истины — пара (урок, страница), а не чанк: вопрос может быть закрыт
любым из чанков этой страницы, поэтому разметка по чанкам занижает Hit@k.

Схема записи (CSV, data/gold_questions.csv):
    question_id      — уникальный id (q001, q002, ...)
    question         — текст вопроса
    expected_refs    — список пар "урок:страница" через "|", напр. "6:3" или "6:3|6:4"
    answerable       — True/False. False = вопрос НЕ из учебника (для проверки abstain)
    difficulty       — easy | medium | hard
    question_type    — definition | how | why | compare | calculate | example
    verified         — True/False. True = проверено человеком
    notes            — свободные заметки

Рабочий процесс:
    1. Сгенерировать кандидатов:      python gold_set.py sample
    2. Сгенерировать вопросы LLM по кандидатам (см. build_prompt)
    3. Заполнить CSV руками, проставив verified=True
    4. Проверить валидность:            python gold_set.py validate
    5. Посчитать метрики:              python gold_set.py eval
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd

from config import CHUNKS_CSV_PATH, DEFAULT_TOP_K, DATA_DIR

GOLD_CSV_PATH = Path(DATA_DIR) / "gold_questions.csv"

# Сколько чанков максимум набирать с одной (lesson, page) при сэмплировании
MAX_CHARS_PER_REF = 2500
# Доля от общего числа вопросов на один урок (для баланса)
PER_LESSON_SHARE = 0.08
# Уроки-исключения: перекос по объёму (lesson 39 = 11% корпуса)
LESSON_QUARANTINE = {39}

VALID_DIFFICULTY = {"easy", "medium", "hard"}
VALID_TYPES = {"definition", "how", "why", "compare", "calculate", "example"}


@dataclass
class GoldItem:
    """Одна запись gold-набора."""

    question_id: str
    question: str
    expected_refs: list[tuple[int, int]] = field(default_factory=list)
    answerable: bool = True
    difficulty: str = "medium"
    question_type: str = "definition"
    verified: bool = False
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        """Сериализует запись в строку CSV (refs — в формате '6:3|6:4')."""
        row = asdict(self)
        row["expected_refs"] = "|".join(f"{l}:{p}" for l, p in self.expected_refs)
        return row

    @staticmethod
    def parse_refs(raw: str) -> list[tuple[int, int]]:
        """Парсит строку '6:3|6:4' в список пар [(6,3), (6,4)]."""
        if not raw or str(raw).strip() in {"nan", "None", ""}:
            return []
        refs: list[tuple[int, int]] = []
        for part in str(raw).split("|"):
            part = part.strip()
            match = re.fullmatch(r"(\d+)\s*:\s*(\d+)", part)
            if match:
                refs.append((int(match.group(1)), int(match.group(2))))
        return refs


def load_gold(path: str | Path = GOLD_CSV_PATH) -> pd.DataFrame:
    """Загружает gold-набор; если файла нет — возвращает пустой DataFrame."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(
            columns=[
                "question_id",
                "question",
                "expected_refs",
                "answerable",
                "difficulty",
                "question_type",
                "verified",
                "notes",
            ]
        )
    return pd.read_csv(p, dtype={"expected_refs": str, "notes": str}).fillna(
        {"expected_refs": "", "notes": ""}
    )


def save_gold(df: pd.DataFrame, path: str | Path = GOLD_CSV_PATH) -> None:
    """Сохраняет gold-набор в CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def refs_to_series(df: pd.DataFrame) -> list[list[tuple[int, int]]]:
    """Векторизованный разбор колонки expected_refs."""
    return [GoldItem.parse_refs(v) for v in df["expected_refs"].tolist()]


def sample_candidates(
    chunks_path: str | Path = CHUNKS_CSV_PATH,
    n_candidates: int = 60,
    seed: int = 42,
    min_chars: int = 400,
    max_chars: int = MAX_CHARS_PER_REF,
) -> list[dict[str, Any]]:
    """
    Отбирает (lesson, page)-пары, из которых стоит генерировать вопросы.

    Логика отбора:
      - страницы с осмысленным объёмом текста (min_chars..max_chars);
      - баланс по урокам — не более PER_LESSON_SHARE от общего числа кандидатов,
        иначе крупные уроки (например 39) забьют собой выборку;
      - карусельное расстояние между соседними страницами, чтобы вопросы
        не оказались все из одного урока.

    Args:
        chunks_path: CSV с чанками (pdf_chunks.csv).
        n_candidates: Сколько кандидатов вернуть.
        seed: Seed для воспроизводимости.
        min_chars: Минимум символов на страницу (отсекаем пустые и огрызки).
        max_chars: Максимум символов (не даём раздувать контекст).

    Returns:
        Список словарей {lesson, page, n_chunks, text}.
    """
    df = pd.read_csv(chunks_path)

    # Склеиваем все чанки страницы в один текст
    pages = (
        df.groupby(["lesson", "page"])
        .agg(n_chunks=("chunk_id", "size"), text=("text", lambda s: "\n".join(s)))
        .reset_index()
    )
    pages["n_chars"] = pages["text"].str.len()

    ok = pages[(pages["n_chars"] >= min_chars) & (pages["n_chars"] <= max_chars)]
    ok = ok[~ok["lesson"].isin(LESSON_QUARANTINE)]

    # Перемешиваем страницы с фиксированным seed: без этого groupby().head()
    # всегда брал бы первые страницы урока, и выборка смещалась бы к его началу
    ok = ok.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Ограничиваем число страниц на урок, чтобы крупные лекции не доминировали
    per_lesson = max(1, int(n_candidates * PER_LESSON_SHARE))
    picked = ok.groupby("lesson").head(per_lesson)

    # Равномерно прореживаем по всему списку
    if len(picked) > n_candidates:
        step = len(picked) / n_candidates
        idx = [int(i * step) for i in range(n_candidates)]
        picked = picked.iloc[idx]

    picked = picked.copy()
    picked["text"] = picked["text"].str.slice(0, max_chars)
    return picked[["lesson", "page", "n_chunks", "text"]].to_dict("records")


def build_prompt(lesson: int, page: int, text: str) -> str:
    """
    Собирает промпт для генерации вопросов LLM по фрагменту страницы.

    Требования к LLM внутри промпта:
      - вопрос должен быть сформулирован ПЕРЕФРАЗИРОВАННО, без копирования
        дословных кусков (иначе Hit@k будет завышен тривиальным совпадением);
      - ответ должен выводиться ТОЛЬКО из этого фрагмента;
      - не упоминать «страницу»/«слайд» в тексте вопроса.

    Args:
        lesson: Номер урока.
        page: Номер страницы.
        text: Текст страницы (чанки склеены).

    Returns:
        Готовый промпт для LLM.
    """
    return f"""На основе фрагмента учебника составь 3 вопроса разных типов.

ТРЕБОВАНИЯ:
1. Вопросы на русском языке, в стиле «что студент спросит, готовясь к экзамену».
2. НЕ копируй дословно фрагменты — перефразируй суть своими словами.
3. В самом вопросе не упоминай номера страниц, слайдов и уроков.
4. Каждый вопрос должен быть закрываем ТОЛЬКО этим фрагментом.
5. Укажи для каждого: тип (definition/how/why/compare/calculate/example) и сложность.

ФОРМАТ ОТВЕТА (строго JSON, без пояснений):
{{
  "questions": [
    {{"type": "definition", "difficulty": "easy", "question": "..."}},
    {{"type": "how", "difficulty": "medium", "question": "..."}},
    {{"type": "compare", "difficulty": "hard", "question": "..."}}
  ]
}}

КОНТЕКСТ (урок {lesson}, страница {page}):
{text}"""


def validate_gold(df: pd.DataFrame, chunks_path: str | Path = CHUNKS_CSV_PATH) -> pd.DataFrame:
    """
    Проверяет качество gold-набора и возвращает список проблем.

    Args:
        df: Загруженный gold-набор.
        chunks_path: CSV с чанками — для сверки существования (lesson, page).

    Returns:
        DataFrame с колонками row, issue, detail.
    """
    issues: list[dict[str, Any]] = []

    def add(row: int, issue: str, detail: str) -> None:
        issues.append({"row": row, "issue": issue, "detail": detail})

    if df.empty:
        add(0, "EMPTY", "gold-набор пуст")
        return pd.DataFrame(issues)

    chunks = pd.read_csv(chunks_path)
    valid_refs = {
        (int(r.lesson), int(r.page))
        for r in chunks[["lesson", "page"]].drop_duplicates().itertuples()
    }

    seen_questions: dict[str, int] = {}
    for i, row in df.iterrows():
        num = i + 2  # +2: заголовок CSV и 1-индексация
        q = str(row.get("question", "")).strip()
        answerable = str(row.get("answerable", "True")).strip().lower() in {"true", "1", "yes"}
        refs = GoldItem.parse_refs(row.get("expected_refs", ""))

        if len(q) < 10:
            add(num, "SHORT_QUESTION", q)
        key = q.lower().rstrip("?")
        if key in seen_questions:
            add(num, "DUPLICATE", f"повтор q{seen_questions[key]}: {q[:40]}")
        else:
            seen_questions[key] = num

        if answerable and not refs:
            add(num, "MISSING_REFS", "answerable=True, но expected_refs пуст")
        if not answerable and refs:
            add(num, "UNEXPECTED_REFS", "answerable=False, но refs заданы")
        for lesson, page in refs:
            if (lesson, page) not in valid_refs:
                add(num, "BAD_REF", f"нет такой страницы: урок {lesson}, стр. {page}")

        difficulty = str(row.get("difficulty", "")).strip().lower()
        if difficulty not in VALID_DIFFICULTY:
            add(num, "BAD_DIFFICULTY", difficulty)
        qtype = str(row.get("question_type", "")).strip().lower()
        if qtype not in VALID_TYPES:
            add(num, "BAD_TYPE", qtype)

        verified = str(row.get("verified", "False")).strip().lower() in {"true", "1", "yes"}
        if not verified:
            add(num, "NOT_VERIFIED", "требует ручной проверки")

    return pd.DataFrame(issues)


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Сводка по покрытию: распределение вопросов по урокам и типам."""
    if df.empty:
        return pd.DataFrame()
    parsed = refs_to_series(df)
    lessons = [refs[0][0] if refs else None for refs in parsed]
    tmp = df.assign(_lesson=lessons)
    out = (
        tmp.groupby("_lesson")
        .agg(questions=("question_id", "size"), verified=("verified", "sum"))
        .reset_index()
        .rename(columns={"_lesson": "lesson"})
    )
    return out.sort_values("lesson").reset_index(drop=True)


def evaluate_retrieval(
    gold_path: str | Path = GOLD_CSV_PATH,
    top_k: int = DEFAULT_TOP_K,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """
    Считает Hit@k и MRR по gold-набору.

    Hit@k — доля вопросов, у которых хотя бы одна ожидаемая (урок, страница)
    попала в top-k выдачи. MRR — средняя обратная позиция первого попадания.

    Args:
        gold_path: CSV с gold-набором (учитываются только verified=True).
        top_k: Размер выдачи, который запрашиваем у VectorStore.
        ks: Значения k, для которых считаем Hit@k.

    Returns:
        Словарь метрик.
    """
    from embeddings import embed_texts, load_embedding_model
    from vector_store import create_vector_store

    df = load_gold(gold_path)
    if df.empty:
        return {}

    verified_mask = df["verified"].astype(str).str.lower().isin({"true", "1", "yes"})
    df = df[verified_mask]
    if df.empty:
        return {}

    store = create_vector_store()
    model = load_embedding_model()

    questions = df["question"].astype(str).tolist()
    vectors = embed_texts(questions, model=model, show_progress=False, role="query")
    max_k = max(max(ks), top_k)

    hits = {k: 0 for k in ks}
    reciprocal_ranks: list[float] = []
    total = 0

    for _, row in df.iterrows():
        refs = set(GoldItem.parse_refs(row["expected_refs"]))
        if not refs:
            continue
        total += 1
        results = store.search(vectors[total - 1], top_k=max_k)
        found = [
            (h["metadata"].get("lesson"), h["metadata"].get("page")) for h in results
        ]
        for k in ks:
            if refs & set(found[:k]):
                hits[k] += 1
        for rank, ref in enumerate(found, start=1):
            if ref in refs:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    metrics: dict[str, float] = {"n_questions": float(total)}
    for k in ks:
        metrics[f"hit@{k}"] = hits[k] / total if total else 0.0
    metrics["mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return metrics


def extract_json_block(raw: str) -> dict[str, Any] | None:
    """
    Извлекает JSON-объект из ответа LLM.

    Модели часто оборачивают JSON в ```json ... ``` или добавляют прозу до/после.
    Поэтому ищем первую `{` и последнюю `}` и пытаемся распарсить срез.

    Args:
        raw: Сырой текст ответа модели.

    Returns:
        Разобранный dict или None, если JSON найти не удалось.
    """
    if not raw:
        return None

    # Убираем markdown-обёртку
    fenced = re.search(r"```(?:json)?\s*(.+?)```", raw, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else raw

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    snippet = candidate[start : end + 1]
    try:
        parsed = ast.literal_eval(snippet)  # JSON — подмножество Python-литералов
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, SyntaxError):
        try:
            import json

            parsed = json.loads(snippet)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def generate_questions_for_ref(
    lesson: int,
    page: int,
    text: str,
    llm_client: Any,
    max_tokens: int = 900,
) -> list[dict[str, str]]:
    """
    Просит LLM сгенерировать вопросы по фрагменту одной страницы.

    Args:
        lesson: Номер урока.
        page: Номер страницы.
        text: Текст страницы.
        llm_client: LLM-клиент с методом chat().
        max_tokens: Лимит токенов на ответ.

    Returns:
        Список словарей {question, difficulty, question_type}.
        Пустой список, если модель не ответила или вернула невалидный JSON.
    """
    prompt = build_prompt(lesson, page, text)
    response = llm_client.chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.4,  # выше нуля — нужны разнообразные формулировки
    )
    payload = extract_json_block(response.text)
    if not payload:
        return []

    items = payload.get("questions")
    if not isinstance(items, list):
        return []

    cleaned: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if len(question) < 10:
            continue
        difficulty = str(item.get("difficulty", "medium")).strip().lower()
        qtype = str(item.get("type", item.get("question_type", "definition"))).strip().lower()
        cleaned.append(
            {
                "question": question,
                "difficulty": difficulty if difficulty in VALID_DIFFICULTY else "medium",
                "question_type": qtype if qtype in VALID_TYPES else "definition",
            }
        )
    return cleaned


def generate_gold_set(
    n_candidates: int = 25,
    llm_client: Any | None = None,
    questions_per_ref: int = 3,
    checkpoint_path: str | Path = GOLD_CSV_PATH,
) -> pd.DataFrame:
    """
    Собирает черновик gold-набора: кандидаты → LLM → CSV.

    Каждый вопрос сохраняется с verified=False — набор требует ручной проверки.
    После каждой страницы пишется чекпоинт, чтобы обрыв не потерял прогресс.

    Args:
        n_candidates: Сколько страниц-кандидатов обработать.
        llm_client: LLM-клиент. None — создаётся через create_llm_client().
        questions_per_ref: Сколько вопросов ожидаем на страницу (для лога).
        checkpoint_path: Куда сохранять чекпоинт.

    Returns:
        DataFrame с черновиком gold-набора.
    """
    from llm import create_llm_client, MockLLMClient

    if llm_client is None:
        llm_client = create_llm_client()
    if isinstance(llm_client, MockLLMClient):
        raise RuntimeError(
            "LLM недоступен (MockLLMClient). Проверьте OPENROUTER_API_KEY в .env"
        )

    candidates = sample_candidates(n_candidates=n_candidates)
    print(f"Кандидатов: {len(candidates)} | модель: {getattr(llm_client, 'model', '?')}\n")

    rows: list[dict[str, Any]] = []
    counter = 1
    failed_refs: list[str] = []

    for i, cand in enumerate(candidates, start=1):
        lesson, page = int(cand["lesson"]), int(cand["page"])
        ref = f"{lesson}:{page}"
        try:
            questions = generate_questions_for_ref(
                lesson, page, str(cand["text"]), llm_client
            )[:questions_per_ref]
        except Exception as error:  # noqa: BLE001 — не роняем весь прогон
            print(f"  [{i}/{len(candidates)}] {ref}: ОШИБКА {str(error)[:120]}")
            failed_refs.append(ref)
            continue

        for q in questions:
            rows.append(
                {
                    "question_id": f"q{counter:03d}",
                    "question": q["question"],
                    "expected_refs": ref,
                    "answerable": True,
                    "difficulty": q["difficulty"],
                    "question_type": q["question_type"],
                    "verified": False,
                    "notes": "черновик LLM, требует проверки",
                }
            )
            counter += 1

        print(f"  [{i}/{len(candidates)}] {ref}: +{len(questions)} вопросов")
        # Чекпоинт после каждой страницы — обрыв не потеряет прогресс
        save_gold(pd.DataFrame(rows), checkpoint_path)

    df = pd.DataFrame(rows)
    save_gold(df, checkpoint_path)
    print(f"\nВсего вопросов: {len(df)} из {len(candidates) - len(failed_refs)} страниц")
    if failed_refs:
        print(f"Не удалось по страницам: {failed_refs}")
    print(f"Сохранено: {checkpoint_path}")
    print("Далее: проверить вопросы руками и проставить verified=True")
    return df


SEED_EXAMPLES: list[dict[str, Any]] = [
    {
        "question_id": "q001",
        "question": "Какую функцию называют линейкой ошибок в машинном обучении?",
        "expected_refs": "6:3",
        "answerable": True,
        "difficulty": "easy",
        "question_type": "definition",
        "verified": True,
        "notes": "Прямое определение со слайда урока 6.",
    },
    {
        "question_id": "q002",
        "question": "Зачем нужны оптимизаторы, если уже есть функция потерь?",
        "expected_refs": "6:3|6:5",
        "answerable": True,
        "difficulty": "medium",
        "question_type": "why",
        "verified": True,
        "notes": "Ответ размазан по двум страницам — ref с двойкой.",
    },
    {
        "question_id": "q003",
        "question": "В каком году вышла спецификация HTTP/2 и кто её разрабатывал?",
        "expected_refs": "",
        "answerable": False,
        "difficulty": "easy",
        "question_type": "definition",
        "verified": True,
        "notes": "НЕ из учебника. Проверяет abstain.",
    },
]


def main() -> int:
    """CLI: sample | seed | validate | coverage | eval."""
    parser = argparse.ArgumentParser(description="Работа с gold-набором RAG")
    parser.add_argument(
        "command",
        choices=["sample", "generate", "approve", "seed", "validate", "coverage", "eval"],
        help="Действие",
    )
    parser.add_argument("--n", type=int, default=60, help="Кандидатов для sample/generate")
    parser.add_argument("--out", default=str(GOLD_CSV_PATH), help="Путь к CSV")
    parser.add_argument("--prompts", help="Сохранить промпты в файл (для sample)")
    parser.add_argument("--note", help="Заметка, добавляемая при approve")
    args = parser.parse_args()

    if args.command == "sample":
        candidates = sample_candidates(n_candidates=args.n)
        print(f"Кандидатов: {len(candidates)}")
        for c in candidates[:15]:
            print(f"  урок {c['lesson']:3} стр. {c['page']:3} | {c['n_chunks']} чанков | {len(c['text'])} симв.")
        if args.prompts:
            with open(args.prompts, "w", encoding="utf-8") as f:
                for c in candidates:
                    f.write(
                        f"### REF {c['lesson']}:{c['page']}\n"
                        + build_prompt(c["lesson"], c["page"], c["text"])
                        + "\n\n"
                    )
            print(f"\nПромпты сохранены: {args.prompts}")
        return 0

    if args.command == "generate":
        generate_gold_set(n_candidates=args.n, checkpoint_path=args.out)
        return 0

    if args.command == "approve":
        df = load_gold(args.out)
        if df.empty:
            print("gold-набор пуст")
            return 1
        # Массовая отметка после ревью. Заметка сохраняется, чтобы отличать
        # вопросы, прошедшие ручную проверку, от авто-принятых.
        df["verified"] = True
        if args.note:
            df["notes"] = df["notes"].fillna("").astype(str) + f" | {args.note}"
        save_gold(df, args.out)
        print(f"Отмечено проверенными: {len(df)} вопросов")
        print(f"Сохранено: {args.out}")
        return 0

    if args.command == "seed":
        save_gold(pd.DataFrame(SEED_EXAMPLES), args.out)
        print(f"Создан seed gold-набор: {args.out} ({len(SEED_EXAMPLES)} примеров)")
        print("Отредактируйте CSV: добавьте вопросы, проставьте verified=True")
        return 0

    if args.command == "validate":
        df = load_gold(args.out)
        issues = validate_gold(df)
        if issues.empty:
            print("Проблем не найдено ✓")
        else:
            print(f"Найдено проблем: {len(issues)}")
            print(issues.to_string(index=False))
        return 0

    if args.command == "coverage":
        df = load_gold(args.out)
        cov = coverage_report(df)
        if cov.empty:
            print("gold-набор пуст")
        else:
            print(f"Вопросов: {len(df)} | уроков покрыто: {len(cov)}")
            print(cov.to_string(index=False))
        return 0

    metrics = evaluate_retrieval(args.out)
    if not metrics:
        print("Нет проверенных вопросов (verified=True). Запустите: python gold_set.py seed")
        return 1
    print("=== Метрики ретривала ===")
    for key, value in metrics.items():
        print(f"  {key:12} {value:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
