"""
Отчёт по качеству retrieval: какие вопросы проваливаются и почему.

Показывает три вещи:
  1. Hit@k и MRR по gold-набору;
  2. провальные вопросы с разбивкой по типу/сложности;
  3. долю «мусорного» текста на странице — главный предиктор провала.

Мусорным считается текст, который pdfplumber собрал из таблиц и схем
по колонкам: односимвольные токены и бессмысленные обрывки вроде
"Н И а м п е р е а т в д л л е и н н". Такие страницы дают плохие
эмбеддинги, и точный ответ по ним не находится.

Запуск:
    python retrieval_report.py            # top_k=10
    python retrieval_report.py --top-k 20
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

from config import DEFAULT_TOP_K
from embeddings import embed_texts, load_embedding_model
from gold_set import GoldItem, load_gold
from vector_store import create_vector_store

# Порог доли мусорных токенов, выше которого страницу считаем «сломанной»
GARBLE_THRESHOLD = 0.15

VOWELS = set("АЕЁИОУЫЭЮЯаеёиоуыэюяAEIOUYaeiouy")


def garble_ratio(text: str) -> float:
    """
    Доля «мусорных» токенов — признак плохо извлечённой таблицы или схемы.

    Мусор: токены длиной 1 символ, а также 2-3 символа без гласных
    (типично при чтении текста по колонкам).

    Args:
        text: Текст страницы.

    Returns:
        Доля мусорных токенов от 0.0 до 1.0.
    """
    tokens = [t for t in re.split(r"\s+", text or "") if t]
    if not tokens:
        return 0.0

    junk = 0
    for token in tokens:
        letters = re.sub(r"[^А-Яа-яЁёA-Za-z]", "", token)
        if len(letters) <= 1:
            junk += 1
        elif len(letters) <= 3 and not (set(letters) & VOWELS):
            junk += 1
    return junk / len(tokens)


def page_text_from_store(store, lesson: int, page: int) -> str:
    """Собирает текст всех чанков указанной страницы из векторного хранилища."""
    got = store.collection.get(
        where={"$and": [{"lesson": lesson}, {"page": page}]},
        include=["documents"],
    )
    return "\n".join(got.get("documents") or [])


def build_report(top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
    """
    Прогоняет gold-набор по индексу и собирает детальную таблицу результатов.

    Args:
        top_k: Размер выдачи для оценки попадания.

    Returns:
        DataFrame с колонками rank, ref, type, difficulty, garble, question.
    """
    gold = load_gold()
    gold = gold[gold["verified"].astype(str).str.lower().isin({"true", "1", "yes"})]
    if gold.empty:
        return pd.DataFrame()

    store = create_vector_store()
    model = load_embedding_model()
    vectors = embed_texts(
        gold["question"].astype(str).tolist(), model=model, show_progress=False, role="query"
    )

    rows: list[dict[str, object]] = []
    for i, (_, row) in enumerate(gold.iterrows()):
        refs = set(GoldItem.parse_refs(row["expected_refs"]))
        hits = store.search(vectors[i], top_k=top_k)
        found = [(h["metadata"].get("lesson"), h["metadata"].get("page")) for h in hits]
        rank = next((r for r, ref in enumerate(found, 1) if ref in refs), 0)

        lesson, page = next(iter(refs))
        rows.append(
            {
                "rank": rank,
                "ref": row["expected_refs"],
                "type": row["question_type"],
                "difficulty": row["difficulty"],
                "garble": garble_ratio(page_text_from_store(store, lesson, page)),
                "question": str(row["question"]),
            }
        )
    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame, top_k: int) -> None:
    """Печатает сводку отчёта."""
    total = len(df)
    found = int((df["rank"] > 0).sum())
    print(f"Вопросов: {total} | top_k={top_k}")
    print(f"Найдено: {found} ({found / total * 100:.1f}%) | провалов: {total - found}\n")

    failed = df[df["rank"] == 0]
    print("=== Провалы по типу вопроса ===")
    print(f"{'тип':<12}{'провалов':>10}{'всего':>8}{'доля':>8}")
    for qtype in sorted(df["type"].dropna().unique()):
        sub = df[df["type"] == qtype]
        bad = int((sub["rank"] == 0).sum())
        print(f"{qtype:<12}{bad:>10}{len(sub):>8}{bad / len(sub) * 100:>7.0f}%")

    print(f"\n=== Мусорность текста страницы (порог {GARBLE_THRESHOLD:.0%}) ===")
    ok = df[df["rank"] > 0]
    print(f"  успех : средняя {ok['garble'].mean():.3f}, выше порога {int((ok['garble'] > GARBLE_THRESHOLD).sum())}/{len(ok)}")
    if len(failed):
        print(f"  провал: средняя {failed['garble'].mean():.3f}, выше порога {int((failed['garble'] > GARBLE_THRESHOLD).sum())}/{len(failed)}")

    if len(failed):
        print("\n=== Провальные вопросы ===")
        for _, r in failed.sort_values("garble", ascending=False).iterrows():
            flag = " [МУСОРНЫЙ ТЕКСТ]" if r["garble"] > GARBLE_THRESHOLD else ""
            print(f"  {r['ref']:6} мусор={r['garble']:.2f} [{r['type']:9}] {r['question'][:60]}{flag}")


def main() -> int:
    """CLI-обёртка отчёта."""
    parser = argparse.ArgumentParser(description="Отчёт по качеству retrieval")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Размер выдачи")
    args = parser.parse_args()

    df = build_report(top_k=args.top_k)
    if df.empty:
        print("Нет проверенных вопросов. Сначала: python gold_set.py generate")
        return 1

    print_report(df, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
