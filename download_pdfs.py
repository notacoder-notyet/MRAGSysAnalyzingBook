"""
Скачивание PDF-презентаций уроков с публичного Яндекс Диска.

На диске лежат папки вида "Lesson 1", "Lesson 2", ... В каждой — PDF с презентацией.
Имена PDF между уроками повторяются, поэтому файл переименовывается в
`lesson_<N>.pdf` — так работает `extract_lesson_number` из `pars_pdf.py`.

Использование:
    python download_pdfs.py                 # скачать все недостающие
    python download_pdfs.py --dry-run       # только показать план
    python download_pdfs.py --limit 5       # первые 5 уроков
    python download_pdfs.py --overwrite     # перекачать заново
    python download_pdfs.py --dest data/raw # другая папка
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import httpx

from config import (
    RAW_PDF_DIR,
    YANDEX_API_BASE,
    YANDEX_DISK_PUBLIC_KEY,
    YANDEX_MAX_RETRIES,
    YANDEX_PAGE_LIMIT,
    YANDEX_SKIP_FILENAME_MARKERS,
    YANDEX_TIMEOUT,
)

# Расширения файлов, которые считаем презентациями
PDF_EXTENSIONS = {".pdf"}


def is_skipped(filename: str) -> bool:
    """
    Проверяет, нужно ли исключить файл (домашка, учебный кейс и т.п.).

    Args:
        filename: Имя файла на диске.

    Returns:
        True, если файл не нужен для RAG-индексации.
    """
    lowered = filename.lower()
    return any(marker in lowered for marker in YANDEX_SKIP_FILENAME_MARKERS)


class YandexDiskClient:
    """Клиент публичного API Яндекс Диска (только чтение)."""

    def __init__(
        self,
        public_key: str = YANDEX_DISK_PUBLIC_KEY,
        timeout: float = YANDEX_TIMEOUT,
    ) -> None:
        """
        Args:
            public_key: Ссылка на публичный диск (вида https://disk.yandex.ru/d/XXXX).
            timeout: Таймаут HTTP-запросов в секундах.
        """
        self.public_key = public_key
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "MRAGSysAnalyzingBook/1.0"},
        )

    def close(self) -> None:
        """Закрывает HTTP-соединения."""
        self._client.close()

    def __enter__(self) -> "YandexDiskClient":
        """Поддержка context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Закрывает соединения при выходе из context manager."""
        self.close()

    def list_dir(self, path: str = "/") -> list[dict[str, Any]]:
        """
        Возвращает содержимое папки на Яндекс Диске.

        Args:
            path: Путь внутри публичного диска (например "/" или "/Lesson 1").

        Returns:
            Список элементов (файлов и папок).

        Raises:
            httpx.HTTPStatusError: Если API вернул ошибку.
        """
        items: list[dict[str, Any]] = []
        offset = 0

        # API отдаёт результаты постранично — обходим все страницы
        while True:
            response = self._client.get(
                f"{YANDEX_API_BASE}/resources",
                params={
                    "public_key": self.public_key,
                    "path": path,
                    "limit": YANDEX_PAGE_LIMIT,
                    "offset": offset,
                },
            )
            response.raise_for_status()
            embedded = (response.json() or {}).get("_embedded") or {}
            page_items: list[dict[str, Any]] = embedded.get("items") or []
            items.extend(page_items)

            total = int(embedded.get("total") or len(items))
            offset += len(page_items)
            if not page_items or offset >= total:
                break

        return items

    def get_download_href(self, path: str) -> str:
        """
        Получает временную прямую ссылку на скачивание файла.

        Args:
            path: Путь к файлу на диске (например "/Lesson 1/file.pdf").

        Returns:
            URL для скачивания.

        Raises:
            httpx.HTTPStatusError: Если API вернул ошибку.
        """
        response = self._client.get(
            f"{YANDEX_API_BASE}/resources/download",
            params={"public_key": self.public_key, "path": path},
        )
        response.raise_for_status()
        return str(response.json()["href"])

    def download(self, remote_path: str, dest_path: Path) -> int:
        """
        Скачивает файл на локальный диск.

        Args:
            remote_path: Путь к файлу на Яндекс Диске.
            dest_path: Локальный путь для сохранения.

        Returns:
            Размер скачанного файла в байтах.

        Raises:
            RuntimeError: Если файл не удалось скачать после всех попыток.
        """
        last_error: Exception | None = None
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        for attempt in range(1, YANDEX_MAX_RETRIES + 1):
            try:
                href = self.get_download_href(remote_path)
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Пишем во временный файл и переименовываем только после успеха —
                # при обрыве связи не остаётся «полускачанных» PDF
                bytes_written = 0
                with self._client.stream("GET", href) as stream:
                    stream.raise_for_status()
                    with open(tmp_path, "wb") as target:
                        for chunk in stream.iter_bytes(chunk_size=1 << 16):
                            target.write(chunk)
                            bytes_written += len(chunk)

                tmp_path.replace(dest_path)
                return bytes_written

            except Exception as error:  # noqa: BLE001 — ловим всё, чтобы повторить
                last_error = error
                tmp_path.unlink(missing_ok=True)  # чистим недокачанный файл

                if attempt < YANDEX_MAX_RETRIES:
                    delay = 2**attempt  # экспоненциальная задержка: 2с, 4с, 8с
                    print(
                        f"\n    ! попытка {attempt}/{YANDEX_MAX_RETRIES} не удалась "
                        f"({error}), повтор через {delay}с"
                    )
                    time.sleep(delay)

        raise RuntimeError(f"Не удалось скачать {remote_path}: {last_error}")


def extract_lesson_number(name: str) -> int | None:
    """
    Извлекает номер урока из имени папки вида "Lesson 12".

    Args:
        name: Имя папки или файла.

    Returns:
        Номер урока или None, если номер не распознан.
    """
    match = re.search(r"lesson[\s_-]?(\d+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def build_target_name(folder_name: str, file_index: int, file_total: int) -> str:
    """
    Формирует локальное имя файла: `lesson_<N>.pdf` или `lesson_<N>_<M>.pdf`.

    Номер урока берём из имени папки: имена самих PDF между уроками повторяются.
    Если в папке несколько PDF, добавляем порядковый номер, чтобы ничего не потерять.

    Args:
        folder_name: Имя папки урока на диске ("Lesson 3", "Lesson39").
        file_index: Порядковый номер PDF в папке (начиная с 1).
        file_total: Сколько всего PDF в папке.

    Returns:
        Имя файла для сохранения.
    """
    lesson = extract_lesson_number(folder_name)
    if lesson is None:
        # Номер не распознан — сохраняем как есть, чтобы не потерять файл
        return f"{file_index}_{folder_name}.pdf"
    if file_total > 1:
        # Несколько PDF в уроке — нумеруем, чтобы избежать перезаписи
        return f"lesson_{lesson}_{file_index}.pdf"
    return f"lesson_{lesson}.pdf"


def find_pdfs(
    client: YandexDiskClient, root: str = "/"
) -> Iterator[tuple[str, str, str, int]]:
    """
    Обходит дерево папок и выдаёт только полезные PDF-файлы.

    Рекурсивный обход: если PDF лежит во вложенной папке (например Lesson13/Презентация/),
    он тоже будет найден. Ненужные файлы (домашки, учебные кейсы) отсеиваются.

    Args:
        client: Клиент Яндекс Диска.
        root: Корень обхода.

    Yields:
        Кортежи (remote_dir, target_name, remote_name, size_bytes),
        где remote_dir — полный путь к папке на диске (может быть вложенной).
    """
    stack: list[str] = [root]

    while stack:
        current = stack.pop()
        items = client.list_dir(current)

        # Отделяем PDF от подпапок; служебные файлы и папки отсеиваем
        pdfs: list[dict[str, Any]] = []
        subdirs: list[str] = []
        for item in items:
            item_name = str(item.get("name", ""))
            if item.get("type") == "dir":
                # Папку «Домашнее задание» не обходим целиком
                if not is_skipped(item_name):
                    subdirs.append(str(item["path"]))
            elif item.get("type") == "file":
                if Path(item_name).suffix.lower() not in PDF_EXTENSIONS:
                    continue
                if is_skipped(item_name):
                    continue
                pdfs.append(item)

        # Имя папки урока для извлечения номера: берём последний значимый сегмент
        folder_name = _lesson_folder_name(current)

        for index, item in enumerate(pdfs, start=1):
            yield (
                current,
                build_target_name(folder_name, index, len(pdfs)),
                str(item["name"]),
                int(item.get("size") or 0),
            )

        stack.extend(subdirs)


def _lesson_folder_name(remote_dir: str) -> str:
    """
    Определяет имя папки урока по удалённому пути.

    Для вложенных папок (Lesson13/Презентация) поднимаемся вверх до сегмента,
    в котором распознаётся номер урока.

    Args:
        remote_dir: Путь к папке на диске.

    Returns:
        Имя папки урока ("Lesson 3") либо последний сегмент пути.
    """
    segments = [s for s in remote_dir.split("/") if s]
    for segment in reversed(segments):
        if extract_lesson_number(segment) is not None:
            return segment
    return segments[-1] if segments else "root"


def count_filtered_pdfs(client: YandexDiskClient, root: str = "/") -> int:
    """
    Считает PDF, отсеянные как ненужные (домашки, учебные кейсы).

    Args:
        client: Клиент Яндекс Диска.
        root: Корень обхода.

    Returns:
        Количество отсеянных файлов.
    """
    stack: list[str] = [root]
    count = 0

    while stack:
        current = stack.pop()
        for item in client.list_dir(current):
            item_name = str(item.get("name", ""))
            if item.get("type") == "dir":
                stack.append(str(item["path"]))
            elif (
                item.get("type") == "file"
                and Path(item_name).suffix.lower() in PDF_EXTENSIONS
                and is_skipped(item_name)
            ):
                count += 1

    return count


def human_size(num_bytes: int) -> str:
    """
    Форматирует байты в читаемый вид.

    Args:
        num_bytes: Количество байт.

    Returns:
        Строка вида "9.5 MB".
    """
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def main() -> int:
    """Точка входа CLI: обходит диск и скачивает PDF.

    Returns:
        0 — успех, 1 — были ошибки скачивания или доступа к диску.
    """
    parser = argparse.ArgumentParser(
        description="Скачивание PDF уроков с публичного Яндекс Диска"
    )
    parser.add_argument(
        "--dest",
        default=str(RAW_PDF_DIR),
        help=f"Папка для сохранения PDF (по умолчанию {RAW_PDF_DIR})",
    )
    parser.add_argument(
        "--public-key",
        default=YANDEX_DISK_PUBLIC_KEY,
        help="Ссылка на публичный Яндекс Диск",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Скачать только первые N PDF (0 — все)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перекачать файлы, которые уже скачаны",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать список файлов, ничего не скачивать",
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Источник: {args.public_key}")
    print(f"Приёмник: {dest_dir}")
    print(f"Режим:    {'DRY-RUN (без скачивания)' if args.dry_run else 'скачивание'}")
    print("-" * 60)

    downloaded = 0
    skipped = 0
    failed = 0
    filtered = 0
    total_bytes = 0

    with YandexDiskClient(args.public_key) as client:
        try:
            # Считаем отсеянные файлы (домашки, учебные кейсы) для отчёта
            filtered = count_filtered_pdfs(client)

            for remote_dir, target_name, remote_name, size in find_pdfs(client):
                if args.limit and downloaded >= args.limit:
                    print(f"\nДостигнут лимит --limit={args.limit}, останавливаюсь.")
                    break

                # remote_dir — полный путь к папке (может быть вложенной),
                # поэтому подставляем его как есть, а не собираем из имени папки
                remote_path = f"{remote_dir.rstrip('/')}/{remote_name}"
                target_path = dest_dir / target_name
                exists = target_path.exists() and target_path.stat().st_size > 0

                if exists and not args.overwrite:
                    print(f"  ⊘ пропущен (уже скачан): {target_name}")
                    skipped += 1
                    continue

                if args.dry_run:
                    marker = "перекачать" if exists else "скачать"
                    print(f"  → {marker}: {target_name}  [{human_size(size)}]  ← {remote_name}")
                    downloaded += 1
                    continue

                print(
                    f"  ↓ {target_name}  [{human_size(size)}]",
                    end="",
                    flush=True,
                )
                try:
                    written = client.download(remote_path, target_path)
                    total_bytes += written
                    downloaded += 1
                    print(f"  ✓ {human_size(written)}")
                except (RuntimeError, httpx.HTTPError) as error:
                    failed += 1
                    print(f"\n  ✗ ОШИБКА: {error}")

        except httpx.HTTPStatusError as error:
            print(f"\nОшибка доступа к диску: {error}")
            print("Проверьте, что ссылка публичная и открыта для чтения.")
            return 1
        except httpx.RequestError as error:
            print(f"\nСетевая ошибка: {error}")
            return 1

    print("-" * 60)
    print(f"Скачано:      {downloaded}")
    print(f"Пропущено:    {skipped}")
    print(f"Отсеяно:      {filtered}  (домашки / учебные кейсы)")
    if failed:
        print(f"Ошибок:       {failed}")
    if total_bytes:
        print(f"Всего данных: {human_size(total_bytes)}")

    if failed:
        print("Завершено с ошибками.")
        return 1

    print("Готово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())