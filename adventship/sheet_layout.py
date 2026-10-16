"""Pure layout logic: day -> cell address from the sheet's CSV export."""

from __future__ import annotations

import re

from adventship.ctx import EXIT_CONFIG, ShipError


def column_letters(index: int) -> str:
    """0-based column index -> A1 letters (0 -> A, 25 -> Z, 26 -> AA, 35 -> AJ)."""
    if index < 0:
        raise ValueError(index)
    letters = ""
    n = index + 1
    while n:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def resolve_cell(rows: list[list[str]], day: int, slot_row: int, owner: str) -> str:
    """Header `День N ...` in row 1 gives the column; `slot_row` (1-based) the row."""
    if not owner.strip():
        raise ShipError(
            "не задан владелец строки (SHEET_SLOT-проверка)",
            exit_code=EXIT_CONFIG,
            hint="Задай SHEET_OWNER в .env или корректный GITHUB_URL.",
        )
    header = rows[0] if rows else []
    pattern = re.compile(rf"^День\s+{day}\b")
    columns = [i for i, cell in enumerate(header) if pattern.match(cell.strip())]
    if not columns:
        raise ShipError(f"в шапке таблицы нет колонки «День {day}»")
    if len(columns) > 1:
        raise ShipError(f"в шапке таблицы несколько колонок «День {day}»: нужна ровно одна")
    if not 1 <= slot_row <= len(rows):
        raise ShipError(f"слот {slot_row} вне таблицы (строк: {len(rows)})")
    row = rows[slot_row - 1]
    name = row[0] if row else ""
    if owner.strip().lower() not in name.lower():
        raise ShipError(
            f"в строке {slot_row} ячейка A не содержит владельца «{owner}»",
            hint="Проверь SHEET_SLOT и SHEET_OWNER; в чужую строку писать нельзя.",
        )
    return f"{column_letters(columns[0])}{slot_row}"
