import pytest

from adventship.ctx import ShipError
from adventship.sheet_layout import column_letters, resolve_cell


def make_rows(day_col=35, width=40, owner_row=3, owner="Fake Owner (@fakeowner)"):
    header = [""] + [f"День {i} (01.01)" for i in range(1, width) if i != 35]
    header += [""] * (width - len(header))
    header[day_col] = "День 35 (16.10)"
    rows = [header]
    for r in range(2, 6):
        rows.append([owner if r == owner_row else f"Other {r} (@other{r})"] + [""] * (width - 1))
    return rows


@pytest.mark.parametrize(
    ("index", "letters"),
    [
        (0, "A"),
        (25, "Z"),
        (26, "AA"),
        (35, "AJ"),
        (51, "AZ"),
        (52, "BA"),
        (701, "ZZ"),
        (702, "AAA"),
    ],
)
def test_column_letters(index, letters):
    assert column_letters(index) == letters


def test_resolve_cell():
    assert resolve_cell(make_rows(), 35, 3, "FAKEOWNER") == "AJ3"


def test_resolve_cell_edge_columns():
    assert resolve_cell(make_rows(day_col=26, width=30), 35, 3, "fakeowner") == "AA3"
    assert resolve_cell(make_rows(day_col=52, width=60), 35, 3, "fakeowner") == "BA3"


def test_day_must_match_whole_number():
    rows = make_rows()
    rows[0][35] = "День 350 (01.01)"
    with pytest.raises(ShipError, match="нет колонки «День 35»"):
        resolve_cell(rows, 35, 3, "fakeowner")


def test_missing_header():
    with pytest.raises(ShipError, match="нет колонки «День 99»"):
        resolve_cell(make_rows(), 99, 3, "fakeowner")


def test_duplicate_header():
    rows = make_rows()
    rows[0][3] = "День 35 (dup)"
    with pytest.raises(ShipError, match="несколько колонок"):
        resolve_cell(rows, 35, 3, "fakeowner")


@pytest.mark.parametrize("slot", [0, 6, 99])
def test_slot_out_of_range(slot):
    with pytest.raises(ShipError, match="вне таблицы"):
        resolve_cell(make_rows(), 35, slot, "fakeowner")


def test_wrong_owner_row_rejected():
    with pytest.raises(ShipError, match="не содержит владельца"):
        resolve_cell(make_rows(), 35, 2, "fakeowner")


def test_empty_owner_rejected():
    with pytest.raises(ShipError, match="не задан владелец"):
        resolve_cell(make_rows(), 35, 3, "  ")
