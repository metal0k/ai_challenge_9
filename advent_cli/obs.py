"""Управление OBS Studio через obs-websocket.

Проверено на OBS 32.2.2 с obs-websocket 5.7.4. Протокол гарантирует, что
`StopRecord` возвращает `outputPath` — путь к сохранённому файлу.

Настройки пользователя не портим: своя сцена, а активная сцена и папка
записи возвращаются на место после прогона.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from advent_core.errors import AdventError

SCENE_NAME = "AI Advent Demo"
SOURCE_NAME = "Terminal Capture"

# priority=2 — сопоставление окна по исполняемому файлу. Заголовок Windows
# Terminal меняется от текущей команды, а окна может вообще не быть в момент
# создания источника, поэтому матчим по exe.
WINDOW_SETTINGS = {
    "window": "::WindowsTerminal.exe",
    "priority": 2,
    "method": 2,  # Windows 10 (WGC) — корректно снимает Windows Terminal
    "client_area": True,
    "cursor": True,
}


def connect():
    """Подключение к obs-websocket. OBS должен быть запущен."""
    try:
        import obsws_python as obs
    except ImportError as exc:
        raise AdventError("Не установлен obsws-python. Выполни: uv sync") from exc

    host = os.getenv("OBS_WS_HOST", "localhost")
    port = int(os.getenv("OBS_WS_PORT", "4455"))
    password = os.getenv("OBS_WS_PASSWORD", "")

    try:
        return obs.ReqClient(host=host, port=port, password=password, timeout=5)
    except Exception as exc:
        raise AdventError(
            f"Не удалось подключиться к OBS на {host}:{port}.",
            hint=(
                "Запусти OBS Studio и проверь Инструменты → Настройки сервера WebSocket. "
                "Пароль должен совпадать с OBS_WS_PASSWORD в .env."
            ),
        ) from exc


def ensure_scene(client, prefer_title: str | None = None) -> None:
    """Создаёт сцену с захватом окна терминала и привязывает её к окну.

    Настройки источника переприменяются на каждом запуске: источник, созданный
    когда окна ещё не было, остаётся пустым навсегда — OBS не перепривязывается
    сам. Именно так получается чёрное видео, которое обнаруживается уже после
    записи.
    """
    scenes = {s["sceneName"] for s in client.get_scene_list().scenes}
    if SCENE_NAME not in scenes:
        client.create_scene(SCENE_NAME)

    items = {item["sourceName"] for item in client.get_scene_item_list(SCENE_NAME).scene_items}
    if SOURCE_NAME not in items:
        client.create_input(SCENE_NAME, SOURCE_NAME, "window_capture", WINDOW_SETTINGS, True)

    settings = dict(WINDOW_SETTINGS)
    exact = _enumerate_window(client, prefer_title=prefer_title)
    if exact:
        # Точная строка "title:class:exe" от самого OBS надёжнее шаблона.
        settings["window"] = exact
    client.set_input_settings(SOURCE_NAME, settings, True)
    time.sleep(1.0)  # источнику нужен кадр, чтобы захватить окно
    fit_to_canvas(client)


def fit_to_canvas(client) -> None:
    """Растягивает захват окна на весь холст с сохранением пропорций.

    Без этого окно ложится в кадр 1:1 и на холсте 1920x1080 занимает угол, а
    остальное остаётся чёрным полем — текст в терминале выходит нечитаемым.
    SCALE_INNER вписывает источник целиком, не обрезая содержимое.
    """
    try:
        video = client.get_video_settings()
        width, height = video.base_width, video.base_height
        item_id = client.get_scene_item_id(SCENE_NAME, SOURCE_NAME).scene_item_id
        client.set_scene_item_transform(
            SCENE_NAME,
            item_id,
            {
                "boundsType": "OBS_BOUNDS_SCALE_INNER",
                "boundsWidth": float(width),
                "boundsHeight": float(height),
                "boundsAlignment": 0,  # по центру
                "positionX": 0.0,
                "positionY": 0.0,
                "alignment": 5,  # левый верхний угол как точка отсчёта
                "cropLeft": 0,
                "cropRight": 0,
                "cropTop": 0,
                "cropBottom": 0,
            },
        )
    except Exception as exc:  # noqa: BLE001 — кадр без растяжки лучше, чем отказ
        print(f"ВНИМАНИЕ: не удалось вписать источник в холст: {exc}", file=sys.stderr)


def _enumerate_window(
    client, needle: str = "WindowsTerminal.exe", prefer_title: str | None = None
) -> str | None:
    """Берёт точную строку окна из списка, который отдаёт сам OBS.

    Терминалов у пользователя обычно несколько, и первый попавшийся — не
    обязательно наш. Поэтому сначала ищем окно с известным заголовком демо,
    и только потом откатываемся на любое.
    """
    try:
        items = client.get_input_properties_list_property_items(
            SOURCE_NAME, "window"
        ).property_items
    except Exception:
        return None

    matches = [i["itemValue"] for i in items if needle in i.get("itemValue", "")]
    if not matches:
        return None
    if prefer_title:
        titled = [m for m in matches if prefer_title.lower() in m.lower()]
        if titled:
            return titled[0]
    return matches[0]


def verify_capture(client, min_bright_ratio: float = 0.005) -> None:
    """Убеждается, что источник реально что-то снимает, а не чёрный экран.

    Проверка до старта записи: иначе о пустом кадре узнаёшь только когда
    видео уже записано и время потрачено.
    """
    try:
        shot = client.get_source_screenshot(SOURCE_NAME, "png", 480, 270, 60)
        raw = shot.image_data.split(",", 1)[-1]
        data = base64.b64decode(raw)
    except Exception as exc:
        raise AdventError(f"Не удалось снять пробный кадр источника: {exc}") from exc

    if not _has_content(data, min_bright_ratio):
        raise AdventError(
            "Источник захвата даёт пустой кадр — видео получится чёрным.",
            hint=(
                "Окно Windows Terminal должно быть открыто и не свёрнуто. "
                f"Проверь источник «{SOURCE_NAME}» в сцене «{SCENE_NAME}» в OBS."
            ),
        )


def _has_content(png_bytes: bytes, min_ratio: float) -> bool:
    """Доля нечёрных пикселей в кадре."""
    from PIL import Image

    image = Image.open(io.BytesIO(png_bytes)).convert("L")
    pixels = list(image.getdata())
    if not pixels:
        return False
    bright = sum(1 for value in pixels if value > 24)
    return bright / len(pixels) >= min_ratio


def start_recording(client) -> None:
    if client.get_record_status().output_active:
        raise AdventError(
            "OBS уже пишет.",
            hint="Останови текущую запись вручную, чтобы не испортить чужой файл.",
        )
    client.start_record()


def stop_recording(client) -> Path:
    """Останавливает запись и возвращает путь к готовому файлу.

    OBS дописывает контейнер уже после ответа на StopRecord, поэтому ждём,
    пока файл перестанет расти — иначе перемещать его рано.
    """
    response = client.stop_record()
    raw = getattr(response, "output_path", None)
    if not raw:
        raise AdventError("OBS не вернул путь к файлу записи.")

    path = Path(raw)
    _wait_until_stable(path)
    return path


def _wait_until_stable(path: Path, timeout: float = 120.0) -> None:
    """Ждёт, пока OBS допишет контейнер.

    Молчаливый выход по таймауту означал бы, что дальше файл трогают, пока
    OBS ещё держит его открытым — на Windows это PermissionError голым
    traceback вместо человеческого сообщения.
    """
    deadline = time.monotonic() + timeout
    previous = -1
    while time.monotonic() < deadline:
        if path.exists():
            size = path.stat().st_size
            if size > 0 and size == previous:
                return
            previous = size
        time.sleep(0.5)

    raise AdventError(
        f"OBS не закончил запись файла за {timeout:.0f} с: {path}",
        hint="Файл всё ещё растёт или заблокирован. Проверь состояние OBS и повтори.",
    )


@contextmanager
def program_scene(client, scene: str):
    """Временно делает сцену активной и возвращает прежнюю обратно."""
    previous = client.get_current_program_scene().current_program_scene_name
    client.set_current_program_scene(scene)
    try:
        yield
    finally:
        _restore(
            "сцена",
            previous,
            lambda: client.set_current_program_scene(previous),
            lambda: client.get_current_program_scene().current_program_scene_name,
        )


@contextmanager
def record_directory(client, directory: Path):
    """Временно переключает папку записи OBS и возвращает прежнюю.

    Активный профиль пользователя может писать куда угодно — в том числе в
    каталог чужого проекта. Своя временная папка гарантирует, что после
    переноса файла ничего постороннего не остаётся.
    """
    directory.mkdir(parents=True, exist_ok=True)
    try:
        previous = client.get_record_directory().record_directory
    except Exception:
        previous = None

    try:
        client.set_record_directory(str(directory))
    except Exception as exc:
        raise AdventError(
            f"Не удалось переключить папку записи OBS на {directory}: {exc}"
        ) from exc

    try:
        yield
    finally:
        if previous:
            _restore(
                "папка записи",
                previous,
                lambda: client.set_record_directory(previous),
                lambda: client.get_record_directory().record_directory,
            )


def _restore(what: str, expected: str, apply, read) -> None:
    """Возвращает настройку OBS и убеждается, что она действительно вернулась.

    Молчаливый suppress здесь уже подводил: после падения записи у пользователя
    осталась чужая активная сцена и чужая папка вывода, и заметить это можно
    было только вручную. Поэтому — повтор и явное предупреждение.
    """
    for attempt in range(3):
        try:
            apply()
            if read() == expected:
                return
        except Exception:  # noqa: BLE001 — OBS может быть занят, пробуем ещё
            pass
        time.sleep(0.4 * (attempt + 1))

    print(
        f"ВНИМАНИЕ: не удалось вернуть {what} в OBS на {expected!r} — поправь вручную",
        file=sys.stderr,
    )
