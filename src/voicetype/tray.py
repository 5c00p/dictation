"""System tray icon, menu and status indication (spec 3.4).

The icon colour reflects the controller state (idle / recording / processing).
The menu exposes status, language choice, input-device choice, opening the
config, and exit.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pystray
from PIL import Image, ImageDraw
from pystray import Menu, MenuItem

from .audio.capture import list_input_devices

if TYPE_CHECKING:
    from .controller import Controller, State

logger = logging.getLogger(__name__)

_STATE_COLORS = {
    "idle": (90, 90, 95),
    "recording": (210, 60, 60),
    "processing": (220, 170, 40),
    "error": (120, 120, 120),
}


def _make_icon(color: tuple[int, int, int]) -> Image.Image:
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((6, 6, size - 6, size - 6), fill=color)
    # A small white "microphone" glyph.
    draw.rounded_rectangle((27, 16, 37, 38), radius=5, fill=(255, 255, 255))
    draw.arc((23, 30, 41, 46), start=0, end=180, fill=(255, 255, 255), width=3)
    draw.line((32, 46, 32, 52), fill=(255, 255, 255), width=3)
    return img


class Tray:
    def __init__(self, controller: "Controller") -> None:
        self.controller = controller
        self._icons = {name: _make_icon(c) for name, c in _STATE_COLORS.items()}
        self.icon = pystray.Icon(
            "voicetype",
            icon=self._icons["idle"],
            title="VoiceType — idle",
            menu=self._build_menu(),
        )

    # -- menu -------------------------------------------------------------
    def _build_menu(self) -> Menu:
        def lang_item(code: str, label: str) -> MenuItem:
            return MenuItem(
                label,
                lambda icon, item: self.controller.set_language(code),
                checked=lambda item, c=code: self.controller.language == c,
                radio=True,
            )

        language_menu = Menu(
            lang_item("ru", "Русский"),
            lang_item("be", "Беларуская"),
            lang_item("auto", "Авто (RU/BE)"),
        )

        return Menu(
            MenuItem(lambda item: self._status_text(), None, enabled=False),
            Menu.SEPARATOR,
            MenuItem("Язык", language_menu),
            MenuItem("Устройство ввода", self._device_menu()),
            Menu.SEPARATOR,
            MenuItem(lambda item: self._hotkeys_text(), None, enabled=False),
            MenuItem("Открыть настройки (config.toml)", self._open_config),
            MenuItem("Перезагрузить настройки", self._reload_config),
            MenuItem("Выход", self._quit),
        )

    def _hotkeys_text(self) -> str:
        hk = self.controller.config.hotkeys
        return f"PTT: {hk.push_to_talk} | Toggle: {hk.toggle}"

    def _select_device(self, name: str):
        # pystray actions must accept exactly (icon, item); bind name here.
        return lambda icon, item: self.controller.set_input_device(name)

    def _device_checked(self, name: str):
        return lambda item: self.controller.input_device_name == name

    def _device_menu(self) -> Menu:
        items = [
            MenuItem(
                "По умолчанию",
                self._select_device(""),
                checked=self._device_checked(""),
                radio=True,
            )
        ]
        for _idx, name in list_input_devices():
            items.append(
                MenuItem(
                    name[:40],
                    self._select_device(name),
                    checked=self._device_checked(name),
                    radio=True,
                )
            )
        return Menu(*items)

    def _status_text(self) -> str:
        state = self.controller.state_name
        lang = self.controller.language.upper()
        return f"Статус: {state} | Язык: {lang}"

    # -- actions ----------------------------------------------------------
    def _open_config(self, icon=None, item=None) -> None:  # noqa: ANN001
        path = str(self.controller.config_path)
        try:
            if sys.platform == "win32":
                os.startfile(path)  # noqa: SCS110
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            logger.error("Could not open config %s: %s", path, exc)

    def _reload_config(self, icon=None, item=None) -> None:  # noqa: ANN001
        try:
            self.controller.reload()
            self.icon.update_menu()
        except Exception as exc:
            logger.error("Reload failed: %s", exc)

    def _quit(self, icon=None, item=None) -> None:  # noqa: ANN001
        logger.info("Quit requested from tray")
        self.controller.shutdown()
        self.icon.stop()

    # -- state indication -------------------------------------------------
    def set_state(self, state_name: str) -> None:
        icon_img = self._icons.get(state_name, self._icons["idle"])
        self.icon.icon = icon_img
        self.icon.title = f"VoiceType — {state_name}"
        # Refresh menu so the dynamic status line / checkmarks update.
        try:
            self.icon.update_menu()
        except Exception:
            pass

    def run(self) -> None:
        """Blocking — must be called on the main thread on Windows."""
        self.icon.run()
