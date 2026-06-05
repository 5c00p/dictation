"""Manual smoke test: build all components without loading the model or UI loop."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

from voicetype.audio.capture import list_input_devices
from voicetype.config import Config
from voicetype.controller import Controller
from voicetype.hotkeys import HotkeyManager
from voicetype.inject import create_injector
from voicetype.stt import create_provider
from voicetype.tray import Tray

cfg = Config()
c = Controller(cfg, Path("config.toml"))
t = Tray(c)
c.set_tray(t)
hk = HotkeyManager(
    cfg.hotkeys.push_to_talk,
    cfg.hotkeys.toggle,
    cfg.hotkeys.switch_language,
    c.on_ptt_start,
    c.on_ptt_stop,
    c.on_toggle,
    c.switch_language,
)
print("WIRING OK")
print("input devices:", len(list_input_devices()))
print("injector:", type(create_injector(cfg.inject)).__name__)
print("provider:", type(create_provider(cfg.stt)).__name__)
c.set_language("be")
print("lang ->", c.language)
c.switch_language()
print("switch ->", c.language)
print("tray state set:", t.set_state("recording") or "ok")
