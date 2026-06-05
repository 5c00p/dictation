"""Entry point: initialise logging, config, controller, hotkeys and tray.

The tray icon owns the main thread (required on Windows); the global hotkey
listener and all transcription work run on background threads.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .autostart import apply_autostart
from .config import config_path, load_config
from .controller import Controller
from .tray import Tray


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voicetype", description="Voice dictation")
    parser.add_argument("--config", help="Path to config.toml", default=None)
    args = parser.parse_args(argv)

    path = config_path(args.config)
    config = load_config(path)
    _setup_logging(config.app.log_level)
    log = logging.getLogger("voicetype")
    log.info("Starting VoiceType")

    apply_autostart(config.app.autostart)

    controller = Controller(config, path)

    tray = Tray(controller)
    controller.set_tray(tray)

    # Load the model up front so the first dictation isn't penalised.
    log.info("Loading speech model (this can take a while on first run)...")
    controller.warmup()
    log.info("Ready.")

    controller.start_hotkeys()
    try:
        tray.run()  # blocks on the main thread until quit
    finally:
        controller.shutdown()
    log.info("VoiceType stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
