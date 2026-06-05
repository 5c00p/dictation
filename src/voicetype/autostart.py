"""Windows autostart via the HKCU ``Run`` registry key (spec 4).

Enabled/disabled from the ``[app] autostart`` config flag at startup. Uses the
current interpreter + ``-m voicetype`` when running from source, or the frozen
executable path when packaged with PyInstaller.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_VALUE_NAME = "VoiceType"


def _launch_command() -> str:
    if getattr(sys, "frozen", False):  # PyInstaller one-file exe
        return f'"{sys.executable}"'
    return f'"{sys.executable}" -m voicetype'


def apply_autostart(enabled: bool) -> None:
    if sys.platform != "win32":
        return
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_ALL_ACCESS
        ) as key:
            if enabled:
                winreg.SetValueEx(
                    key, _VALUE_NAME, 0, winreg.REG_SZ, _launch_command()
                )
                logger.info("Autostart enabled")
            else:
                try:
                    winreg.DeleteValue(key, _VALUE_NAME)
                    logger.info("Autostart disabled")
                except FileNotFoundError:
                    pass
    except Exception as exc:
        logger.warning("Could not update autostart setting: %s", exc)
