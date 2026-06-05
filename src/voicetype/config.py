"""Configuration loading and validation for VoiceType.

Loads ``config.toml`` (creating it from the bundled defaults on first run) and
validates it with pydantic. All comments / log messages are in English per the
project conventions; user-facing config keys mirror the spec in
``doc/TASK_voice_dictation.md`` section 8.
"""

from __future__ import annotations

import logging
import shutil
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# Where the user config lives. Kept next to the executable / project root so it
# is easy to find and edit by hand.
DEFAULT_CONFIG_NAME = "config.toml"


class STTConfig(BaseModel):
    provider: Literal["local_whisper", "openai", "yandex"] = "local_whisper"
    model: str = "small"  # tiny|base|small|medium|large-v3|turbo
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: Literal["auto", "float16", "int8", "int8_float16", "float32"] = "auto"
    default_language: Literal["ru", "be", "auto"] = "ru"
    initial_prompt: str = (
        "doTERRA, DigestZen, TerraShield, PB Restore, Deep Blue, "
        "FastAPI, asyncpg, Redmine, Nextcloud"
    )
    # Greedy decoding (beam_size=1) is much faster on CPU; raise to 5 for a small
    # quality gain at the cost of speed.
    beam_size: int = 1
    # CPU threads for CTranslate2. 0 = auto (use all logical cores).
    cpu_threads: int = 0


class HotkeysConfig(BaseModel):
    push_to_talk: str = "ctrl+alt"
    toggle: str = "ctrl+d"
    switch_language: str = "ctrl+alt+l"


class AudioConfig(BaseModel):
    input_device: str = ""  # empty = system default device
    sample_rate: int = 16000


class InjectConfig(BaseModel):
    method: Literal["clipboard", "unicode"] = "clipboard"
    restore_clipboard: bool = True
    # Delays (seconds) around the clipboard paste dance — see spec 7.1.
    paste_pre_delay: float = 0.05
    paste_post_delay: float = 0.12


class VADConfig(BaseModel):
    # Used by the toggle (continuous) mode to split speech into chunks.
    threshold: float = 0.5
    min_silence_ms: int = 700
    min_speech_ms: int = 250
    speech_pad_ms: int = 200


class PostprocessConfig(BaseModel):
    capitalize_first: bool = True
    voice_punctuation: bool = True  # "точка" -> "."
    numbers_to_digits: bool = False
    raw_mode: bool = False  # when True, skip all postprocessing (code/logs dictation)


class AppConfig(BaseModel):
    autostart: bool = False
    sound_feedback: bool = True
    log_level: str = "INFO"
    log_transcripts: bool = False  # privacy: never log recognized text by default


class Config(BaseModel):
    stt: STTConfig = Field(default_factory=STTConfig)
    hotkeys: HotkeysConfig = Field(default_factory=HotkeysConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    inject: InjectConfig = Field(default_factory=InjectConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    app: AppConfig = Field(default_factory=AppConfig)


def _project_root() -> Path:
    # src/voicetype/config.py -> project root is three levels up.
    return Path(__file__).resolve().parents[2]


def config_path(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    return _project_root() / DEFAULT_CONFIG_NAME


def ensure_config_exists(path: Path) -> None:
    """Create ``config.toml`` from ``config.toml.example`` if it is missing."""
    if path.exists():
        return
    example = _project_root() / "config.toml.example"
    if example.exists():
        shutil.copyfile(example, path)
        logger.info("Created %s from config.toml.example", path.name)
    else:
        logger.warning("No config found and no example available; using built-in defaults")


def load_config(explicit: str | Path | None = None) -> Config:
    """Load and validate configuration, falling back to defaults on any error."""
    path = config_path(explicit)
    ensure_config_exists(path)

    if not path.exists():
        return Config()

    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        return Config.model_validate(raw)
    except (tomllib.TOMLDecodeError, ValidationError, OSError) as exc:
        logger.error("Failed to load config from %s: %s — using defaults", path, exc)
        return Config()
