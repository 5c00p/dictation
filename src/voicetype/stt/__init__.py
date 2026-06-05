"""Speech-to-text providers."""

from __future__ import annotations

from ..config import STTConfig
from .base import STTProvider, TranscriptionResult

__all__ = ["STTProvider", "TranscriptionResult", "create_provider"]


def create_provider(cfg: STTConfig) -> STTProvider:
    """Instantiate the STT provider selected in config (spec section 5)."""
    if cfg.provider == "local_whisper":
        from .local_whisper import LocalWhisperProvider

        return LocalWhisperProvider(cfg)
    if cfg.provider == "openai":
        from .cloud.openai_whisper import OpenAIWhisperProvider

        return OpenAIWhisperProvider(cfg)
    if cfg.provider == "yandex":
        from .cloud.yandex_speechkit import YandexSpeechKitProvider

        return YandexSpeechKitProvider(cfg)
    raise ValueError(f"Unknown STT provider: {cfg.provider!r}")
