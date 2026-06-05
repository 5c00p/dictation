"""Local STT via faster-whisper (CTranslate2).

Auto-detects CUDA and falls back to CPU int8 (spec section 4). On the target
machine there is no CUDA GPU, so the effective default is CPU + int8 — usable,
though large models will be slow; the model size is configurable.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import STTConfig
from .base import STTProvider, TranscriptionResult

logger = logging.getLogger(__name__)


def _detect_device(requested: str) -> tuple[str, str]:
    """Return ``(device, compute_type)`` resolving the ``auto`` settings."""
    device = requested
    if requested == "auto":
        device = "cpu"
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
        except Exception as exc:  # pragma: no cover - depends on host
            logger.debug("CUDA detection failed, using CPU: %s", exc)
    return device


class LocalWhisperProvider(STTProvider):
    def __init__(self, cfg: STTConfig) -> None:
        self.cfg = cfg
        self._device = _detect_device(cfg.device)
        self._compute_type = self._resolve_compute_type(cfg.compute_type, self._device)
        self._model = None  # lazy — load on first use / warmup

    @staticmethod
    def _resolve_compute_type(requested: str, device: str) -> str:
        if requested != "auto":
            return requested
        return "float16" if device == "cuda" else "int8"

    def _ensure_model(self):
        if self._model is None:
            import os

            from faster_whisper import WhisperModel

            cpu_threads = self.cfg.cpu_threads or (os.cpu_count() or 0)
            logger.info(
                "Loading Whisper model %r on %s (%s, cpu_threads=%s)",
                self.cfg.model,
                self._device,
                self._compute_type,
                cpu_threads,
            )
            self._model = WhisperModel(
                self.cfg.model,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=cpu_threads,
            )
        return self._model

    def warmup(self) -> None:
        model = self._ensure_model()
        try:
            silence = np.zeros(self.cfg_sample_rate, dtype=np.float32)
            list(model.transcribe(silence, language="ru", beam_size=1)[0])
        except Exception as exc:  # pragma: no cover
            logger.debug("Warmup inference skipped: %s", exc)

    @property
    def cfg_sample_rate(self) -> int:
        return 16000

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        if audio is None or audio.size == 0:
            return TranscriptionResult(text="")

        model = self._ensure_model()
        lang = language or self.cfg.default_language
        whisper_lang = None if lang == "auto" else lang

        try:
            segments, info = model.transcribe(
                audio,
                language=whisper_lang,
                beam_size=self.cfg.beam_size,
                initial_prompt=self.cfg.initial_prompt or None,
                vad_filter=True,
            )
            text = "".join(seg.text for seg in segments).strip()
            detected = getattr(info, "language", None)
            return TranscriptionResult(text=text, language=detected)
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            return TranscriptionResult(text="")

    def close(self) -> None:
        self._model = None
