"""OpenAI Whisper API provider (optional).

Sends a complete utterance as a WAV to the transcription endpoint. No streaming
(spec section 6). Requires the ``cloud`` extra (``openai``) and an API key in the
``OPENAI_API_KEY`` environment variable. Note: using this sends audio to the
cloud — do not use it for confidential prompts (spec section 2 privacy note).
"""

from __future__ import annotations

import io
import logging
import os
import wave
from typing import Optional

import numpy as np

from ...config import STTConfig
from ..base import STTProvider, TranscriptionResult

logger = logging.getLogger(__name__)


def _to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class OpenAIWhisperProvider(STTProvider):
    def __init__(self, cfg: STTConfig) -> None:
        self.cfg = cfg
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = OpenAI()
        return self._client

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        if audio is None or audio.size == 0:
            return TranscriptionResult(text="")
        try:
            client = self._ensure_client()
            lang = language or self.cfg.default_language
            wav = _to_wav_bytes(audio)
            file_obj = ("audio.wav", wav, "audio/wav")
            kwargs = {"model": "whisper-1", "file": file_obj}
            if lang != "auto":
                kwargs["language"] = lang
            if self.cfg.initial_prompt:
                kwargs["prompt"] = self.cfg.initial_prompt
            resp = client.audio.transcriptions.create(**kwargs)
            return TranscriptionResult(text=(resp.text or "").strip(), language=lang)
        except Exception as exc:
            logger.error("OpenAI transcription failed: %s", exc)
            return TranscriptionResult(text="")
