"""Yandex SpeechKit provider (optional, stub).

Best Russian quality with streaming support (spec section 6); Belarusian support
must be verified before relying on it. This is a minimal synchronous REST
implementation against the short-audio recognition endpoint. Requires
``YANDEX_API_KEY`` (and optionally ``YANDEX_FOLDER_ID``) in the environment.

As with any cloud provider, audio leaves the machine — keep ``local_whisper``
for confidential prompts.
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

_RECOGNIZE_URL = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"


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


class YandexSpeechKitProvider(STTProvider):
    def __init__(self, cfg: STTConfig) -> None:
        self.cfg = cfg

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        if audio is None or audio.size == 0:
            return TranscriptionResult(text="")
        try:
            import requests

            api_key = os.environ.get("YANDEX_API_KEY")
            if not api_key:
                raise RuntimeError("YANDEX_API_KEY is not set")
            lang = language or self.cfg.default_language
            # SpeechKit expects BCP-47-ish codes; map our short codes.
            lang_code = {"ru": "ru-RU", "be": "be-BY", "auto": "auto"}.get(lang, "ru-RU")
            params = {"lang": lang_code, "format": "lpcm", "sampleRateHertz": "16000"}
            folder = os.environ.get("YANDEX_FOLDER_ID")
            if folder:
                params["folderId"] = folder
            # lpcm = raw 16-bit PCM without WAV header.
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            resp = requests.post(
                _RECOGNIZE_URL,
                params=params,
                data=pcm,
                headers={"Authorization": f"Api-Key {api_key}"},
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json().get("result", "").strip()
            return TranscriptionResult(text=text, language=lang)
        except Exception as exc:
            logger.error("Yandex transcription failed: %s", exc)
            return TranscriptionResult(text="")
