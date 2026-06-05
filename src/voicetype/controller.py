"""Orchestration: hotkey -> capture -> STT -> postprocess -> inject (spec 5).

Holds the runtime state and wires the pieces together. Hotkey callbacks run on
the pynput listener thread, so any heavy work (transcription) is dispatched to
worker threads to keep key handling responsive. A single inference lock
serialises transcription + injection so chunks never interleave.
"""

from __future__ import annotations

import enum
import logging
import sys
import threading
from pathlib import Path
from queue import Empty
from typing import Optional

import numpy as np

from .audio.capture import PushToTalkRecorder, StreamingRecorder, resolve_device
from .audio.vad import VoiceSegmenter
from .config import Config, load_config
from .hotkeys import HotkeyManager
from .inject import create_injector
from .postprocess import postprocess
from .stt import create_provider

logger = logging.getLogger(__name__)

_WIN = sys.platform == "win32"
if _WIN:
    import winsound


class State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"


class Controller:
    def __init__(self, config: Config, config_path: Path) -> None:
        self.config = config
        self.config_path = config_path

        self.language = config.stt.default_language
        self.input_device_name = config.audio.input_device

        self._provider = create_provider(config.stt)
        self._injector = create_injector(config.inject)

        self._state = State.IDLE
        self._tray = None  # set via set_tray
        self._hotkeys: Optional[HotkeyManager] = None

        self._infer_lock = threading.Lock()

        # Push-to-talk
        self._ptt_recorder: Optional[PushToTalkRecorder] = None

        # Toggle / streaming
        self._toggle_running = False
        self._toggle_thread: Optional[threading.Thread] = None
        self._stream_recorder: Optional[StreamingRecorder] = None

    # -- wiring -----------------------------------------------------------
    def set_tray(self, tray) -> None:  # noqa: ANN001
        self._tray = tray

    def warmup(self) -> None:
        try:
            self._provider.warmup()
        except Exception as exc:  # pragma: no cover
            logger.debug("Provider warmup skipped: %s", exc)

    # -- hotkeys ----------------------------------------------------------
    def start_hotkeys(self) -> None:
        """(Re)build and start the global hotkey listener from current config."""
        if self._hotkeys is not None:
            self._hotkeys.stop()
        self._hotkeys = HotkeyManager(
            push_to_talk=self.config.hotkeys.push_to_talk,
            toggle=self.config.hotkeys.toggle,
            switch_language=self.config.hotkeys.switch_language,
            on_ptt_start=self.on_ptt_start,
            on_ptt_stop=self.on_ptt_stop,
            on_toggle=self.on_toggle,
            on_switch_language=self.switch_language,
        )
        self._hotkeys.start()

    def stop_hotkeys(self) -> None:
        if self._hotkeys is not None:
            self._hotkeys.stop()
            self._hotkeys = None

    # -- live config reload ----------------------------------------------
    def reload(self) -> None:
        """Reload config.toml and apply changes without restarting the app.

        Postprocess/VAD/audio settings are read live from ``self.config`` on each
        use, so they apply immediately. Hotkeys are re-registered. The STT
        provider and injector are rebuilt only if their settings changed (model
        reload runs in the background since it can be slow)."""
        logger.info("Reloading configuration")
        old = self.config
        new = load_config(self.config_path)
        self.config = new

        if new.inject != old.inject:
            self._injector = create_injector(new.inject)

        stt_changed = new.stt != old.stt
        if stt_changed:
            self.language = new.stt.default_language
            self._set_state(State.PROCESSING)

            def _reload_model() -> None:
                try:
                    old_provider = self._provider
                    provider = create_provider(new.stt)
                    provider.warmup()
                    self._provider = provider
                    try:
                        old_provider.close()
                    except Exception:
                        pass
                    logger.info("STT provider reloaded (model=%s)", new.stt.model)
                except Exception as exc:
                    logger.error("Failed to reload STT provider: %s", exc)
                finally:
                    self._set_state(State.IDLE)

            threading.Thread(target=_reload_model, daemon=True).start()

        if new.hotkeys != old.hotkeys:
            self.start_hotkeys()

        if self._tray is not None:
            self._tray.set_state(self._state.value)

    # -- state ------------------------------------------------------------
    @property
    def state_name(self) -> str:
        return self._state.value

    def _set_state(self, state: State) -> None:
        self._state = state
        if self._tray is not None:
            self._tray.set_state(state.value)

    def _device_index(self) -> Optional[int]:
        return resolve_device(self.input_device_name)

    # -- sound feedback ---------------------------------------------------
    def _beep(self, start: bool) -> None:
        if not self.config.app.sound_feedback or not _WIN:
            return
        try:
            freq = 880 if start else 520
            winsound.Beep(freq, 90)
        except Exception:
            pass

    # =====================================================================
    # Push-to-talk
    # =====================================================================
    def on_ptt_start(self) -> None:
        if self._toggle_running:
            return  # don't mix modes
        if self._ptt_recorder is not None:
            return
        logger.info("Push-to-talk start")
        self._ptt_recorder = PushToTalkRecorder(
            sample_rate=self.config.audio.sample_rate,
            device=self._device_index(),
        )
        self._ptt_recorder.start()
        self._set_state(State.RECORDING)
        self._beep(start=True)

    def on_ptt_stop(self) -> None:
        recorder, self._ptt_recorder = self._ptt_recorder, None
        if recorder is None:
            return
        logger.info("Push-to-talk stop")
        self._beep(start=False)
        audio = recorder.stop()
        self._set_state(State.PROCESSING)
        threading.Thread(
            target=self._transcribe_and_inject,
            args=(audio,),
            kwargs={"return_state": State.IDLE},
            daemon=True,
        ).start()

    # =====================================================================
    # Toggle (continuous, VAD-chunked)
    # =====================================================================
    def on_toggle(self) -> None:
        if self._toggle_running:
            self._stop_toggle()
        else:
            self._start_toggle()

    def _start_toggle(self) -> None:
        if self._ptt_recorder is not None:
            return
        logger.info("Toggle dictation start")
        self._toggle_running = True
        self._set_state(State.RECORDING)
        self._beep(start=True)
        self._toggle_thread = threading.Thread(target=self._toggle_loop, daemon=True)
        self._toggle_thread.start()

    def _stop_toggle(self) -> None:
        logger.info("Toggle dictation stop")
        self._toggle_running = False
        self._beep(start=False)
        # The loop thread will flush remaining audio and reset state to idle.

    def _toggle_loop(self) -> None:
        rec = StreamingRecorder(
            sample_rate=self.config.audio.sample_rate,
            device=self._device_index(),
        )
        seg = VoiceSegmenter(
            sample_rate=self.config.audio.sample_rate,
            threshold=self.config.vad.threshold,
            min_silence_ms=self.config.vad.min_silence_ms,
            min_speech_ms=self.config.vad.min_speech_ms,
            speech_pad_ms=self.config.vad.speech_pad_ms,
        )
        self._stream_recorder = rec
        rec.start()
        try:
            while self._toggle_running:
                try:
                    frame = rec.frames.get(timeout=0.1)
                except Empty:
                    continue
                utterance = seg.feed(frame)
                if utterance is not None:
                    self._transcribe_and_inject(utterance)
            # Stopped: flush any buffered speech.
            tail = seg.flush()
            if tail is not None:
                self._transcribe_and_inject(tail)
        finally:
            rec.stop()
            self._stream_recorder = None
            self._set_state(State.IDLE)

    # =====================================================================
    # Shared inference + injection
    # =====================================================================
    def _transcribe_and_inject(
        self, audio: np.ndarray, return_state: Optional[State] = None
    ) -> None:
        try:
            if audio is None or audio.size == 0:
                return
            with self._infer_lock:
                if self._state != State.RECORDING:
                    self._set_state(State.PROCESSING)
                result = self._provider.transcribe(audio, language=self.language)
                text = postprocess(result.text, self.config.postprocess)
                if self.config.app.log_transcripts:
                    logger.info("Transcript (%s): %s", result.language, text)
                else:
                    logger.info(
                        "Transcribed %d chars (lang=%s)", len(text), result.language
                    )
                if text:
                    self._injector.inject(text)
        except Exception as exc:
            logger.exception("Transcription/injection failed: %s", exc)
            self._set_state(State.ERROR)
        finally:
            if return_state is not None and not self._toggle_running:
                self._set_state(return_state)
            elif not self._toggle_running and self._ptt_recorder is None:
                self._set_state(State.IDLE)

    # =====================================================================
    # Language / device controls
    # =====================================================================
    def set_language(self, code: str) -> None:
        if code in ("ru", "be", "auto"):
            self.language = code
            logger.info("Language set to %s", code)
            if self._tray is not None:
                self._tray.set_state(self._state.value)

    def switch_language(self) -> None:
        # Hotkey cycles RU -> BE -> RU (auto stays only via tray menu).
        self.set_language("be" if self.language != "be" else "ru")

    def set_input_device(self, name: str) -> None:
        self.input_device_name = name
        logger.info("Input device set to %r", name or "(default)")
        if self._tray is not None:
            self._tray.set_state(self._state.value)

    # =====================================================================
    def shutdown(self) -> None:
        self._toggle_running = False
        self.stop_hotkeys()
        if self._ptt_recorder is not None:
            try:
                self._ptt_recorder.stop()
            except Exception:
                pass
            self._ptt_recorder = None
        try:
            self._provider.close()
        except Exception:
            pass
