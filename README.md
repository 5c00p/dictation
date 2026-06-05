# VoiceType

System-wide speech-to-text dictation for Windows. Hold a hotkey (or toggle), speak
Russian or Belarusian, and the recognized text is pasted at the caret in **any**
application — Claude Desktop, VS Code, browsers, Telegram, 1C:EDT, Notepad.

Local-first (faster-whisper) so confidential prompts never leave the machine.
Optional cloud providers (OpenAI, Yandex SpeechKit) are available behind a flag.

See [doc/TASK_voice_dictation.md](doc/TASK_voice_dictation.md) for the full spec.

## Requirements

- Windows 10/11 x64
- Python 3.12 + [uv](https://docs.astral.sh/uv/)
- No CUDA GPU is required — runs on CPU with int8 (this machine has Intel Iris Xe,
  so CPU is the default). A CUDA GPU is auto-detected and used if present.

## Install

```powershell
uv sync
```

The first run downloads the Whisper model (large-v3 ≈ 1.5 GB). On CPU, `large-v3`
is accurate but slow — switch `model` to `small` or `medium` in `config.toml` for
lower latency.

## Run

```powershell
uv run voicetype
# or
uv run python -m voicetype
```

A microphone icon appears in the system tray:

- **idle** — grey · **recording** — red · **processing** — yellow

### Hotkeys (default)

| Action | Hotkey | Behaviour |
|--------|--------|-----------|
| Push-to-talk | hold `Ctrl+Alt` | speak while held, release to insert |
| Toggle dictation | `Ctrl+Alt+Space` | continuous; inserts chunks on pauses (VAD) |
| Switch language | `Ctrl+Alt+L` | cycle RU ↔ BE |

All hotkeys, language, device, injection method and postprocessing are configured
in `config.toml` (created from `config.toml.example` on first run) or via the tray
menu.

## Voice punctuation

With `voice_punctuation = true`, say commands such as: `точка` → `.`,
`запятая` → `,`, `новая строка` → newline, `знак вопроса` → `?` (RU + BE words
supported). Set `raw_mode = true` to dictate code/logs without any postprocessing.

## Text injection

- **clipboard** (default): saves your clipboard, pastes via Ctrl+V, restores it —
  most reliable for Unicode in Electron/browsers/1C:EDT.
- **unicode**: per-character `SendInput` fallback for fields where paste is blocked.

## Cloud providers (optional)

Install the extra and set the relevant API key, then set `provider` in config:

```powershell
uv sync --extra cloud
$env:OPENAI_API_KEY = "..."     # provider = "openai"
$env:YANDEX_API_KEY = "..."     # provider = "yandex"
```

> Cloud providers send audio off the machine — keep `local_whisper` for
> confidential prompts.

## Build a standalone .exe

```powershell
uv sync --extra build
uv run pyinstaller --noconfirm --onefile --noconsole --name VoiceType src/voicetype/__main__.py
```

Set `autostart = true` in `config.toml` to register the app in the HKCU `Run` key
so it launches at login.

## Project layout

```
src/voicetype/
├─ __main__.py        entry point (tray + hotkeys)
├─ config.py          config.toml loading/validation (pydantic)
├─ hotkeys.py         global hotkeys, push-to-talk hold detection
├─ controller.py      orchestration: hotkey → capture → STT → inject
├─ audio/             capture.py (sounddevice), vad.py (segmenter)
├─ stt/               base.py, local_whisper.py, cloud/{openai,yandex}
├─ inject/            clipboard.py (primary), unicode_input.py (fallback)
├─ postprocess.py     punctuation, capitalization, numbers
├─ tray.py            tray icon, menu, status colour
└─ autostart.py       Windows Run-key autostart
```
