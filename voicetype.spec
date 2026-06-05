# PyInstaller spec for VoiceType (spec section 9, iteration 3 item 11).
# Build:  uv sync --extra build  &&  uv run pyinstaller --noconfirm voicetype.spec
#
# faster-whisper / ctranslate2 / onnxruntime ship native libs and data files that
# PyInstaller does not pick up automatically, so we collect them explicitly.

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("faster_whisper", "ctranslate2", "onnxruntime", "tokenizers"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# pynput / pystray backends are imported dynamically.
hiddenimports += [
    "pynput.keyboard._win32",
    "pynput.mouse._win32",
    "pystray._win32",
]


a = Analysis(
    ["src/voicetype/__main__.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="VoiceType",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # tray app, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
