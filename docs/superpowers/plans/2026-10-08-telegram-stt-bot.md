# Telegram STT Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Telegram bot that transcribes voice/audio messages with the local QwenCleo ASR model and offers an inline "Summarize it" button that calls an OpenAI-compatible chat endpoint.

**Architecture:** Single async `python-telegram-bot` process. Audio is downloaded, converted with ffmpeg to 16 kHz mono WAV, and transcribed in a 2-slot thread pool (model lazy-loaded, inference serialized with a lock). Transcripts are stored in an in-memory token store (TTL + single-use) so the callback button can retrieve them; summarization uses the `openai` SDK pointed at any compatible base URL.

**Tech Stack:** Python 3.14, python-telegram-bot 22.x, openai SDK, qwencleo-asr 0.2.1, ffmpeg, Docker (python:3.14-slim).

**Spec:** `docs/superpowers/specs/2026-10-08-telegram-stt-bot-design.md`

## Global Constraints

- **No unit tests / no test suite** — explicit user instruction. Verify each task with smoke commands (imports, small script runs), never with pytest.
- **Commit every step** with commitlint-style Conventional Commits (`type: lowercase imperative subject`, no period, ≤72 chars).
- No code comments; minimal one-line docstrings only.
- User-facing Telegram messages never contain raw HTML unescaped user content; transcripts are `html.escape`d before sending with `ParseMode.HTML`.
- Env config only — no hardcoded tokens/keys. Fail fast at startup on missing `TELEGRAM_BOT_TOKEN` / `OPENAI_API_KEY` or missing `ffmpeg`.
- Repo root = `/Users/usefmahmud/Documents/programming/python/telegram-bot-stt`; use `.venv/bin/python` for all commands.
- Spec defaults that must match code: `ASR_MAX_NEW_TOKENS=2048`, `MAX_FILE_SIZE_MB=20`, `MAX_AUDIO_DURATION_SECONDS=1800`, `SUMMARIZE_MAX_CHARS=60000`, `STORE_TTL_SECONDS=3600`, `STORE_MAX_ENTRIES=500`, `OPENAI_BASE_URL=https://api.openai.com/v1`, `OPENAI_MODEL=gpt-4o-mini`, `ASR_DEVICE=cpu`, `ASR_LANGUAGE=Arabic` (empty = auto-detect).

---

### Task 1: Scaffold project

**Files:**
- Create: `requirements.txt`, `.env.example`, `.dockerignore`, `bot/__init__.py`

**Interfaces:**
- Produces: `bot` package importable as `bot`; dependency set installed in `.venv`.

- [ ] **Step 1: Write files**

`requirements.txt`:
```
python-telegram-bot>=22.0,<23
openai>=1.40,<4
qwencleo-asr==0.2.1
```

`.env.example`:
```
TELEGRAM_BOT_TOKEN=123456789:replace-me
OPENAI_API_KEY=sk-replace-me
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
# ASR_DEVICE=cpu
# ASR_MAX_NEW_TOKENS=2048
# ASR_LANGUAGE=Arabic
# MAX_FILE_SIZE_MB=20
# MAX_AUDIO_DURATION_SECONDS=1800
# SUMMARIZE_MAX_CHARS=60000
# STORE_TTL_SECONDS=3600
# STORE_MAX_ENTRIES=500
```

`.dockerignore`:
```
.venv
.git
docs
__pycache__
*.wav
.env
*.md
```

`bot/__init__.py`:
```python
"""Telegram STT bot package."""
```

- [ ] **Step 2: Install dependencies into the venv**

Run: `.venv/bin/pip install "python-telegram-bot>=22.0,<23" "openai>=1.40,<4"`
Expected: installs python-telegram-bot 22.x and openai; qwencleo-asr already present.

- [ ] **Step 3: Smoke — package imports**

Run: `.venv/bin/python -c "import bot; import telegram; import openai; print(telegram.__version__, openai.__version__)"`
Expected: prints `22.x.x 3.x.x` (no traceback).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt .env.example .dockerignore bot/__init__.py
git commit -m "chore: scaffold bot package with requirements and env template"
```

---

### Task 2: Configuration (`bot/config.py`)

**Files:**
- Create: `bot/config.py`

**Interfaces:**
- Produces: `class ConfigError(RuntimeError)`; `@dataclass(frozen=True) class Config` with fields `telegram_bot_token: str`, `openai_api_key: str`, `openai_base_url: str`, `openai_model: str`, `asr_device: str`, `asr_max_new_tokens: int`, `asr_language: str | None`, `max_file_size_mb: int`, `max_audio_duration_seconds: int`, `summarize_max_chars: int`, `store_ttl_seconds: int`, `store_max_entries: int`; classmethod `Config.from_env() -> Config`.

- [ ] **Step 1: Write `bot/config.py`**

```python
"""Environment-based configuration with fail-fast validation."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"missing required environment variable {name}")
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    asr_device: str
    asr_max_new_tokens: int
    asr_language: str | None
    max_file_size_mb: int
    max_audio_duration_seconds: int
    summarize_max_chars: int
    store_ttl_seconds: int
    store_max_entries: int

    @classmethod
    def from_env(cls) -> "Config":
        for tool in ("ffmpeg", "ffprobe"):
            if shutil.which(tool) is None:
                raise ConfigError(f"{tool} not found on PATH")
        language = os.environ.get("ASR_LANGUAGE", "Arabic").strip()
        return cls(
            telegram_bot_token=_require("TELEGRAM_BOT_TOKEN"),
            openai_api_key=_require("OPENAI_API_KEY"),
            openai_base_url=os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ).strip().rstrip("/"),
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip(),
            asr_device=os.environ.get("ASR_DEVICE", "cpu").strip(),
            asr_max_new_tokens=_int_env("ASR_MAX_NEW_TOKENS", 2048),
            asr_language=language or None,
            max_file_size_mb=_int_env("MAX_FILE_SIZE_MB", 20),
            max_audio_duration_seconds=_int_env("MAX_AUDIO_DURATION_SECONDS", 1800),
            summarize_max_chars=_int_env("SUMMARIZE_MAX_CHARS", 60000),
            store_ttl_seconds=_int_env("STORE_TTL_SECONDS", 3600),
            store_max_entries=_int_env("STORE_MAX_ENTRIES", 500),
        )
```

- [ ] **Step 2: Smoke — failure without required vars**

Run:
```bash
.venv/bin/python - <<'EOF'
import os
for key in ("TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY"):
    os.environ.pop(key, None)
from bot.config import Config, ConfigError
try:
    Config.from_env()
except ConfigError as e:
    print("OK:", e)
else:
    raise SystemExit("expected ConfigError")
EOF
```
Expected: `OK: missing required environment variable TELEGRAM_BOT_TOKEN`

- [ ] **Step 3: Smoke — success with required vars**

Run:
```bash
TELEGRAM_BOT_TOKEN=dummy OPENAI_API_KEY=dummy .venv/bin/python - <<'EOF'
from bot.config import Config
c = Config.from_env()
assert c.openai_base_url == "https://api.openai.com/v1"
assert c.asr_max_new_tokens == 2048
assert c.asr_language == "Arabic"
assert c.max_file_size_mb == 20
print("OK")
EOF
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add bot/config.py
git commit -m "feat: add env based configuration with fail fast validation"
```

---

### Task 3: Messages (`bot/messages.py`)

**Files:**
- Create: `bot/messages.py`

**Interfaces:**
- Consumes: nothing.
- Produces: constants `TELEGRAM_LIMIT=4096`, `TRANSCRIBING`, `SUMMARIZING`, `SUMMARIZE_BUTTON`, `TRANSCRIBE_FAILED`, `SUMMARY_FAILED`, `EXPIRED`, `NOT_OWNER`, `NOT_AUDIO_HINT`, `START_TEXT`, `HELP_TEXT`; functions `too_big(limit_mb: int) -> str`, `too_long(limit_seconds: int) -> str`, `split_text(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]`, `format_transcript(name: str, duration: float | None, language: str | None, text: str) -> list[str]`.

- [ ] **Step 1: Write `bot/messages.py`**

```python
"""User-facing message templates and 4096-char splitting."""
from __future__ import annotations

import html

TELEGRAM_LIMIT = 4096

TRANSCRIBING = "⏳ Transcribing…"
SUMMARIZING = "📤 Summarizing…"
SUMMARIZE_BUTTON = "✨ Summarize it"
TRANSCRIBE_FAILED = (
    "⚠️ Couldn't transcribe this file — it may be corrupted "
    "or in an unsupported format."
)
SUMMARY_FAILED = "⚠️ Summarization failed — try again later."
EXPIRED = "This transcript has expired — send the audio again."
NOT_OWNER = "Not yours 😄"
NOT_AUDIO_HINT = (
    "🎤 Send me a <b>voice message</b> or an <b>audio file</b> "
    "(mp3, wav, m4a, ogg, flac…) and I'll transcribe it.\n"
    "Then press <b>✨ Summarize it</b> for an AI summary."
)
START_TEXT = (
    "👋 Hi! Send me a <b>voice message</b> or an <b>audio file</b> and "
    "I'll transcribe it.\n\n"
    "Every transcript comes with a <b>✨ Summarize it</b> button that "
    "summarizes the text with AI, in the same language.\n\n"
    "Type /help for limits and supported formats."
)
HELP_TEXT = (
    "🎧 <b>Supported inputs</b>\n"
    "• Voice messages\n"
    "• Audio files: mp3, wav, m4a, ogg, oga, flac, webm, aac, amr…\n\n"
    "📏 <b>Limits</b>\n"
    "• {size_mb} MB per file\n"
    "• {minutes} minutes per recording\n\n"
    "✨ After a transcript is ready, press <b>Summarize it</b> to get an "
    "AI summary in the same language as the audio."
)


def too_big(limit_mb: int) -> str:
    return f"⚠️ File too big — the limit is {limit_mb} MB."


def too_long(limit_seconds: int) -> str:
    minutes = max(1, round(limit_seconds / 60))
    return f"⚠️ Audio too long — the limit is {minutes} minutes."


def split_text(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    parts: list[str] = []
    current = ""
    for line in text.splitlines():
        while len(line) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:limit])
            line = line[limit:]
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) <= limit:
            current = candidate
        elif current:
            parts.append(current)
            current = line
        else:
            current = line
    if current:
        parts.append(current)
    return parts or [""]


def format_transcript(
    name: str,
    duration: float | None,
    language: str | None,
    text: str,
) -> list[str]:
    dur = f"{round(duration)}s" if duration else "?"
    lang = html.escape(language) if language else "auto"
    header = f"🎧 <b>{html.escape(name)}</b> · {dur} · {lang}"
    chunks = [html.escape(chunk) for chunk in split_text(text)]
    if len(header) + 2 + len(chunks[0]) <= TELEGRAM_LIMIT:
        chunks[0] = f"{header}\n\n{chunks[0]}"
    else:
        chunks.insert(0, header)
    return chunks
```

- [ ] **Step 2: Smoke — splitting and formatting**

Run:
```bash
.venv/bin/python - <<'EOF'
from bot.messages import split_text, format_transcript, TELEGRAM_LIMIT

assert split_text("") == [""]
assert split_text("a\nb") == ["a\nb"]
long = ("word " * 5000).strip()
chunks = split_text(long)
assert all(len(c) <= TELEGRAM_LIMIT for c in chunks)
assert "".join(chunks) == long
lines = "\n".join(f"line {i}" for i in range(3000))
chunks = split_text(lines)
assert all(len(c) <= TELEGRAM_LIMIT for c in chunks)
assert "\n".join(chunks) == lines
out = format_transcript("my <file>.mp3", 12.4, "Arabic", "hello <world> & co")
assert out[0].startswith("🎧 <b>my &lt;file&gt;.mp3</b> · 12s · Arabic")
assert "&lt;world&gt;" in out[0] and "<world>" not in out[0]
print("OK")
EOF
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bot/messages.py
git commit -m "feat: add message templates and text splitting utilities"
```

---

### Task 4: Transcript store (`bot/store.py`)

**Files:**
- Create: `bot/store.py`

**Interfaces:**
- Produces: `class TranscriptStore` with `__init__(self, ttl_seconds: int = 3600, max_entries: int = 500)`, `put(text: str, owner_id: int) -> str` (returns token), `claim(token: str, user_id: int) -> tuple[str | None, str | None]` — `(text, None)` on success, `(None, "expired" | "used" | "not_owner")` on failure — and `release(token: str, text: str, owner_id: int) -> None` (restores after a failed summarize).

- [ ] **Step 1: Write `bot/store.py`**

```python
"""In-memory token store for pending summarize callbacks."""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass


@dataclass
class _Entry:
    text: str
    owner_id: int
    expires_at: float
    used: bool = False


class TranscriptStore:
    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 500) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._entries: dict[str, _Entry] = {}

    def put(self, text: str, owner_id: int) -> str:
        self._purge()
        while len(self._entries) >= self._max:
            self._entries.pop(next(iter(self._entries)))
        token = secrets.token_urlsafe(9)
        self._entries[token] = _Entry(
            text=text,
            owner_id=owner_id,
            expires_at=time.monotonic() + self._ttl,
        )
        return token

    def claim(self, token: str, user_id: int) -> tuple[str | None, str | None]:
        self._purge()
        entry = self._entries.get(token)
        if entry is None:
            return None, "expired"
        if entry.used:
            del self._entries[token]
            return None, "used"
        if entry.owner_id != user_id:
            return None, "not_owner"
        entry.used = True
        return entry.text, None

    def release(self, token: str, text: str, owner_id: int) -> None:
        self._entries[token] = _Entry(
            text=text,
            owner_id=owner_id,
            expires_at=time.monotonic() + self._ttl,
        )

    def _purge(self) -> None:
        now = time.monotonic()
        for token in [t for t, e in self._entries.items() if e.expires_at <= now]:
            del self._entries[token]
```

- [ ] **Step 2: Smoke — lifecycle**

Run:
```bash
.venv/bin/python - <<'EOF'
import time
from bot.store import TranscriptStore

s = TranscriptStore(ttl_seconds=1, max_entries=3)
tok = s.put("hello world", owner_id=111)
text, reason = s.claim(tok, 111)
assert (text, reason) == ("hello world", None)
assert s.claim(tok, 111) == (None, "used")
tok2 = s.put("x", 1)
assert s.claim(tok2, 999) == (None, "not_owner")
s.release(tok2, "x", 1)
assert s.claim(tok2, 1) == ("x", None)
tok3 = s.put("y", 1)
time.sleep(1.05)
assert s.claim(tok3, 1) == (None, "expired")
for i in range(5):
    s.put(f"m{i}", 1)
assert len(s._entries) <= 3
print("OK")
EOF
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bot/store.py
git commit -m "feat: add in memory transcript store with ttl and single use tokens"
```

---

### Task 5: Transcriber (`bot/transcriber.py`)

**Files:**
- Create: `bot/transcriber.py`

**Interfaces:**
- Consumes: `qwencleo_asr.QwenCleoASR` (`QwenCleoASR(device=..., max_new_tokens=..., default_language=...)`; `.transcribe(path, language=...) -> TranscriptionResult` with `.text`, `.language`).
- Produces: `class TranscriptionError(RuntimeError)`; `probe_duration(path: Path) -> float | None`; `convert_to_wav(src: Path, dst: Path) -> None` (raises `TranscriptionError`); `class Transcriber.__init__(device="cpu", max_new_tokens=2048, language="Arabic", max_workers=2)` with async `transcribe(src: Path) -> tuple[str, str | None]` returning `(text, language)`.

- [ ] **Step 1: Write `bot/transcriber.py`**

```python
"""Local QwenCleo ASR behind a thread pool, with ffmpeg conversion."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Raised when ffmpeg conversion or ASR inference fails."""


def probe_duration(path: Path) -> float | None:
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def convert_to_wav(src: Path, dst: Path) -> None:
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(src),
                "-ar", "16000", "-ac", "1",
                str(dst),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TranscriptionError(f"ffmpeg failed: {exc}") from exc
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise TranscriptionError(f"ffmpeg failed: {proc.stderr.strip()[:500]}")


class Transcriber:
    def __init__(
        self,
        device: str = "cpu",
        max_new_tokens: int = 2048,
        language: str | None = "Arabic",
        max_workers: int = 2,
    ) -> None:
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._language = language
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="asr")
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from qwencleo_asr import QwenCleoASR

                    logger.info("loading ASR model on %s", self._device)
                    self._model = QwenCleoASR(
                        device=self._device,
                        max_new_tokens=self._max_new_tokens,
                        default_language=self._language,
                    )
        return self._model

    def _transcribe_blocking(self, src: Path) -> tuple[str, str | None]:
        with tempfile.TemporaryDirectory(prefix="stt-") as tmp:
            wav = Path(tmp) / "audio.wav"
            convert_to_wav(src, wav)
            model = self._ensure_model()
            with self._infer_lock:
                result = model.transcribe(str(wav), language=self._language)
        text = (result.text or "").strip()
        language = getattr(result, "language", None)
        return text, language

    async def transcribe(self, src: Path) -> tuple[str, str | None]:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, self._transcribe_blocking, src)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(str(exc)) from exc
```

- [ ] **Step 2: Smoke — ffmpeg conversion + real transcription**

Run:
```bash
.venv/bin/python - <<'EOF'
import asyncio
import tempfile
from pathlib import Path

from bot.transcriber import Transcriber, TranscriptionError, convert_to_wav, probe_duration

with tempfile.TemporaryDirectory() as td:
    wav = Path(td) / "out.wav"
    convert_to_wav(Path("audio.wav"), wav)
    assert wav.exists() and wav.stat().st_size > 0
    dur = probe_duration(wav)
    assert dur and dur > 0.5, dur
    print("convert OK, duration", dur)
    try:
        convert_to_wav(Path("missing.wav"), Path(td) / "x.wav")
    except TranscriptionError:
        print("error path OK")
    else:
        raise SystemExit("expected TranscriptionError")

t = Transcriber(device="cpu", max_new_tokens=2048, language="Arabic")
text, lang = asyncio.run(t.transcribe(Path("audio.wav")))
assert text, "empty transcript"
print("transcript:", text[:120])
print("language:", lang)
EOF
```
Expected: `convert OK, duration …`, `error path OK`, non-empty transcript text (model is cached; first run may take ~1–2 min on CPU).

- [ ] **Step 3: Commit**

```bash
git add bot/transcriber.py
git commit -m "feat: add qwen asr transcriber with ffmpeg conversion"
```

---

### Task 6: Summarizer (`bot/summarizer.py`)

**Files:**
- Create: `bot/summarizer.py`

**Interfaces:**
- Consumes: `openai.AsyncOpenAI(api_key=..., base_url=...)`, `client.chat.completions.create(model=..., messages=[...], temperature=...)`.
- Produces: `class Summarizer.__init__(api_key: str, base_url: str, model: str, max_chars: int = 60000)` with async `summarize(text: str) -> str` (truncates at `max_chars` with `…[truncated]` marker, raises on empty/failed responses).

- [ ] **Step 1: Write `bot/summarizer.py`**

```python
"""OpenAI-compatible chat client for transcript summaries."""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You summarize speech-to-text transcripts. Summarize the transcript the "
    "user sends. Be concise and keep key facts, decisions and action items. "
    "Reply in the same language as the transcript."
)


class Summarizer:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_chars: int = 60000,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        self._model = model
        self._max_chars = max_chars

    async def summarize(self, text: str) -> str:
        if len(text) > self._max_chars:
            text = text[: self._max_chars] + "\n…[truncated]"
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("empty summary from model")
        return content.strip()
```

- [ ] **Step 2: Smoke — construction and truncation setup**

Run:
```bash
.venv/bin/python - <<'EOF'
from bot.summarizer import Summarizer, SYSTEM_PROMPT
s = Summarizer(api_key="dummy", base_url="https://example.com/v1", model="test-model", max_chars=10)
assert "same language" in SYSTEM_PROMPT
assert s._max_chars == 10 and s._model == "test-model"
print("OK")
EOF
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add bot/summarizer.py
git commit -m "feat: add openai compatible summarizer client"
```

---

### Task 7: Handlers + entrypoint (`bot/handlers.py`, `bot/__main__.py`)

**Files:**
- Create: `bot/handlers.py`, `bot/__main__.py`

**Interfaces:**
- Consumes: `Config.from_env()` (Task 2), message constants + `format_transcript`/`split_text`/`too_big`/`too_long` (Task 3), `TranscriptStore` (Task 4), `Transcriber` + `probe_duration` + `TranscriptionError` (Task 5), `Summarizer` (Task 6).
- Produces: handlers `cmd_start`, `cmd_help`, `handle_audio`, `handle_summarize`, `handle_fallback` — all `(Update, ContextTypes.DEFAULT_TYPE) -> None`; `build_application(config: Config) -> Application` (registers handlers, wires `bot_data`); `main() -> None`.

- [ ] **Step 1: Write `bot/handlers.py`**

```python
"""Telegram handlers: transcription flow and summarize callback."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from . import messages as m
from .config import Config
from .store import TranscriptStore
from .summarizer import Summarizer
from .transcriber import Transcriber, TranscriptionError, probe_duration

logger = logging.getLogger(__name__)

AUDIO_SUFFIXES = {
    ".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus",
    ".flac", ".webm", ".aac", ".amr", ".mka", ".wma",
}


def _media(message):
    if message.voice is not None:
        return message.voice, "voice message"
    if message.audio is not None:
        return message.audio, message.audio.file_name or "audio"
    document = message.document
    if document is not None:
        mime = (document.mime_type or "").lower()
        suffix = Path(document.file_name or "").suffix.lower()
        if mime.startswith("audio/") or suffix in AUDIO_SUFFIXES:
            return document, document.file_name or "audio"
    return None


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message:
        await update.effective_message.reply_text(m.START_TEXT, parse_mode=ParseMode.HTML)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    text = m.HELP_TEXT.format(
        size_mb=config.max_file_size_mb,
        minutes=max(1, round(config.max_audio_duration_seconds / 60)),
    )
    if update.effective_message:
        await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    store: TranscriptStore = context.bot_data["store"]
    transcriber: Transcriber = context.bot_data["transcriber"]
    message = update.effective_message
    if message is None or update.effective_user is None or update.effective_chat is None:
        return

    media = _media(message)
    if media is None:
        await message.reply_text(m.NOT_AUDIO_HINT, parse_mode=ParseMode.HTML)
        return
    source, name = media

    size = getattr(source, "file_size", None) or 0
    if size > config.max_file_size_mb * 1024 * 1024:
        await message.reply_text(m.too_big(config.max_file_size_mb))
        return

    status = await message.reply_text(m.TRANSCRIBING)
    await update.effective_chat.send_action(ChatAction.TYPING)

    with tempfile.TemporaryDirectory(prefix="stt-") as tmp:
        path = Path(tmp) / (Path(name).name or "audio")
        try:
            tg_file = await source.get_file()
            await tg_file.download_to_drive(custom_path=path)
            duration = probe_duration(path) or getattr(source, "duration", None)
            if duration and duration > config.max_audio_duration_seconds:
                await status.edit_text(m.too_long(config.max_audio_duration_seconds))
                return
            text, language = await transcriber.transcribe(path)
        except TranscriptionError:
            logger.exception("transcription failed for %s", name)
            await status.edit_text(m.TRANSCRIBE_FAILED)
            return
        except Exception:
            logger.exception("unexpected error handling audio from %s", update.effective_user.id)
            await status.edit_text(m.TRANSCRIBE_FAILED)
            return

    if not text:
        await status.edit_text(m.TRANSCRIBE_FAILED)
        return

    token = store.put(text, update.effective_user.id)
    chunks = m.format_transcript(name, duration, language, text)
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton(m.SUMMARIZE_BUTTON, callback_data=f"sum:{token}")]]
    )
    await status.edit_text(chunks[0], reply_markup=keyboard, parse_mode=ParseMode.HTML)
    reply_to = status
    for chunk in chunks[1:]:
        reply_to = await reply_to.reply_text(chunk, parse_mode=ParseMode.HTML)
    logger.info(
        "transcribed user=%s name=%s bytes=%s duration=%ss chars=%d",
        update.effective_user.id, name, size,
        round(duration or 0), len(text),
    )


async def handle_summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    store: TranscriptStore = context.bot_data["store"]
    summarizer: Summarizer = context.bot_data["summarizer"]
    if query is None or not query.data or not query.data.startswith("sum:"):
        return
    token = query.data[len("sum:"):]
    user_id = update.effective_user.id if update.effective_user else 0

    text, reason = store.claim(token, user_id)
    if text is None:
        notice = m.NOT_OWNER if reason == "not_owner" else m.EXPIRED
        await query.answer(notice, show_alert=reason != "not_owner")
        return

    await query.answer()
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        logger.debug("could not remove keyboard", exc_info=True)

    status = None
    if query.message is not None:
        status = await query.message.reply_text(m.SUMMARIZING)

    try:
        summary = await summarizer.summarize(text)
    except Exception:
        logger.exception("summarization failed for user=%s", user_id)
        store.release(token, text, user_id)
        if status is not None:
            await status.edit_text(m.SUMMARY_FAILED)
        elif query.message is not None:
            await query.message.reply_text(m.SUMMARY_FAILED)
        return

    chunks = m.split_text(summary)
    if status is not None:
        await status.edit_text(chunks[0])
        reply_to = status
    else:
        reply_to = await query.message.reply_text(chunks[0])
    for chunk in chunks[1:]:
        reply_to = await reply_to.reply_text(chunk)
    logger.info("summarized user=%s chars=%d out=%d", user_id, len(text), len(summary))


async def handle_fallback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    await message.reply_text(m.NOT_AUDIO_HINT, parse_mode=ParseMode.HTML)
```

- [ ] **Step 2: Write `bot/__main__.py`**

```python
"""Entrypoint: `python -m bot`."""
from __future__ import annotations

import logging

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from .config import Config, ConfigError
from .handlers import (
    cmd_help,
    cmd_start,
    handle_audio,
    handle_fallback,
    handle_summarize,
)
from .store import TranscriptStore
from .summarizer import Summarizer
from .transcriber import Transcriber


def build_application(config: Config) -> Application:
    request = HTTPXRequest(
        connect_timeout=15.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=15.0,
    )
    app = Application.builder().token(config.telegram_bot_token).request(request).build()
    app.bot_data["config"] = config
    app.bot_data["store"] = TranscriptStore(
        config.store_ttl_seconds, config.store_max_entries
    )
    app.bot_data["transcriber"] = Transcriber(
        device=config.asr_device,
        max_new_tokens=config.asr_max_new_tokens,
        language=config.asr_language,
    )
    app.bot_data["summarizer"] = Summarizer(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        model=config.openai_model,
        max_chars=config.summarize_max_chars,
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_audio))
    app.add_handler(CallbackQueryHandler(handle_summarize, pattern=r"^sum:"))
    app.add_handler(
        MessageHandler(
            ~filters.COMMAND
            & ~filters.VOICE
            & ~filters.AUDIO
            & ~filters.Document.ALL,
            handle_fallback,
        )
    )
    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        config = Config.from_env()
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}")
    app = build_application(config)
    app.run_polling(allowed_updates=None)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke — application builds with all handlers**

Run:
```bash
TELEGRAM_BOT_TOKEN=dummy OPENAI_API_KEY=dummy .venv/bin/python - <<'EOF'
from bot.config import Config
from bot.__main__ import build_application

app = build_application(Config.from_env())
counts = {type(h).__name__ for h in app.handlers[0]}
assert "CommandHandler" in counts
assert "MessageHandler" in counts
assert "CallbackQueryHandler" in counts
assert set(app.bot_data) == {"config", "store", "transcriber", "summarizer"}
print("OK", sorted(counts))
EOF
```
Expected: `OK ['CallbackQueryHandler', 'CommandHandler', 'MessageHandler']`

- [ ] **Step 4: Smoke — startup fails fast without config**

Run: `.venv/bin/python -m bot; echo "exit=$?"` (with no env vars set)
Expected: prints `Configuration error: missing required environment variable TELEGRAM_BOT_TOKEN` and `exit=1`.

- [ ] **Step 5: Commit**

```bash
git add bot/handlers.py bot/__main__.py
git commit -m "feat: add telegram handlers for transcribe and summarize flow"
```

---

### Task 8: Docker deployment

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`

**Interfaces:**
- Consumes: `requirements.txt`, `bot/` package, entrypoint `python -m bot`, env from `.env`.

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
FROM python:3.14-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY bot/ ./bot/

CMD ["python", "-m", "bot"]
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
services:
  bot:
    build: .
    env_file: .env
    restart: unless-stopped
    volumes:
      - hf-cache:/cache/huggingface

volumes:
  hf-cache:
```

- [ ] **Step 3: Build the image**

Run: `docker compose build`
Expected: builds to completion (torch CPU wheels + ffmpeg; first build several minutes).

- [ ] **Step 4: Verify entrypoint fails fast inside the container**

Run: `docker compose run --rm bot`
Expected: `Configuration error: missing required environment variable TELEGRAM_BOT_TOKEN`, exit code 1.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: add docker deployment with compose"
```

---

### Task 9: README + final verification

**Files:**
- Create: `README.md`
- Modify: none (verification only)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write `README.md`**

```markdown
# Telegram STT Bot

Telegram bot that transcribes voice messages and audio files locally with
**QwenCleo-ASR** (Egyptian Arabic + English code-switching) and summarizes
the transcript with any OpenAI-compatible chat model via an inline
**✨ Summarize it** button.

## Setup

1. Create a bot with [@BotFather](https://t.me/BotFather) → copy the token.
2. Copy the env template and fill it in:

   ```bash
   cp .env.example .env
   ```

   | Variable | Required | Default | Purpose |
   |---|---|---|---|
   | `TELEGRAM_BOT_TOKEN` | yes | — | BotFather token |
   | `OPENAI_API_KEY` | yes | — | key for the compatible endpoint |
   | `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` | OpenAI / OpenRouter / Groq / Ollama… |
   | `OPENAI_MODEL` | no | `gpt-4o-mini` | chat model name |
   | `ASR_DEVICE` | no | `cpu` | `cpu` or `cuda:0` |
   | `ASR_MAX_NEW_TOKENS` | no | `2048` | transcript length budget |
   | `ASR_LANGUAGE` | no | `Arabic` | empty = auto-detect |
   | `MAX_FILE_SIZE_MB` | no | `20` | max upload size |
   | `MAX_AUDIO_DURATION_SECONDS` | no | `1800` | max recording length |
   | `SUMMARIZE_MAX_CHARS` | no | `60000` | transcript truncation before LLM |
   | `STORE_TTL_SECONDS` | no | `3600` | summarize button lifetime |
   | `STORE_MAX_ENTRIES` | no | `500` | pending transcripts kept |

3. Run:

   ```bash
   docker compose up -d --build
   docker compose logs -f
   ```

The ASR model (~3.5 GB) is downloaded on first start into the `hf-cache`
volume, so later restarts are fast.

## Usage

Send the bot a **voice message** or an **audio file** (mp3, wav, m4a, ogg,
flac…). It replies with the transcript and a **✨ Summarize it** button; the
summary comes back as a reply, in the same language as the transcript.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in
.venv/bin/python -m bot
```

Requires `ffmpeg` on PATH. Spec: `docs/superpowers/specs/2026-10-08-telegram-stt-bot-design.md`.
```

- [ ] **Step 2: Final verification — compile everything**

Run: `.venv/bin/python -m compileall -q bot && echo COMPILE_OK`
Expected: `COMPILE_OK`

- [ ] **Step 3: Final verification — startup path without config**

Run: `env -u TELEGRAM_BOT_TOKEN -u OPENAI_API_KEY .venv/bin/python -m bot; echo exit=$?`
Expected: `Configuration error: …` and `exit=1`

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add readme with setup and usage instructions"
```
