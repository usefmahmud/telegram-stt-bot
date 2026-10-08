# Telegram STT Bot — Design Spec

**Date:** 2026-10-08
**Status:** Approved (verbal design review, no open questions)

## Goal

A professional Telegram bot: user sends an audio file or voice message → bot
replies with the transcript and an inline **"✨ Summarize it"** button → pressing
the button sends the transcript to an OpenAI-compatible chat model and replies
with a summary in the same language as the transcript.

## Decisions (locked)

| Topic | Decision |
|---|---|
| Transcription | Local `qwencleo-asr` (`QwenCleoASR`, Egyptian Arabic + code-switching), runs on the VPS |
| LLM | Any OpenAI-compatible endpoint: `OPENAI_BASE_URL` + `OPENAI_API_KEY` + `OPENAI_MODEL` |
| Telegram library | `python-telegram-bot` v21+ (async) |
| Access control | Public — anyone on Telegram can use the bot (guarded by size/duration limits) |
| Inputs | Voice messages + audio documents (mp3, wav, m4a, ogg, oga, ...) |
| Summary language | Same language as the transcript (instructed in the system prompt) |
| Deployment | Docker + docker-compose on the user's VPS |
| Architecture | Option 1: single async process; ASR in a small thread pool; in-memory callback store |
| Testing | **No unit tests** (explicit user instruction) |

## Project structure

```
telegram-bot-stt/
├── bot/
│   ├── __init__.py
│   ├── __main__.py        # entry: load config, wire app, run polling
│   ├── config.py          # env-based settings dataclass; fails fast on missing keys
│   ├── transcriber.py     # lazy QwenCleoASR wrapper + ffmpeg conversion (16 kHz mono wav)
│   ├── summarizer.py      # OpenAI-compatible chat client wrapper
│   ├── handlers.py        # /start, /help, voice + audio, callback query, fallback
│   ├── messages.py        # templates, HTML formatting, 4096-char splitting
│   └── store.py           # in-memory token → transcript store with TTL + size cap
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .dockerignore
├── .gitignore
├── main.py                # legacy scratch (kept)
└── README.md
```

No database. The callback store lives in memory; restart clears pending
"Summarize" buttons (acceptable — the user re-sends the audio).

## Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — (required) | BotFather token |
| `OPENAI_API_KEY` | — (required) | API key for the compatible endpoint |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible base URL |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model name |
| `ASR_DEVICE` | `cpu` | `cpu` / `cuda:0` for QwenCleoASR |
| `ASR_MAX_NEW_TOKENS` | `2048` | Overrides the library default of 256 (avoids truncation) |
| `ASR_LANGUAGE` | `Arabic` | Language hint passed to the model; empty = auto-detect |
| `MAX_FILE_SIZE_MB` | `20` | Telegram bot download cap |
| `MAX_AUDIO_DURATION_SECONDS` | `1800` | ffprobe check before transcription |
| `SUMMARIZE_MAX_CHARS` | `60000` | Transcript truncation before sending to the LLM |
| `STORE_TTL_SECONDS` | `3600` | Lifetime of pending summarize tokens |
| `STORE_MAX_ENTRIES` | `500` | Cap on stored transcripts |

Startup fails fast with a clear error if required vars are missing or `ffmpeg`
is not found.

## Data flow

### Transcription

1. User sends a voice message or audio document within limits.
2. Bot replies immediately: `⏳ Transcribing…`.
3. File downloaded to a temp dir → `ffmpeg` converts to 16 kHz mono WAV →
   `QwenCleoASR.transcribe()` runs in a 2-slot thread pool
   (`asyncio.to_thread` + bounded executor) so the model is used serially-ish
   and the event loop never blocks.
4. Status message is edited to the result:
   - Header: file name / duration / detected language.
   - Transcript body (HTML-escaped), split into ≤ 4096-char messages; the
     **first** message carries the inline `✨ Summarize it` button
     (`callback_data = "sum:<token>"`).
   - Further chunks are sent as replies to the first message.
5. Temp files removed in `finally`.

### Summarization

1. Callback received → answer callback query (loading state on the button).
2. Ownership rule: the store keeps the `user_id` that sent the audio; a press
   from anyone else gets *"Not yours 😄"* (the bot itself is public). Expired or
   unknown token → *"This transcript has expired — send the audio again."*
3. Single-use tokens: marked used on first press (prevents double billing);
   a second press gets the "expired" notice.
4. `summarizer.summarize(text)` → `chat.completions` with system prompt:
   *"Summarize the following transcript. Be concise and keep key facts,
   decisions and action items. Reply in the same language as the transcript."*
   Transcript over `SUMMARIZE_MAX_CHARS` is truncated with a `…[truncated]` marker.
5. Summary sent as a **reply to the transcript message**, split at 4096 chars.
6. On LLM error: user-facing `⚠️ Summarization failed — try again later.`,
   token restored so retry works, traceback logged.

### Other handlers

- `/start` — welcome + usage; `/help` — formats and limits.
- Non-audio content → polite hint listing accepted inputs.
- All handler errors caught: short user-facing message, full traceback in logs.

## Limits & error handling

- Oversized file / too-long duration → rejection message stating the limit.
- ffmpeg or model failure → `⚠️ Couldn't transcribe this file…` (edited into
  the status message).
- Missing env vars / missing ffmpeg at startup → exit with a clear message.
- Telegram network hiccups → PTB built-in HTTP retries + polling retry.
- LLM timeout / 4xx-5xx → user-facing failure message, details to logs.

## Logging

Stdout `logging` (Docker-friendly). Log metadata only — user id, file size,
durations, latencies, errors — never message content (privacy).

## Deployment

- `Dockerfile`: `python:3.14-slim` + `ffmpeg` + `requirements.txt`
  (CPU torch wheels pulled automatically on linux); entrypoint `python -m bot`.
- `docker-compose.yml`: single `bot` service, `.env` via `env_file`, named
  volume for the Hugging Face model cache (`/root/.cache/huggingface`) so the
  model downloads once.
- `README.md`: BotFather token setup, env vars, `docker compose up -d`,
  logs via `docker compose logs -f`.

## Out of scope

- Unit/integration tests (explicitly excluded).
- Persistence (DB), multi-worker scaling, vLLM backend (future swap possible
  behind `Transcriber`), admin stats, per-user access control.
