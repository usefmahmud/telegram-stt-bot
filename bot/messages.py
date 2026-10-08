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
