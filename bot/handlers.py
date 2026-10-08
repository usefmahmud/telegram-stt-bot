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
            logger.exception(
                "unexpected error handling audio from %s", update.effective_user.id
            )
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
    elif query.message is not None:
        reply_to = await query.message.reply_text(chunks[0])
    else:
        return
    for chunk in chunks[1:]:
        reply_to = await reply_to.reply_text(chunk)
    logger.info("summarized user=%s chars=%d out=%d", user_id, len(text), len(summary))


async def handle_fallback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    await message.reply_text(m.NOT_AUDIO_HINT, parse_mode=ParseMode.HTML)
