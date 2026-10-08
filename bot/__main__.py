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
