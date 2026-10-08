"""Environment-based configuration with fail-fast validation."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from dotenv import load_dotenv


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
        load_dotenv()
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
